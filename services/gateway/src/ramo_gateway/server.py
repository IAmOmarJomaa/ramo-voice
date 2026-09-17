"""
ramo_gateway.server
===================
Sovereign Pure-Python Real-Time Voice Gateway (Port 50000).
100% Bridge-Tauri Compatible WebSocket Protocol & OpenAI REST Endpoints.
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .session import GatewaySession, SessionStore
from .chronos import ChronosBuffer, ChronosCutType
from .pipeline_client import PipelineDispatcher
from .state_machine import ConversationStateMachine, SessionState
from ramo_common.logging import setup_service_logging, tail_service_log, global_log_hub

logger = setup_service_logging("ramo_gateway")

session_store = SessionStore()
dispatcher = PipelineDispatcher(sample_rate=16000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ramo_gateway server (Port 50000)...")
    await dispatcher.initialize()
    logger.info("ramO Sovereign Voice Gateway ready for Bridge-Tauri connections.")
    yield


app = FastAPI(
    title="ramO Voice Gateway API",
    version="0.1.0",
    description="Sovereign Duplex Real-Time Voice Gateway with Barge-in and Bridge-Tauri Wire Compatibility",
    lifespan=lifespan,
)


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "service": "ramo-gateway",
        "port": int(os.getenv("PORT", 50000)),
        "active_sessions": session_store.count(),
        "vad_state": "active",
        "gateway": "ramO V4 Enterprise Gateway (Pure Python)",
    }


@app.get("/logs")
async def get_logs(tail: int = 100):
    return {
        "service": "ramo_gateway",
        "lines": tail_service_log("ramo_gateway", n=tail),
    }


@app.websocket("/v1/logs/stream")
async def stream_logs(websocket: WebSocket):
    """Real-time live WebSocket log streamer for UI observability dashboards."""
    await websocket.accept()
    queue = global_log_hub.subscribe()
    try:
        while True:
            item = await queue.get()
            await websocket.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        global_log_hub.unsubscribe(queue)


# -----------------------------------------------------------------------------
# WebSocket: Bridge-Tauri Protocol (/v1/stream & /v1/realtime)
# -----------------------------------------------------------------------------

async def _safe_send_json(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except Exception as e:
        logger.debug(f"Failed to send JSON (client disconnected): {e}")
        return False


async def _handle_audio_cut(
    websocket: WebSocket,
    sess: GatewaySession,
    sm: ConversationStateMachine,
    pcm_bytes: bytes,
    is_final: bool,
    trace_id: str,
):
    """Processes a discrete speech chunk through the full neural pipeline."""
    duration_s = len(pcm_bytes) / 32000.0
    logger.info(
        f"✂️ [CHRONOS] Trace: {trace_id} | Cut: {'FINAL' if is_final else 'PROVISIONAL'} "
        f"| Duration: {duration_s:.2f}s ({len(pcm_bytes)} bytes) | Source: {sess.source}"
    )

    # 0. Audio cleaning
    audio_f32, _ = dispatcher.clean_audio_pcm(pcm_bytes)
    logger.debug(f"🧼 [CLEAN] Filtered {len(audio_f32)} samples (16kHz float32) via HPF/AGC")

    # 1. State machine barge-in check
    interrupted = sm.on_speech_start()
    if interrupted:
        logger.info(f"⚡ [BARGE_IN] Speech detected during assistant turn — sending interrupt")
        await _safe_send_json(websocket, {"type": "interrupt"})

    # 2. Overlap / Crosstalk Detection
    is_overlap, overlap_score = dispatcher.detect_overlap(audio_f32)
    if is_overlap:
        logger.info(f"👥 [CROSSTALK] Detected simultaneous speech (score: {overlap_score:.2f})")

    # 3. Speaker identification & inheritance
    last_spk = sess.get_last_known_speaker()
    speaker_id = dispatcher.identify_speaker(audio_f32, last_known=last_spk, is_overlap=is_overlap)
    sess.set_last_known_speaker(speaker_id)
    is_owner = sess.source == "mic"
    logger.info(f"👥 [DIAR_OUT] Speaker: {speaker_id} | Owner: {is_owner} | Overlap: {is_overlap}")

    # 4. Speech-to-text (Faster-Whisper) with acoustic prompt continuation
    prompt = sess.get_stt_prompt() if is_final else ""
    stt_res = await dispatcher.process_stt(audio_f32, initial_prompt=prompt if prompt else None)
    raw_transcript = stt_res.get("raw_text", "").strip()
    words_list = stt_res.get("words", [])

    if not raw_transcript:
        if is_final:
            sess.advance_chunk_seq()
        return

    # Apply n-gram boundary deduplication on finalized chunks
    if is_final:
        transcript_text, words_list = sess.deduplicate_transcript(raw_transcript, words_list)
        if not transcript_text:
            logger.info("✂️ [DEDUP] Entire chunk was redundant overlap tail — skipping re-emission")
            sess.advance_chunk_seq()
            return
    else:
        transcript_text = raw_transcript

    # 5. Voice Harvesting (Strict Single-Speaker Monologue Rule)
    if is_final and not is_overlap:
        dispatcher.harvest_speech_turn(
            speaker_id=speaker_id,
            audio_f32=audio_f32,
            transcript=transcript_text,
            is_overlap=False,
        )
    elif is_overlap:
        logger.info(f"🚫 [HARVEST_SKIP] Excluded turn for '{speaker_id}' from cloning pool due to crosstalk")

    emotion_tag = stt_res.get("emotion", "<|NEUTRAL|>")
    stt_lang = stt_res.get("language", "en")
    word_count = len(words_list)
    logger.info(
        f"👂 [STT_OUT] '{transcript_text}' | Lang: {stt_lang} | Words: {word_count} | Emotion: {emotion_tag}"
    )

    # 4. Emit transcript frame (matching Bridge-Tauri)
    timestamp_str = time.strftime("%I:%M %p").lstrip("0")
    transcript_frame = {
        "type": "transcript",
        "chunk_id": trace_id,
        "text": transcript_text,
        "speaker": speaker_id,
        "is_owner": is_owner,
        "is_final": is_final,
        "language": stt_lang,
        "source": sess.source,
        "words": words_list,
        "timestamp": timestamp_str,
        "emotion": emotion_tag,
        "event": "<|Speech|>",
    }
    await _safe_send_json(websocket, transcript_frame)

    if not is_final:
        return

    sess.advance_chunk_seq()

    # Record dialogue turn into session memory
    sess.dialogue_history.append(f"{speaker_id}: {transcript_text}")

    # 5. Translation
    target_lang = sess.target_language
    source_lang = stt_lang
    translated_text, is_bypass, llm_action = await dispatcher.translate_text(
        text=transcript_text,
        source_lang=source_lang,
        target_lang=target_lang,
        session_id=sess.session_id,
        speaker_id=speaker_id,
    )

    if translated_text:
        logger.info(
            f"🌐 [TRANS_OUT] [{source_lang} -> {target_lang}] '{transcript_text}' -> '{translated_text}'"
        )
        await _safe_send_json(
            websocket,
            {
                "type": "translation_result",
                "chunk_id": trace_id,
                "text": translated_text,
                "speaker": speaker_id,
                "speaker_id": speaker_id,
                "is_owner": False,
                "is_final": True,
                "language": target_lang,
                "timestamp": timestamp_str,
            }
        )

    # 6. Action items (Prioritize LLM semantic extraction, fallback to regex only on fast-bypass)
    action = llm_action or (dispatcher.check_action_item(transcript_text) if is_bypass else None)
    if action:
        logger.info(f"📋 [ACTION_ITEM] Detected action '{action}' in transcript '{transcript_text}'")
        sess.action_items.append({"action": action, "speaker": speaker_id, "text": transcript_text})
        await _safe_send_json(
            websocket,
            {
                "type": "action_item",
                "action_item": action,
                "action": action,
                "source_text": transcript_text,
                "speaker": speaker_id,
                "chunk_id": trace_id,
                "studio_id": sess.studio_id,
                "meeting_id": sess.meeting_id,
                "timestamp": timestamp_str,
                "detected_action_item": action,
            }
        )

    # Node 2: Trigger async Meeting Intelligence Synthesizer periodically
    if len(sess.dialogue_history) >= 2 and len(sess.dialogue_history) % 3 == 0:
        async def _run_intelligence_bg(turns: list, ws: WebSocket, s_id: str):
            try:
                intel = await dispatcher.extract_meeting_intelligence(turns)
                if any(intel.values()):
                    logger.info(
                        f"🧠 [MEETING_INTELLIGENCE] Extracted {len(intel.get('action_items', []))} action items, "
                        f"{len(intel.get('direct_orders', []))} direct orders"
                    )
                    await _safe_send_json(
                        ws,
                        {
                            "type": "meeting_intelligence",
                            "session_id": s_id,
                            **intel,
                        },
                    )
            except Exception as ie:
                logger.debug(f"Background intelligence task error: {ie}")

        asyncio.create_task(_run_intelligence_bg(list(sess.dialogue_history[-6:]), websocket, sess.session_id))

    # 7. Auto TTS Synthesis (strictly optional, defaults to False)
    if sess.auto_tts and translated_text:
        prosody_buf = dispatcher.get_prosody_buffer(sess.session_id)
        clauses = prosody_buf.add_text(translated_text)
        if is_final:
            clauses.extend(prosody_buf.flush())

        for clause in clauses:
            tts_pcm, tts_sr, speed = await dispatcher.synthesize_speech(
                text=clause,
                speaker_id=speaker_id,
                engine_type=sess.tts_engine,
                speaker_audio=audio_f32,
            )
            if len(tts_pcm) > 0:
                tts_dur = len(tts_pcm) / (2.0 * tts_sr)
                logger.info(
                    f"🔊 [TTS_OUT] Synthesized {len(tts_pcm)} bytes ({tts_dur:.2f}s, speed: {speed}x) @ {tts_sr}Hz for '{speaker_id}'"
                )
                await _safe_send_json(
                    websocket,
                    {
                        "type": "tts_audio",
                        "chunk_id": trace_id,
                        "speaker_id": speaker_id,
                        "data": base64.b64encode(tts_pcm).decode("ascii"),
                        "sample_rate": tts_sr,
                        "speed": speed,
                    }
                )
                await _safe_send_json(
                    websocket,
                    {
                        "type": "tts_end",
                        "chunk_id": trace_id,
                        "speaker_id": speaker_id,
                    }
                )


@app.websocket("/v1/stream")
@app.websocket("/v1/realtime")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Bridge-Tauri compatible real-time duplex streaming WebSocket.
    """
    await websocket.accept()
    session_id = f"sess_{int(time.time() * 1000) % 1000000}"
    sess = session_store.create(session_id)
    sm = ConversationStateMachine(silence_timeout_sec=0.3)
    chronos = ChronosBuffer(sample_rate=16000)
    dispatcher.reset_speaker_session()

    # 1. Send immediate connected handshake confirmation
    await websocket.send_json(
        {
            "type": "connected",
            "status": "ok",
            "session": session_id,
            "gateway": "ramO V4 Enterprise Gateway",
        }
    )
    logger.info(f"🟢 [WS Connected] Session {session_id} registered.")

    trace_counter = 0

    # Producer-Consumer queue: decouples network I/O from neural inference
    cut_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    async def cut_consumer_worker():
        while True:
            try:
                item = await cut_queue.get()
            except asyncio.CancelledError:
                break

            if item is None:
                cut_queue.task_done()
                break

            try:
                pcm_data, cut_is_final, c_trace_id = item
                await _handle_audio_cut(websocket, sess, sm, pcm_data, cut_is_final, c_trace_id)
            except asyncio.CancelledError:
                break
            except Exception as ce:
                logger.error(f"Error in audio cut consumer: {ce}")
            finally:
                cut_queue.task_done()

    consumer_task = asyncio.create_task(cut_consumer_worker())

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            trace_counter += 1
            trace_id = f"trc_{session_id}_{trace_counter}"

            # Binary audio input
            if "bytes" in msg and msg["bytes"]:
                pcm_bytes = msg["bytes"]
                logger.debug(f"🎙️ [AUDIO_IN] Received {len(pcm_bytes)} bytes PCM audio from {sess.source}")
                cuts = chronos.add_audio(pcm_bytes)
                for cut in cuts:
                    await cut_queue.put((cut.pcm_data, cut.is_final, sess.get_current_chunk_id()))

                # Check provisional tick
                if chronos.should_trigger_provisional():
                    snapshot = chronos.get_provisional_snapshot()
                    if snapshot:
                        await cut_queue.put((snapshot, False, sess.get_current_chunk_id()))

            # JSON text frame input
            elif "text" in msg and msg["text"]:
                raw_text = msg["text"]
                try:
                    data = json.loads(raw_text)
                    logger.debug(f"📩 [WS_TEXT] Control frame: {data.get('type') or data.get('action')}")
                except Exception:
                    continue

                msg_type = data.get("type", "")
                cmd_str = data.get("command", "")

                if msg_type == "audio":
                    sess.update_config(
                        target_language=data.get("target_language") or data.get("target_lang"),
                        auto_tts=data.get("auto_tts"),
                        source=data.get("source"),
                        context_summary=data.get("context_summary"),
                    )
                    b64_data = data.get("data")
                    if b64_data:
                        try:
                            pcm_bytes = base64.b64decode(b64_data)
                            cuts = chronos.add_audio(pcm_bytes)
                            for cut in cuts:
                                await cut_queue.put((cut.pcm_data, cut.is_final, sess.get_current_chunk_id()))

                            # Check provisional tick for live streaming grey preview
                            if chronos.should_trigger_provisional():
                                snapshot = chronos.get_provisional_snapshot()
                                if snapshot:
                                    await cut_queue.put((snapshot, False, sess.get_current_chunk_id()))
                        except Exception as e:
                            logger.error(f"Error decoding base64 audio: {e}")

                elif msg_type in ("config", "update_state"):
                    sess.update_config(
                        target_language=data.get("target_language"),
                        auto_tts=data.get("auto_tts"),
                        source=data.get("source"),
                        context_summary=data.get("context_summary"),
                        studio_id=data.get("studio_id"),
                        meeting_id=data.get("meeting_id"),
                        tts_engine=data.get("tts_engine"),
                    )

                elif msg_type == "eos":
                    cut = chronos.flush()
                    if cut:
                        await cut_queue.put((cut.pcm_data, True, sess.get_current_chunk_id()))

                elif msg_type in ("tts_request", "tts") or cmd_str in ("tts", "tts_request"):
                    text_to_speak = data.get("text", "")
                    spk_id = data.get("speaker_id", "Speaker 1")
                    voice = data.get("voice", "af_heart")
                    dedup_key = f"{spk_id}:{text_to_speak}"

                    if not sess.is_duplicate_tts(dedup_key):
                        tts_pcm, tts_sr, speed = await dispatcher.synthesize_speech(
                            text=text_to_speak,
                            voice=voice,
                            speaker_id=spk_id,
                            engine_type=sess.tts_engine,
                        )
                        if len(tts_pcm) > 0:
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "tts_audio",
                                    "chunk_id": trace_id,
                                    "speaker_id": spk_id,
                                    "data": base64.b64encode(tts_pcm).decode("ascii"),
                                    "sample_rate": tts_sr,
                                },
                            )
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "tts_end",
                                    "chunk_id": trace_id,
                                    "speaker_id": spk_id,
                                },
                            )

                elif msg_type == "agent_listen" or cmd_str == "agent_listen":
                    enabled = bool(data.get("enabled", True))
                    sess.set_agent_mode(enabled)
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "agent_listen_ack",
                            "agent_mode": enabled,
                            "session_id": session_id,
                        },
                    )

                elif msg_type == "silence":
                    sm.on_speech_stop()
                    await _safe_send_json(websocket, {"type": "state_change", "state": sm.state.value})

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"🔴 [WS Disconnected] Session {session_id} disconnected.")
    finally:
        consumer_task.cancel()
        session_store.remove(session_id)


# -----------------------------------------------------------------------------
# REST Endpoints: OpenAI Compatibility
# -----------------------------------------------------------------------------

class SpeechPayload(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="af_heart")
    speed: float = Field(default=1.0)
    response_format: str = Field(default="wav")


@app.post("/v1/audio/speech")
async def text_to_speech_endpoint(payload: SpeechPayload):
    """OpenAI-compatible speech synthesis endpoint."""
    pcm16, sr, _ = await dispatcher.synthesize_speech(payload.input, voice=payload.voice)

    # Encode to WAV container
    import wave
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16)
    buf.seek(0)

    return Response(content=buf.read(), media_type="audio/wav")


@app.post("/v1/audio/transcriptions")
async def audio_transcription_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    """OpenAI-compatible speech transcription endpoint."""
    import soundfile as sf
    import io
    contents = await file.read()
    with io.BytesIO(contents) as buf:
        audio, sr = sf.read(buf, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    result = await dispatcher.process_stt(audio)
    return {
        "text": result.get("raw_text", ""),
        "words": result.get("words", []),
        "emotion": result.get("emotion", ""),
        "language": result.get("language", language),
    }


class TranslatePayload(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str = "fr"


@app.post("/v1/translate")
async def translate_endpoint(payload: TranslatePayload):
    """Direct translation endpoint."""
    translated, is_bypass, action = await dispatcher.translate_text(
        text=payload.text,
        source_lang=payload.source_language,
        target_lang=payload.target_language,
    )
    return {
        "translated_text": translated,
        "is_bypass": is_bypass,
        "action": action,
        "source_language": payload.source_language,
        "target_language": payload.target_language,
    }
