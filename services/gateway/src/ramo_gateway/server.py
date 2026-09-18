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
import re
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .session import GatewaySession, SessionStore
from .chronos import LiveUtterancePipeline, UtteranceEvent, LocalAgreement, ChronosBuffer
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


async def _dispatch_translation_and_tts(
    websocket: WebSocket,
    sess: GatewaySession,
    speaker_id: str,
    transcript_text: str,
    trace_id: str,
    timestamp_str: str,
    source_lang: str,
    audio_f32: np.ndarray,
    is_final: bool,
):
    """
    Decoupled background task: translates text and synthesizes speech
    asynchronously without stalling real-time STT transcription.
    """
    try:
        # Record dialogue turn into session memory
        sess.dialogue_history.append(f"{speaker_id}: {transcript_text}")

        # 1. Translation
        target_lang = sess.target_language
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
                },
            )

        # 2. Action items
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
                },
            )

        # 3. Trigger async Meeting Intelligence Synthesizer periodically
        if len(sess.dialogue_history) >= 2 and len(sess.dialogue_history) % 3 == 0:
            async def _run_intelligence_bg(turns: list, ws: WebSocket, s_id: str):
                try:
                    intel = await dispatcher.extract_meeting_intelligence(turns)
                    if any(intel.values()):
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

        # 4. Auto TTS Synthesis
        if sess.auto_tts and translated_text:
            prosody_buf = dispatcher.get_prosody_buffer(sess.session_id)
            clauses = prosody_buf.add_text(translated_text)
            if is_final:
                clauses.extend(prosody_buf.flush())

            if not clauses and translated_text:
                clauses = [translated_text]

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
                        },
                    )
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "tts_end",
                            "chunk_id": trace_id,
                            "speaker_id": speaker_id,
                        },
                    )
    except Exception as e:
        logger.error(f"Error in decoupled translation/TTS worker: {e}", exc_info=True)


async def _run_background_diarization(
    websocket: WebSocket,
    sess: GatewaySession,
    line_id: str,
    audio_f32: np.ndarray,
    duration_s: float,
    transcript: str,
):
    """
    Decoupled background task: extracts CampPlus speaker embedding and updates clustering
    without delaying real-time transcript emission.
    """
    try:
        # 1. Overlap detection
        is_overlap, overlap_score = dispatcher.detect_overlap(audio_f32)
        if is_overlap:
            logger.info(f"👥 [CROSSTALK] Detected simultaneous speech (score: {overlap_score:.2f}) on '{line_id}'")

        # 2. Speaker identification via CampPlus with duration scaling (Moonshine standard)
        last_spk = sess.get_last_known_speaker()
        loop = asyncio.get_running_loop()
        speaker_id = await loop.run_in_executor(
            None,
            dispatcher.identify_speaker,
            audio_f32,
            last_spk,
            is_overlap,
            True,
            duration_s,
        )
        sess.set_last_known_speaker(speaker_id)

        logger.info(
            f"👥 [DIAR_ASYNC] (id={line_id}) Identified speaker: '{speaker_id}' (overlap: {is_overlap})"
        )

        # 3. Emit speaker_detected frame to Bridge-Tauri
        await _safe_send_json(
            websocket,
            {
                "type": "speaker_detected",
                "chunk_id": line_id,
                "temp_id": line_id,
                "speaker": speaker_id,
                "speaker_id": speaker_id,
                "is_overlap": is_overlap,
            },
        )

        # 4. Harvest clean speech turn into voiceprint pool for zero-shot cloning
        if not is_overlap and transcript:
            dispatcher.harvest_speech_turn(
                speaker_id=speaker_id,
                audio_f32=audio_f32,
                transcript=transcript,
                is_overlap=False,
            )
    except Exception as de:
        logger.error(f"Error in decoupled diarization worker on '{line_id}': {de}", exc_info=True)


async def _handle_stt_event(
    websocket: WebSocket,
    sess: GatewaySession,
    sm: ConversationStateMachine,
    pipeline: LiveUtterancePipeline,
    local_agreement: LocalAgreement,
    event: UtteranceEvent,
    background_tasks: Optional[Set[asyncio.Task]] = None,
):
    """
    Processes an UtteranceEvent from LiveUtterancePipeline with immediate transcript emission.
    Decouples STT emission from Diarization and Translation.
    """
    t0 = time.monotonic()
    is_owner = sess.source == "mic"
    timestamp_str = time.strftime("%I:%M %p").lstrip("0")
    revision = sess.next_revision()
    last_spk = sess.get_last_known_speaker()

    # 1. Glossary prompt only (never pass rolling dialogue into initial_prompt!)
    glossary_prompt = (
        ", ".join(sess.dynamic_glossary.keys())
        if getattr(sess, "dynamic_glossary", None)
        else None
    )

    if not event.is_final:
        # 1. State machine barge-in check
        interrupted = sm.on_speech_start()
        if interrupted:
            logger.info("⚡ [BARGE_IN] Speech detected during assistant turn — sending interrupt")
            await _safe_send_json(websocket, {"type": "interrupt"})

        # 2. Provisional STT preview pass
        stt_res = await dispatcher.process_stt(event.audio, initial_prompt=glossary_prompt)
        dt = time.monotonic() - t0
        pipeline.record_decode_wall_time(dt)

        raw_transcript = stt_res.get("raw_text", "").strip()
        words_list = stt_res.get("words", [])
        stt_lang = stt_res.get("language", "en")
        emotion_tag = stt_res.get("emotion", "<|NEUTRAL|>")

        if not raw_transcript:
            return

        preview_text = raw_transcript

        # Immediate WebSocket provisional emission (<1.2s latency)
        # Note: chunk_id AND utterance_id are event.line_id
        await _safe_send_json(
            websocket,
            {
                "type": "transcript",
                "event": "transcript_provisional",
                "utterance_id": event.line_id,
                "revision": revision,
                "chunk_id": event.line_id,
                "text": preview_text,
                "speaker": last_spk,
                "is_owner": is_owner,
                "is_final": False,
                "language": stt_lang,
                "source": sess.source,
                "words": words_list,
                "timestamp": timestamp_str,
                "emotion": emotion_tag,
            },
        )
        return

    # FINAL COMMIT: Batch-transcribe 100% of the accumulated turn audio
    sm.on_speech_stop()
    stt_res = await dispatcher.process_stt(event.audio, initial_prompt=glossary_prompt)
    dt = time.monotonic() - t0
    pipeline.record_decode_wall_time(dt)

    raw_transcript = stt_res.get("raw_text", "").strip()
    words_list = stt_res.get("words", [])
    stt_lang = stt_res.get("language", "en")
    emotion_tag = stt_res.get("emotion", "<|NEUTRAL|>")

    final_text = raw_transcript
    if not final_text:
        return

    # Echo suppression guard: reject identical consecutive transcripts within 8.0s
    now = time.monotonic()
    if sess.last_final_transcript and final_text.lower() == sess.last_final_transcript.lower():
        if (now - sess.last_final_time) < 8.0:
            logger.warning(
                f"🛡️ [ECHO_SUPPRESSION] Discarded identical consecutive transcript within 8s: '{final_text}'"
            )
            return

    sess.last_final_transcript = final_text
    sess.last_final_time = now

    logger.info(
        f"👂 [STT_FINAL] (id={event.line_id}) '{final_text}' | Lang: {stt_lang} | Emotion: {emotion_tag}"
    )

    # Immediate final transcript emission to WebSocket (<1.2s latency)
    await _safe_send_json(
        websocket,
        {
            "type": "transcript",
            "event": "transcript_final",
            "utterance_id": event.line_id,
            "revision": revision,
            "chunk_id": event.line_id,
            "text": final_text,
            "speaker": last_spk,
            "is_owner": is_owner,
            "is_final": True,
            "language": stt_lang,
            "source": sess.source,
            "words": words_list,
            "timestamp": timestamp_str,
            "emotion": emotion_tag,
        },
    )

    # Decouple Diarization and Translation into independent background tasks
    if background_tasks is not None:
        d_task = asyncio.create_task(
            _run_background_diarization(
                websocket=websocket,
                sess=sess,
                line_id=event.line_id,
                audio_f32=event.audio,
                duration_s=event.duration_s,
                transcript=final_text,
            )
        )
        background_tasks.add(d_task)
        d_task.add_done_callback(background_tasks.discard)

        t_task = asyncio.create_task(
            _dispatch_translation_and_tts(
                websocket=websocket,
                sess=sess,
                speaker_id=last_spk,
                transcript_text=final_text,
                trace_id=event.line_id,
                timestamp_str=timestamp_str,
                source_lang=stt_lang,
                audio_f32=event.audio,
                is_final=True,
            )
        )
        background_tasks.add(t_task)
        t_task.add_done_callback(background_tasks.discard)


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
    pipeline = LiveUtterancePipeline(sample_rate=16000, session_id=session_id, vad_redemption_ms=2000)
    local_agreement = LocalAgreement(n_agreement=2)
    background_tasks: Set[asyncio.Task] = set()
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

    # Producer-Consumer queue: decouples network I/O from neural inference (unbounded to prevent Error 1011)
    stt_queue: asyncio.Queue[UtteranceEvent] = asyncio.Queue()

    async def stt_consumer_worker():
        while True:
            try:
                event = await stt_queue.get()
            except asyncio.CancelledError:
                break

            if event is None:
                stt_queue.task_done()
                break

            try:
                await _handle_stt_event(
                    websocket=websocket,
                    sess=sess,
                    sm=sm,
                    pipeline=pipeline,
                    local_agreement=local_agreement,
                    event=event,
                    background_tasks=background_tasks,
                )
            except asyncio.CancelledError:
                break
            except Exception as ce:
                logger.error(f"Error in STT event consumer: {ce}", exc_info=True)
            finally:
                stt_queue.task_done()

    consumer_task = asyncio.create_task(stt_consumer_worker())

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
                audio_f32, _ = dispatcher.clean_audio_pcm(pcm_bytes)
                events = pipeline.push_audio(audio_f32)
                for ev in events:
                    stt_queue.put_nowait(ev)

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
                            audio_f32, _ = dispatcher.clean_audio_pcm(pcm_bytes)
                            events = pipeline.push_audio(audio_f32)
                            for ev in events:
                                stt_queue.put_nowait(ev)
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
                    if "dynamic_glossary" in data and isinstance(data["dynamic_glossary"], dict):
                        sess.dynamic_glossary = data["dynamic_glossary"]

                elif msg_type == "eos":
                    events = pipeline.flush()
                    for ev in events:
                        stt_queue.put_nowait(ev)

                elif msg_type in ("tts_request", "tts") or cmd_str in ("tts", "tts_request"):
                    text_to_speak = data.get("text", "")
                    spk_id = data.get("speaker_id", "Speaker 1")
                    voice = data.get("voice", "af_heart")
                    req_chunk_id = data.get("chunk_id") or sess.get_current_chunk_id()
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
                                    "chunk_id": req_chunk_id,
                                    "speaker_id": spk_id,
                                    "data": base64.b64encode(tts_pcm).decode("ascii"),
                                    "sample_rate": tts_sr,
                                },
                            )
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "tts_end",
                                    "chunk_id": req_chunk_id,
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
        for t in list(background_tasks):
            t.cancel()
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
