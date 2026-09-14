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

async def _handle_audio_cut(
    websocket: WebSocket,
    sess: GatewaySession,
    sm: ConversationStateMachine,
    pcm_bytes: bytes,
    is_final: bool,
    trace_id: str,
):
    """Processes a discrete speech chunk through the full neural pipeline."""
    # 0. Audio cleaning
    audio_f32, _ = dispatcher.clean_audio_pcm(pcm_bytes)

    # 1. State machine barge-in check
    interrupted = sm.on_speech_start()
    if interrupted:
        await websocket.send_json({"type": "interrupt"})

    # 2. Speaker identification & inheritance
    last_spk = sess.get_last_known_speaker()
    speaker_id = dispatcher.identify_speaker(audio_f32, last_known=last_spk)
    sess.set_last_known_speaker(speaker_id)
    is_owner = sess.source == "mic"

    # 3. Speech-to-text
    stt_res = await dispatcher.process_stt(audio_f32)
    transcript_text = stt_res.get("raw_text", "").strip()

    if not transcript_text:
        return

    # 4. Emit transcript frame (matching Bridge-Tauri)
    timestamp_str = time.strftime("%I:%M %p").lstrip("0")
    transcript_frame = {
        "type": "transcript",
        "chunk_id": trace_id,
        "text": transcript_text,
        "speaker": speaker_id,
        "is_owner": is_owner,
        "is_final": is_final,
        "language": stt_res.get("language", "en"),
        "source": sess.source,
        "words": stt_res.get("words", []),
        "timestamp": timestamp_str,
        "emotion": stt_res.get("emotion", "<|NEUTRAL|>"),
        "event": "<|Speech|>",
    }
    await websocket.send_json(transcript_frame)

    if not is_final:
        return

    # 5. Translation
    target_lang = sess.target_language
    source_lang = stt_res.get("language", "en")
    translated_text, _ = await dispatcher.translate_text(
        text=transcript_text,
        source_lang=source_lang,
        target_lang=target_lang,
        session_id=sess.session_id,
        speaker_id=speaker_id,
    )

    if translated_text:
        await websocket.send_json(
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

    # 6. Action items
    action = dispatcher.check_action_item(transcript_text)
    if action:
        await websocket.send_json(
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

    # 7. Auto TTS Synthesis
    if sess.auto_tts and translated_text:
        tts_pcm, tts_sr = await dispatcher.synthesize_speech(
            text=translated_text,
            voice="af_heart",
            engine_type=sess.tts_engine,
        )
        if len(tts_pcm) > 0:
            await websocket.send_json(
                {
                    "type": "tts_audio",
                    "chunk_id": trace_id,
                    "speaker_id": speaker_id,
                    "data": base64.b64encode(tts_pcm).decode("ascii"),
                    "sample_rate": tts_sr,
                }
            )
            await websocket.send_json(
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
                cuts = chronos.add_audio(pcm_bytes)
                for cut in cuts:
                    await _handle_audio_cut(websocket, sess, sm, cut.pcm_data, cut.is_final, trace_id)

                # Check provisional tick
                if chronos.should_trigger_provisional():
                    snapshot = chronos.get_provisional_snapshot()
                    if snapshot:
                        asyncio.create_task(
                            _handle_audio_cut(websocket, sess, sm, snapshot, False, trace_id + "_prov")
                        )

            # JSON text frame input
            elif "text" in msg and msg["text"]:
                raw_text = msg["text"]
                try:
                    data = json.loads(raw_text)
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
                                await _handle_audio_cut(websocket, sess, sm, cut.pcm_data, cut.is_final, trace_id)
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
                        await _handle_audio_cut(websocket, sess, sm, cut.pcm_data, True, trace_id + "_EOS")

                elif msg_type in ("tts_request", "tts") or cmd_str in ("tts", "tts_request"):
                    text_to_speak = data.get("text", "")
                    spk_id = data.get("speaker_id", "Speaker 1")
                    voice = data.get("voice", "af_heart")
                    dedup_key = f"{spk_id}:{text_to_speak}"

                    if not sess.is_duplicate_tts(dedup_key):
                        tts_pcm, tts_sr = await dispatcher.synthesize_speech(
                            text=text_to_speak,
                            voice=voice,
                            engine_type=sess.tts_engine,
                        )
                        if len(tts_pcm) > 0:
                            await websocket.send_json(
                                {
                                    "type": "tts_audio",
                                    "chunk_id": trace_id,
                                    "speaker_id": spk_id,
                                    "data": base64.b64encode(tts_pcm).decode("ascii"),
                                    "sample_rate": tts_sr,
                                }
                            )
                            await websocket.send_json(
                                {
                                    "type": "tts_end",
                                    "chunk_id": trace_id,
                                    "speaker_id": spk_id,
                                }
                            )

                elif msg_type == "agent_listen" or cmd_str == "agent_listen":
                    enabled = bool(data.get("enabled", True))
                    sess.set_agent_mode(enabled)
                    await websocket.send_json(
                        {
                            "type": "agent_listen_ack",
                            "agent_mode": enabled,
                            "session_id": session_id,
                        }
                    )

                elif msg_type == "silence":
                    sm.on_speech_stop()
                    await websocket.send_json({"type": "state_change", "state": sm.state.value})

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"🔴 [WS Disconnected] Session {session_id} disconnected.")
    finally:
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
    pcm16, sr = await dispatcher.synthesize_speech(payload.input, voice=payload.voice, speed=payload.speed)

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
    translated, is_bypass = await dispatcher.translate_text(
        text=payload.text,
        source_lang=payload.source_language,
        target_lang=payload.target_language,
    )
    return {
        "translated_text": translated,
        "is_bypass": is_bypass,
        "source_language": payload.source_language,
        "target_language": payload.target_language,
    }
