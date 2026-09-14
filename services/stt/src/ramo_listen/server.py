"""
ramo_listen.server
==================
Production FastAPI REST & WebSocket streaming server for ramo_listen.
Exposes OpenAI-compatible transcriptions endpoint and real-time streaming WebSocket.
"""

import io
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from pydantic import BaseModel

from .buffer import AudioRingBuffer
from .local_agreement import LocalAgreement
from .engines.sensevoice_engine import SenseVoiceEngine
from ramo_common.logging import setup_service_logging, tail_service_log

logger = setup_service_logging("ramo_listen")

engine = SenseVoiceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ramo_listen SenseVoice engine...")
    await engine.load()
    yield


app = FastAPI(
    title="ramO Listen API",
    version="0.1.0",
    description="Sovereign Real-Time Streaming STT & Emotion Intelligence Microservice",
    lifespan=lifespan
)


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "engine": engine.engine_id,
        "sample_rate": engine.sample_rate
    }


@app.get("/logs")
async def get_logs(tail: int = 100):
    return {"service": "ramo_listen", "lines": tail_service_log("ramo_listen", n=tail)}


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    """
    OpenAI-compatible transcription endpoint enriched with SenseVoice emotion tags.
    """
    contents = await file.read()
    with io.BytesIO(contents) as buf:
        audio_data, sr = sf.read(buf, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    result = await engine.transcribe(audio_data, sample_rate=sr)
    return result


@app.websocket("/v1/listen")
async def websocket_listen(ws: WebSocket):
    """
    Real-time duplex streaming ASR via WebSocket.
    Client streams 16kHz PCM16 audio bytes.
    Server returns committed and tentative tokens stabilized by LocalAgreement.
    """
    await ws.accept()
    buffer = AudioRingBuffer(target_sr=16000, max_duration_sec=30.0)
    stabilizer = LocalAgreement(n_agreement=2)

    try:
        while True:
            message = await ws.receive()
            if "bytes" in message and message["bytes"]:
                # Incoming raw PCM16 chunk
                chunk_bytes = message["bytes"]
                pcm_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                pcm_float32 = (pcm_int16.astype(np.float32) / 32768.0)
                buffer.push(pcm_float32, input_sr=16000)

                # Process window if buffer has at least 0.5s of audio
                if buffer.duration_sec >= 0.5:
                    window_audio = buffer.get_window(duration_sec=2.0)
                    trans_res = await engine.transcribe(window_audio, sample_rate=16000)
                    agreement = stabilizer.step(trans_res.get("raw_text", ""))

                    await ws.send_json({
                        "type": "partial",
                        "committed": agreement.committed,
                        "tentative": agreement.tentative,
                        "emotion": trans_res.get("emotion", "NEUTRAL"),
                        "tags": trans_res.get("tags", []),
                    })

            elif "text" in message and message["text"]:
                import json
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "flush":
                        final_text = stabilizer.flush()
                        await ws.send_json({
                            "type": "final",
                            "text": final_text
                        })
                        buffer.clear()
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket listen client disconnected.")
