"""
ramo_clean.server
=================
FastAPI HTTP REST & WebSocket streaming server for Audio Cleaning & Preconditioning.
Runs standalone on Port 50054.
"""

import io
import logging
from typing import Optional
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ramo_clean.pipeline import AudioPreconditioner

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ramO Clean (Audio Preconditioning Service)",
    description="Sovereign 5-Stage Audio Preconditioning (HPF, VAD, Spectral Gate, AGC)",
    version="0.1.0",
)

preconditioner = AudioPreconditioner(sample_rate=16000)


@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "service": "ramo-clean",
        "sample_rate": 16000,
        "stages": ["hpf_80hz", "spectral_gate", "agc_-20dbfs", "silero_vad_v5"],
    }


@app.post("/v1/audio/clean")
async def clean_audio(file: UploadFile = File(...)):
    """Clean uploaded audio file through 5-stage preconditioning pipeline."""
    content = await file.read()
    with io.BytesIO(content) as f:
        audio, sr = sf.read(f, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Resample to 16000 if needed
    if sr != 16000:
        from scipy import signal
        target_len = int(round(len(audio) * 16000.0 / sr))
        audio = signal.resample(audio, target_len).astype(np.float32)

    result = preconditioner.process_chunk(audio, stream=False)

    out_buf = io.BytesIO()
    sf.write(out_buf, result.audio, 16000, format="WAV", subtype="PCM_16")
    out_buf.seek(0)

    return Response(
        content=out_buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="cleaned.wav"'},
    )


@app.post("/v1/audio/vad")
async def vad_audio(file: UploadFile = File(...)):
    """Run VAD speech detection on uploaded audio."""
    content = await file.read()
    with io.BytesIO(content) as f:
        audio, sr = sf.read(f, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    result = preconditioner.process_chunk(audio, stream=False)
    events_payload = [
        {"type": type(e).__name__, "timestamp_samples": getattr(e, "timestamp_samples", 0)}
        for e in result.vad_events
    ]
    return {
        "is_speech": result.is_speech,
        "events": events_payload,
        "sample_rate": 16000,
    }


@app.websocket("/v1/audio/clean/stream")
async def clean_stream(websocket: WebSocket):
    """Real-time duplex WebSocket for low-latency streaming preconditioning."""
    await websocket.accept()
    stream_preconditioner = AudioPreconditioner(sample_rate=16000)
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            # Convert int16 PCM to float32
            chunk_int16 = np.frombuffer(data, dtype=np.int16)
            chunk_f32 = chunk_int16.astype(np.float32) / 32768.0

            result = stream_preconditioner.process_chunk(chunk_f32, stream=True)

            # Convert back to int16 PCM
            cleaned_int16 = (result.audio * 32767.0).astype(np.int16)
            await websocket.send_bytes(cleaned_int16.tobytes())
    except WebSocketDisconnect:
        logger.info("Client disconnected from clean audio stream")
