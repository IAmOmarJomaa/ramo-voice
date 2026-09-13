"""
ramo_voice.server
=================
Production FastAPI REST & WebSocket server for ramo_voice.
Exposes OpenAI & ElevenLabs compatible endpoints for instant speech & voice cloning.
"""

import io
import time
import wave
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np

from .profiles import default_store, VoiceProfile
from .engines.base import BaseTTSEngine
from .engines.supertonic_engine import SupertonicEngine
from .engines.cloning_engine import FlowMatchingCloningEngine
from .chunker import split_text_into_chunks, concatenate_audio_chunks
from .purifier import clean_vocal_prompt

logger = logging.getLogger("ramo_voice.server")

# Active engines
supertonic_engine = SupertonicEngine()
cloning_engine = FlowMatchingCloningEngine()


def get_engine_for_profile(profile: Optional[VoiceProfile]) -> BaseTTSEngine:
    """Route voice request: cloned voices to flow matching, presets to instant ONNX."""
    if profile and profile.voice_type == "cloned":
        return cloning_engine
    return supertonic_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ramo_voice engines and registering default presets...")
    await supertonic_engine.load()
    await cloning_engine.load()

    # Seed default profiles
    default_store.register(VoiceProfile(
        voice_id="af_heart",
        name="Heart (Warm Female)",
        voice_type="preset",
        sample_rate=44100,
        language="en"
    ))
    default_store.register(VoiceProfile(
        voice_id="am_adam",
        name="Adam (Natural Male)",
        voice_type="preset",
        sample_rate=44100,
        language="en"
    ))
    yield


app = FastAPI(
    title="ramO Voice API",
    version="0.1.0",
    description="Sovereign ElevenLabs Alternative & Voice Synthesis Engine",
    lifespan=lifespan
)


class SpeechRequest(BaseModel):
    input: str = Field(..., description="The text to generate audio for.")
    voice: str = Field(default="af_heart", description="Voice ID or cloned profile name.")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = Field(default="wav", description="Audio format (wav, pcm).")


def _pcm16_to_wav(pcm_float32: np.ndarray, sample_rate: int) -> bytes:
    pcm_int16 = (np.clip(pcm_float32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine": supertonic_engine.engine_id,
        "engines": [supertonic_engine.engine_id, cloning_engine.engine_id],
        "registered_voices": len(default_store.list_profiles()),
        "default_sample_rate": 44100
    }


@app.get("/v1/voices")
async def list_voices():
    profiles = default_store.list_profiles()
    return {
        "voices": [
            {
                "voice_id": p.voice_id,
                "name": p.name,
                "category": p.voice_type,
                "sample_rate": p.sample_rate,
                "language": p.language
            }
            for p in profiles.values()
        ]
    }


@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    """OpenAI & ElevenLabs compatible speech endpoint."""
    t0 = time.perf_counter()
    profile = default_store.get(req.voice)
    if not profile:
        profile = default_store.get("af_heart")

    engine = get_engine_for_profile(profile)
    chunks = split_text_into_chunks(req.input)
    audio_chunks = []
    sr = engine.sample_rate

    for chunk in chunks:
        audio, chunk_sr = await engine.generate_chunk(chunk, profile, speed=req.speed)
        audio_chunks.append(audio)
        sr = chunk_sr

    final_audio = concatenate_audio_chunks(audio_chunks, sample_rate=sr)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Synthesized {len(req.input)} chars using {engine.engine_id} in {latency_ms:.1f}ms (RTF: {latency_ms / (len(final_audio) / sr * 1000):.2f})")

    wav_bytes = _pcm16_to_wav(final_audio, sr)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


@app.post("/v1/voices/clone")
async def clone_voice(
    file: UploadFile = File(...),
    voice_id: str = Form(...),
    name: Optional[str] = Form(None),
):
    """Register a new zero-shot voice profile from a 3-10s audio sample."""
    contents = await file.read()
    import soundfile as sf
    with io.BytesIO(contents) as buf:
        audio_data, sr = sf.read(buf, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    clean_audio = clean_vocal_prompt(audio_data, sample_rate=sr)

    profile = VoiceProfile(
        voice_id=voice_id,
        name=name or voice_id,
        voice_type="cloned",
        sample_rate=24000,
        conditioning_latents=clean_audio
    )
    default_store.register(profile)
    return {
        "status": "success",
        "voice_id": voice_id,
        "name": profile.name,
        "sample_rate": 24000,
        "duration_sec": len(clean_audio) / sr
    }


@app.websocket("/v1/stream")
async def websocket_stream(ws: WebSocket):
    """Real-time streaming synthesis via WebSocket."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text", "")
            voice_id = data.get("voice", "af_heart")
            profile = default_store.get(voice_id) or default_store.get("af_heart")
            engine = get_engine_for_profile(profile)

            if text.strip():
                async for audio_chunk in engine.generate_stream(text, profile):
                    pcm_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    await ws.send_bytes(pcm_int16.tobytes())
                await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
