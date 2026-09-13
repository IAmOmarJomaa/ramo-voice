"""
Integration tests for ramo_listen.server FastAPI application.
"""

import io
import wave
import pytest
import numpy as np
from fastapi.testclient import TestClient
from ramo_listen.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "sensevoice" in data["engine"].lower()


def test_transcription_endpoint(client):
    # Synthesize 1s test WAV (16kHz PCM16)
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    pcm_float = 0.5 * np.sin(2 * np.pi * 200 * t)
    pcm_int16 = (pcm_float * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())

    buf.seek(0)
    files = {"file": ("test.wav", buf, "audio/wav")}
    res = client.post("/v1/audio/transcriptions", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "text" in data
    assert "emotion" in data
