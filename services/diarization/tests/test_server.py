"""
Integration tests for ramo_speaker.server FastAPI application.
"""

import io
import wave
import pytest
import numpy as np
from fastapi.testclient import TestClient
from ramo_speaker.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "diarization" in data["service"].lower()


def test_diarize_endpoint(client):
    # Synthesize 2 seconds of test audio
    sr = 16000
    pcm_float = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 2 * sr, dtype=np.float32))
    pcm_int16 = (pcm_float * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())

    buf.seek(0)
    files = {"file": ("test.wav", buf, "audio/wav")}
    res = client.post("/v1/diarize", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "turns" in data
    assert "speakers" in data


def test_harvest_voiceprint_endpoint(client):
    # Synthesize 4 seconds of test audio
    sr = 16000
    pcm_float = 0.5 * np.sin(2 * np.pi * 250 * np.linspace(0, 4.0, 4 * sr, dtype=np.float32))
    pcm_int16 = (pcm_float * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())

    buf.seek(0)
    files = {"file": ("test.wav", buf, "audio/wav")}
    res = client.post("/v1/harvest/voiceprint", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "harvested_speakers" in data
