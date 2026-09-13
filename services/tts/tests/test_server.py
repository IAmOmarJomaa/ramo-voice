import io
import wave
import pytest
from fastapi.testclient import TestClient
import numpy as np

from ramo_voice.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "supertonic-3"


def test_list_voices(client):
    res = client.get("/v1/voices")
    assert res.status_code == 200
    data = res.json()
    voice_ids = [v["voice_id"] for v in data["voices"]]
    assert "af_heart" in voice_ids
    assert "am_adam" in voice_ids


def test_generate_speech(client):
    payload = {
        "input": "Welcome to the sovereign ramO voice engine. Studio quality sound at forty-four point one kilohertz.",
        "voice": "af_heart",
        "speed": 1.0,
        "response_format": "wav"
    }
    res = client.post("/v1/audio/speech", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"

    with io.BytesIO(res.content) as buf:
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            frames = wf.readframes(wf.getnframes())
            assert len(frames) > 0


def test_clone_voice_endpoint(client):
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    files = {"file": ("test_sample.wav", buf, "audio/wav")}
    data = {"voice_id": "test_speaker_omar", "name": "Omar Clone"}

    res = client.post("/v1/voices/clone", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["status"] == "success"
    assert res_data["voice_id"] == "test_speaker_omar"

    voices_res = client.get("/v1/voices")
    voice_ids = [v["voice_id"] for v in voices_res.json()["voices"]]
    assert "test_speaker_omar" in voice_ids
