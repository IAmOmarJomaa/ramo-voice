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


def test_register_speaker_endpoint_with_fallback_and_ready(client):
    import base64
    import soundfile as sf
    sr = 24000

    # Short audio (1.0s) -> fallback
    audio_short = (0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, sr, dtype=np.float32)))
    buf_short = io.BytesIO()
    sf.write(buf_short, audio_short, sr, format="WAV")
    b64_short = base64.b64encode(buf_short.getvalue()).decode("utf-8")

    res_short = client.post("/v1/voices/register_speaker", json={
        "speaker_id": "spk_short",
        "audio_base64": b64_short,
        "prompt_text": "Short greeting.",
        "sample_rate": sr
    })
    assert res_short.status_code == 200
    data_short = res_short.json()
    assert data_short["use_fallback"] is True
    assert data_short["engine"] == "supertonic"

    # Full audio (4.8s) -> ready for F5-TTS
    audio_full = (0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 4.8, int(4.8 * sr), dtype=np.float32)))
    buf_full = io.BytesIO()
    sf.write(buf_full, audio_full, sr, format="WAV")
    b64_full = base64.b64encode(buf_full.getvalue()).decode("utf-8")

    res_full = client.post("/v1/voices/register_speaker", json={
        "speaker_id": "spk_full",
        "audio_base64": b64_full,
        "prompt_text": "This is a full meeting monologue statement.",
        "sample_rate": sr
    })
    assert res_full.status_code == 200
    data_full = res_full.json()
    assert data_full["use_fallback"] is False
    assert data_full["engine"] == "f5-tts"

    # Synthesize with spk_full
    res_speech = client.post("/v1/audio/speech", json={
        "input": "Synthesizing translated reply.",
        "voice": "spk_full",
        "speed": 1.0
    })
    assert res_speech.status_code == 200
