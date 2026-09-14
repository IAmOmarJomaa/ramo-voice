import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from ramo_clean.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["service"] == "ramo-clean"


def test_clean_audio_endpoint(client):
    # Generate 0.5s audio WAV
    sr = 16000
    audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr)))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    res = client.post(
        "/v1/audio/clean",
        files={"file": ("test.wav", buf.read(), "audio/wav")},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"

    # Verify returned WAV is readable and correct length
    out_audio, out_sr = sf.read(io.BytesIO(res.content), dtype="float32")
    assert out_sr == 16000
    assert len(out_audio) > 0
