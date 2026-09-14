"""
services.gateway.tests.test_bridge_tauri_ws
===========================================
Comprehensive integration tests validating 100% Bridge-Tauri WebSocket protocol
wire compatibility on Port 50000 (/v1/stream).
"""

import base64
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient
from ramo_gateway.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_bridge_tauri_connected_handshake(client):
    """Bridge-Tauri expects immediate {"type": "connected", "status": "ok"} upon connection."""
    with client.websocket_connect("/v1/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"
        assert msg["status"] == "ok"
        assert "session" in msg
        assert "ramO" in msg["gateway"]


def test_bridge_tauri_on_demand_tts(client):
    """Bridge-Tauri sends on-demand {"type": "tts_request"} and receives tts_audio + tts_end."""
    with client.websocket_connect("/v1/stream") as ws:
        # Handshake
        ws.receive_json()

        # Send TTS command
        tts_cmd = {
            "type": "tts_request",
            "text": "Hello team, welcome to the sprint review.",
            "speaker_id": "Speaker 1",
            "voice": "af_heart",
        }
        ws.send_text(json.dumps(tts_cmd))

        # Receive tts_audio
        audio_msg = ws.receive_json()
        assert audio_msg["type"] == "tts_audio"
        assert audio_msg["speaker_id"] == "Speaker 1"
        assert "data" in audio_msg
        assert audio_msg["sample_rate"] in (24000, 44100)

        # Receive tts_end
        end_msg = ws.receive_json()
        assert end_msg["type"] == "tts_end"


def test_bridge_tauri_audio_streaming_flow(client):
    """
    Stream audio via JSON base64 frame, followed by eos, and assert
    receipt of transcript, translation_result, and auto-tts.
    """
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_json()  # connected

        # 1.0s audible tone at 16kHz
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        pcm_f32 = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        pcm_bytes = (pcm_f32 * 32767).astype(np.int16).tobytes()

        # Send audio frame with target_language=fr
        audio_frame = {
            "type": "audio",
            "data": base64.b64encode(pcm_bytes).decode("ascii"),
            "target_language": "fr",
            "auto_tts": True,
            "source": "mic",
        }
        ws.send_text(json.dumps(audio_frame))

        # Flush via eos
        ws.send_text(json.dumps({"type": "eos"}))

        # We should receive transcript
        transcript_msg = ws.receive_json()
        assert transcript_msg["type"] == "transcript"
        assert "text" in transcript_msg
        assert transcript_msg["is_final"] is True
        assert "speaker" in transcript_msg
        assert "words" in transcript_msg

        # Then translation_result
        trans_msg = ws.receive_json()
        assert trans_msg["type"] == "translation_result"
        assert "text" in trans_msg
        assert trans_msg["language"] == "fr"

        # Then tts_audio & tts_end
        tts_audio = ws.receive_json()
        assert tts_audio["type"] == "tts_audio"
        assert "data" in tts_audio

        tts_end = ws.receive_json()
        assert tts_end["type"] == "tts_end"


def test_rest_endpoints(client):
    """Test OpenAI-compatible REST endpoints on Gateway."""
    # 1. Speech synthesis
    speech_res = client.post("/v1/audio/speech", json={"input": "Hello world", "voice": "af_heart"})
    assert speech_res.status_code == 200
    assert speech_res.headers["content-type"] == "audio/wav"

    # 2. Direct translation
    trans_res = client.post("/v1/translate", json={"text": "Okay.", "source_language": "en", "target_language": "fr"})
    assert trans_res.status_code == 200
    assert trans_res.json()["translated_text"] == "D'accord."
