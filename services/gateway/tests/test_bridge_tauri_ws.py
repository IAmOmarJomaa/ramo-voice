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
    import os
    import soundfile as sf
    fixture_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "fixtures", "meeting_sample_en.wav")
    )
    audio_data, sr = sf.read(fixture_path, dtype="float32")
    pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()

    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_json()  # connected

        # Stream audio in 0.5s chunks (16000 bytes) matching live mic input
        chunk_size = 16000
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            ws.send_text(
                json.dumps(
                    {
                        "type": "audio",
                        "data": base64.b64encode(chunk).decode("ascii"),
                        "target_language": "fr",
                        "auto_tts": True,
                        "source": "mic",
                    }
                )
            )

        # Flush via eos
        ws.send_text(json.dumps({"type": "eos"}))

        # Collect and assert sequential receipt of pipeline events until tts_end
        received_types = []
        transcript_msg = None
        trans_msg = None
        tts_audio = None
        tts_end = None

        while True:
            msg = ws.receive_json()
            m_type = msg.get("type")
            received_types.append(m_type)
            if m_type == "transcript":
                if msg.get("is_final"):
                    transcript_msg = msg
            elif m_type == "translation_result":
                trans_msg = msg
            elif m_type == "tts_audio":
                tts_audio = msg
            elif m_type == "tts_end":
                tts_end = msg
                break

        assert transcript_msg is not None, "Did not receive final transcript"
        assert "text" in transcript_msg
        assert len(transcript_msg["text"]) > 0
        assert "speaker" in transcript_msg
        assert "words" in transcript_msg

        assert trans_msg is not None, "Did not receive translation_result"
        assert "text" in trans_msg
        assert trans_msg["language"] == "fr"

        assert tts_audio is not None, "Did not receive tts_audio"
        assert "data" in tts_audio

        assert tts_end is not None, "Did not receive tts_end"


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
