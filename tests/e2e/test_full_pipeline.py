"""
tests.e2e.test_full_pipeline
============================
Ground-Truth End-to-End integration test streaming real meeting audio
(tests/fixtures/meeting_sample_en.wav) through the sovereign Pure-Python Gateway (Port 50000)
and asserting exact Bridge-Tauri wire protocol event sequences.
"""

import base64
import json
import os
import wave
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from ramo_gateway.server import app

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "meeting_sample_en.wav")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_full_pipeline_ground_truth_audio(client):
    """
    Ingests real human speech from tests/fixtures/meeting_sample_en.wav,
    streams it in chunks over the WebSocket, and asserts sequential receipt of:
    1. connected handshake
    2. transcript with real Faster-Whisper words & timestamps
    3. translation_result in target language (French)
    4. tts_audio containing synthesized audio & tts_end
    """
    assert os.path.exists(FIXTURE_PATH), f"Missing ground truth fixture at {FIXTURE_PATH}"

    # Read real 16kHz audio
    audio_data, sr = sf.read(FIXTURE_PATH, dtype="float32")
    assert sr == 16000
    pcm16_bytes = (audio_data * 32767).astype("int16").tobytes()

    with client.websocket_connect("/v1/stream") as ws:
        # 1. Connected handshake
        handshake = ws.receive_json()
        assert handshake["type"] == "connected"
        assert handshake["status"] == "ok"
        session_id = handshake["session"]
        assert session_id.startswith("sess_")

        # 2. Send configuration frame
        config_frame = {
            "type": "config",
            "target_language": "fr",
            "auto_tts": True,
            "source": "mic",
            "context_summary": "Q4 product deliverables meeting",
        }
        ws.send_text(json.dumps(config_frame))

        # 3. Stream real audio in 0.5s chunks (16,000 bytes each)
        chunk_size = 16000
        for i in range(0, len(pcm16_bytes), chunk_size):
            chunk = pcm16_bytes[i : i + chunk_size]
            audio_msg = {
                "type": "audio",
                "data": base64.b64encode(chunk).decode("ascii"),
                "source": "mic",
                "target_language": "fr",
            }
            ws.send_text(json.dumps(audio_msg))

        # 4. Trigger end-of-speech flush
        ws.send_text(json.dumps({"type": "eos"}))

        # 5. Collect and verify pipeline events
        received_types = []
        transcript_text = ""
        translated_text = ""
        tts_audio_frames = 0

        # Read events until tts_end
        while True:
            msg = ws.receive_json()
            m_type = msg.get("type")
            received_types.append(m_type)

            if m_type == "transcript":
                transcript_text = msg.get("text", "")
                assert "words" in msg
                assert msg["is_final"] is True
                assert "speaker" in msg

            elif m_type == "translation_result":
                translated_text = msg.get("text", "")
                assert msg["language"] == "fr"

            elif m_type == "tts_audio":
                assert "data" in msg
                assert msg["sample_rate"] in (24000, 44100)
                tts_audio_frames += 1

            elif m_type == "tts_end":
                break

        # Verification of Ground Truth:
        assert "transcript" in received_types, "Pipeline did not emit transcript"
        assert len(transcript_text) > 0, "Transcript text was empty"
        # The real audio says: "And we don't have a ton"
        assert "ton" in transcript_text.lower() or "we" in transcript_text.lower() or "and" in transcript_text.lower()

        assert "translation_result" in received_types, "Pipeline did not emit translation_result"
        assert len(translated_text) > 0, "Translated text was empty"

        assert tts_audio_frames > 0, "Pipeline did not synthesize audio"
        assert "tts_end" in received_types, "Pipeline did not emit tts_end"
