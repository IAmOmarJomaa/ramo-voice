import numpy as np
import pytest
from ramo_gateway.chronos import LiveUtterancePipeline, UtteranceEvent


def test_room_noise_silence_dormancy():
    """
    BUG 2 REPRODUCTION / SOTA INVARIANT:
    Under ambient room noise (RMS = 0.025, exactly as observed in Video Project 39),
    the pipeline MUST remain in silence sleep and emit ZERO UtteranceEvents.
    Whisper must never be called during room pauses.
    """
    pipeline = LiveUtterancePipeline(session_id="test_noise_sess")
    
    # Generate 3 seconds of ambient room noise (RMS 0.025)
    rng = np.random.default_rng(42)
    noise_raw = rng.normal(0, 1, 16000 * 3).astype(np.float32)
    noise_scaled = noise_raw * (0.025 / np.sqrt(np.mean(noise_raw**2)))
    assert np.isclose(np.sqrt(np.mean(noise_scaled**2)), 0.025, atol=0.002)

    # Push in 250ms chunks (4000 samples each)
    events = []
    chunk_size = 4000
    for i in range(0, len(noise_scaled), chunk_size):
        chunk = noise_scaled[i : i + chunk_size]
        evs = pipeline.push_audio(chunk)
        events.extend(evs)

    # Under pure room noise, the pipeline must NOT trigger speech or emit events
    assert len(events) == 0, f"Expected 0 events under room noise, got {len(events)}"
    assert not pipeline.in_speech, "Pipeline should remain in_speech=False under room noise"


def test_zero_sentence_head_truncation_audio():
    """
    BUG 1 REPRODUCTION / SOTA INVARIANT:
    When speech ends, the final UtteranceEvent MUST contain the 100% complete accumulated
    turn audio starting from voice onset (plus look-behind), NOT just a trailing tail.
    """
    pipeline = LiveUtterancePipeline(session_id="test_head_sess")

    # 1. Start with 1s of silence
    silence = np.zeros(16000, dtype=np.float32)
    pipeline.push_audio(silence, is_voice=False)

    # 2. 4 seconds of active speech (sine wave @ 440Hz, RMS ~ 0.1)
    t = np.linspace(0, 4.0, 16000 * 4, endpoint=False)
    speech = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    events = []
    for i in range(0, len(speech), 4000):
        chunk = speech[i : i + 4000]
        evs = pipeline.push_audio(chunk, is_voice=True)
        events.extend(evs)

    assert pipeline.in_speech, "Pipeline should be in speech"

    # 3. 2 seconds of pause (VAD redemption)
    pause = np.zeros(16000 * 2, dtype=np.float32)
    final_events = []
    for i in range(0, len(pause), 4000):
        chunk = pause[i : i + 4000]
        evs = pipeline.push_audio(chunk, is_voice=False)
        final_events.extend(evs)

    # Should have emitted a final event
    assert len(final_events) == 1, f"Expected exactly 1 final event on pause, got {len(final_events)}"
    final_ev = final_events[0]
    assert final_ev.is_final is True

    # The final audio MUST contain the full 4.0s of speech + look-behind buffer
    speech_duration = len(final_ev.audio) / 16000.0
    assert speech_duration >= 4.0, f"Final audio duration ({speech_duration:.2f}s) lost speech head!"


def test_plosive_look_behind_prepended():
    """
    Verifies that the 512ms rolling look-behind buffer is prepended upon SpeechStart
    to capture unvoiced plosive consonants.
    """
    pipeline = LiveUtterancePipeline(session_id="test_plosive_sess")

    # Push unique marker in the look-behind window during silence
    marker_samples = np.full(8192, 0.05, dtype=np.float32)
    pipeline.push_audio(marker_samples, is_voice=False)

    # Speech onset
    speech_chunk = np.full(4000, 0.3, dtype=np.float32)
    pipeline.push_audio(speech_chunk, is_voice=True)

    assert pipeline.in_speech
    # current_samples should start with the marker from the look-behind buffer
    assert len(pipeline.current_samples) == 8192 + 4000
    assert np.allclose(pipeline.current_samples[:8192], 0.05)
