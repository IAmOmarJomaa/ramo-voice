import numpy as np
import pytest
from ramo_clean.silero_vad import SileroProcessor, SpeechStart, SpeechEnd


def test_vad_processor_hysteresis_and_redemption():
    """Verify that SileroProcessor correctly handles state transitions and redemption."""
    processor = SileroProcessor(
        positive_threshold=0.5,
        negative_threshold=0.35,
        min_speech_ms=100,
        redemption_ms=300,
    )

    # 1. Feed pure silence (amplitude 0)
    silence = np.zeros(16000, dtype=np.float32)
    events = processor.process(silence)
    assert len(events) == 0, "Silence triggered false speech event"
    assert not processor.in_speech

    # 2. Feed simulated speech signal (strong energy with vocal frequencies)
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    speech = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    events = processor.process(speech)
    assert processor.in_speech, "VAD failed to detect speech in active audio"

    # 3. Feed short pause (100ms silence < 300ms redemption)
    short_pause = np.zeros(1600, dtype=np.float32)
    pause_events = processor.process(short_pause)
    # Inside redemption window, should remain in speech
    assert processor.in_speech, "VAD prematurely exited speech during short pause inside redemption window"

    # 4. Feed prolonged silence (> 300ms redemption)
    long_silence = np.zeros(8000, dtype=np.float32)
    end_events = processor.process(long_silence)
    assert not processor.in_speech, "VAD failed to transition out of speech after redemption window"
    assert any(isinstance(e, SpeechEnd) for e in end_events), "SpeechEnd was not emitted"
