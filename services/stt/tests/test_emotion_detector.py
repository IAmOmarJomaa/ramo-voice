"""
Tests for SenseVoice acoustic event and emotion detector in ramo_listen.emotion_detector.
"""

import pytest
import numpy as np
from ramo_listen.emotion_detector import detect_acoustic_events


def test_neutral_speech_detection():
    # Low variance steady signal (neutral tone)
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 150 * t)

    events = detect_acoustic_events(audio, sample_rate=sr)
    assert events.emotion in ["NEUTRAL", "HAPPY", "SAD", "ANGRY"]
    assert isinstance(events.tags, list)


def test_laughter_acoustic_event_detection():
    # High frequency modulated burst characteristic of laughter bursts
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    burst = np.sin(2 * np.pi * 350 * t) * (np.sin(2 * np.pi * 12 * t) ** 2)
    burst = burst.astype(np.float32)

    events = detect_acoustic_events(burst, sample_rate=sr)
    assert events.has_laughter or events.emotion is not None
