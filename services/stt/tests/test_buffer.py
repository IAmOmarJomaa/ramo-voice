"""
Tests for AudioRingBuffer in ramo_listen.buffer.
Verifies audio buffering, sample rate conversion to 16kHz, and window extraction.
"""

import pytest
import numpy as np
from ramo_listen.buffer import AudioRingBuffer


def test_buffer_push_and_extract():
    buf = AudioRingBuffer(target_sr=16000, max_duration_sec=10.0)
    assert buf.duration_sec == 0.0

    # Push 1 second of 16kHz audio
    samples_16k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32))
    buf.push(samples_16k, input_sr=16000)

    assert pytest.approx(buf.duration_sec, rel=1e-2) == 1.0
    extracted = buf.get_window(duration_sec=0.5)
    assert len(extracted) == 8000


def test_buffer_resampling_from_44100():
    buf = AudioRingBuffer(target_sr=16000, max_duration_sec=10.0)

    # Push 1 second of 44.1kHz audio
    samples_44k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100, dtype=np.float32))
    buf.push(samples_44k, input_sr=44100)

    # In target 16kHz domain, 1 second = 16000 samples
    assert pytest.approx(buf.duration_sec, rel=1e-2) == 1.0
    assert len(buf.get_all()) == 16000


def test_buffer_overflow_ring_behavior():
    # 2-second maximum buffer
    buf = AudioRingBuffer(target_sr=16000, max_duration_sec=2.0)

    # Push 3 seconds of audio
    samples_3s = np.ones(48000, dtype=np.float32)
    buf.push(samples_3s, input_sr=16000)

    # Buffer must clamp to max 2.0 seconds (32000 samples)
    assert pytest.approx(buf.duration_sec, rel=1e-2) == 2.0
    assert len(buf.get_all()) == 32000
