"""
services/gateway/tests/test_dynamic_pacer.py
============================================
TDD tests for DynamicLatencyPacer: WSOLA time-scale modification and latency budget tracking.
"""

import pytest
import numpy as np
from ramo_gateway.pacer import DynamicLatencyPacer


def test_wsola_speedup_pitch_preservation():
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    original_audio = 0.5 * np.sin(2 * np.pi * 220 * t)  # 220 Hz A3 tone

    pacer = DynamicLatencyPacer(sample_rate=sr, speed_min=1.0, speed_max=1.25)

    # Apply 1.20x speedup
    sped_audio = pacer.apply_wsola(original_audio, speed=1.20)

    # 1. Sped audio must be shorter by ~ 1 / 1.20
    expected_samples = int(len(original_audio) / 1.20)
    assert abs(len(sped_audio) - expected_samples) < sr * 0.05

    # 2. Pitch must NOT be altered (fundamental frequency remains ~220 Hz)
    f0_orig = pacer.estimate_pitch(original_audio)
    f0_sped = pacer.estimate_pitch(sped_audio)
    assert abs(f0_orig - 220.0) < 5.0
    assert abs(f0_sped - 220.0) < 10.0


def test_calculate_dynamic_speed_from_queue():
    pacer = DynamicLatencyPacer(target_latency_ms=1200, speed_min=1.0, speed_max=1.25)

    # Low latency (400ms) -> speed should be 1.0x
    speed_low = pacer.calculate_speed(current_latency_ms=400, queue_words=5)
    assert speed_low == 1.0

    # High latency (1800ms > 1200ms budget) -> speed should ramp up toward 1.25x
    speed_high = pacer.calculate_speed(current_latency_ms=1800, queue_words=25)
    assert speed_high > 1.10
    assert speed_high <= 1.25
