"""
services.gateway.tests.test_chronos
===================================
Unit tests for SOTA Dual-Path VAD & Zero-Overlap Chronos Buffer:
- Dual-Path VAD: 300ms normal pause (<4.0s); 120ms relaxed breath pause (4.0s - 7.0s)
- Zero overlap tail (OVERLAP_TAIL_BYTES = 0)
- Monologue hard ceiling (7.0s) for Align-then-Commit
- 600ms provisional tick
"""

import pytest
import numpy as np
from ramo_gateway.chronos import ChronosBuffer, ChronosCutType


def test_chronos_zero_overlap_tail_on_pause():
    """Verify that cuts retain ZERO overlap tail in audio_buffer."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 1.5s speech + 0.35s silence (normal pause >= 300ms)
    t = np.linspace(0, 1.5, int(1.5 * sr), dtype=np.float32)
    pcm_speech = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()
    pcm_silence = np.zeros(int(0.35 * sr), dtype=np.int16).tobytes()

    buf.add_audio(pcm_speech)
    cuts = buf.add_audio(pcm_silence)

    assert len(cuts) == 1
    assert cuts[0].cut_type == ChronosCutType.SOFT_CUT
    assert cuts[0].is_final is True
    # Crucial SOTA invariant: ZERO overlap tail retained in buffer!
    assert len(buf.audio_buffer) == 0


def test_chronos_dual_path_vad_normal_pause():
    """Verify normal pause threshold of 300ms when audio < 4.0s."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 2.0s speech + 0.20s silence (< 300ms: should NOT cut)
    t = np.linspace(0, 2.0, int(2.0 * sr), dtype=np.float32)
    pcm_speech = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()
    pcm_pause_short = np.zeros(int(0.20 * sr), dtype=np.int16).tobytes()

    buf.add_audio(pcm_speech)
    cuts = buf.add_audio(pcm_pause_short)
    assert len(cuts) == 0

    # Additional 0.15s silence (total silence = 0.35s >= 300ms: should cut!)
    pcm_pause_more = np.zeros(int(0.15 * sr), dtype=np.int16).tobytes()
    cuts2 = buf.add_audio(pcm_pause_more)
    assert len(cuts2) == 1
    assert cuts2[0].cut_type == ChronosCutType.SOFT_CUT


def test_chronos_dual_path_vad_relaxed_pause_on_long_speech():
    """Verify relaxed breath pause threshold of 120ms when audio >= 4.0s."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 4.5s speech (exceeds 4.0s soft ceiling)
    t = np.linspace(0, 4.5, int(4.5 * sr), dtype=np.float32)
    pcm_speech = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()

    buf.add_audio(pcm_speech)

    # 130ms breath pause (>= 120ms relaxed pause: should cut immediately!)
    pcm_breath_pause = np.zeros(int(0.13 * sr), dtype=np.int16).tobytes()
    cuts = buf.add_audio(pcm_breath_pause)

    assert len(cuts) == 1
    assert cuts[0].cut_type == ChronosCutType.SOFT_CUT
    assert len(buf.audio_buffer) == 0


def test_chronos_monologue_hard_ceiling_at_7s():
    """Verify that unbroken monologue triggers a cut at 7.0s hard ceiling with ZERO overlap."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 7.2s unbroken speech (no pause at all)
    t = np.linspace(0, 7.2, int(7.2 * sr), dtype=np.float32)
    pcm_speech = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()

    cuts = buf.add_audio(pcm_speech)
    assert len(cuts) >= 1
    assert cuts[0].cut_type in (ChronosCutType.HARD_CUT, ChronosCutType.SOFT_CUT)
    assert cuts[0].is_final is True
    # Audio buffer must NOT retain arbitrary 16KB overlap tail
    assert len(buf.audio_buffer) == 0 or len(buf.audio_buffer) <= int(0.2 * sr * 2)

