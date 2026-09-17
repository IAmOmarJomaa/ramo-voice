"""
services.gateway.tests.test_chronos
===================================
Unit tests for Chronos Dynamic Auto-Cut Buffer:
- 4.0s hard cut
- 0.5s overlap tail retention
- 500ms conversational silence soft cut
- 500ms provisional tick
- Soft cut tail discard guard
"""

import pytest
import numpy as np
from ramo_gateway.chronos import ChronosBuffer, ChronosCutType


def test_chronos_hard_cut_preserves_overlap_tail():
    buf = ChronosBuffer(sample_rate=16000)

    # 4.5 seconds of loud continuous speech (16kHz PCM16, 2 bytes/sample)
    # 4.5 * 16000 * 2 = 144,000 bytes
    t = np.linspace(0, 4.5, int(4.5 * 16000), dtype=np.float32)
    pcm_float = 0.3 * np.sin(2 * np.pi * 300 * t)
    pcm_bytes = (pcm_float * 32767).astype(np.int16).tobytes()

    cuts = buf.add_audio(pcm_bytes)

    # Expect at least one hard cut
    assert len(cuts) >= 1
    cut = cuts[0]
    assert cut.cut_type == ChronosCutType.HARD_CUT
    assert len(cut.pcm_data) >= 128000  # at least 4.0s

    # Buffer should now retain the 0.5s overlap tail (16,000 bytes)
    assert len(buf.audio_buffer) == 16000


def test_chronos_soft_cut_on_conversational_silence():
    buf = ChronosBuffer(sample_rate=16000)

    # 1.5 seconds speech + 0.6 seconds silence
    sr = 16000
    t_speech = np.linspace(0, 1.5, int(1.5 * sr), dtype=np.float32)
    pcm_speech = (0.4 * np.sin(2 * np.pi * 300 * t_speech) * 32767).astype(np.int16).tobytes()
    pcm_silence = np.zeros(int(0.6 * sr), dtype=np.int16).tobytes()

    # Add speech first: no cut yet
    cuts_speech = buf.add_audio(pcm_speech)
    assert len(cuts_speech) == 0

    # Add silence: should trigger soft cut!
    cuts_silence = buf.add_audio(pcm_silence)
    assert len(cuts_silence) == 1
    assert cuts_silence[0].cut_type == ChronosCutType.SOFT_CUT
    assert len(cuts_silence[0].pcm_data) > 0


def test_chronos_tail_discard_guard():
    buf = ChronosBuffer(sample_rate=16000)

    # Put exactly 0.4s (smaller than overlapTailBytes=16000) of audio then silence
    sr = 16000
    pcm_tail = (0.2 * np.ones(int(0.4 * sr)) * 32767).astype(np.int16).tobytes()
    pcm_silence = np.zeros(int(0.6 * sr), dtype=np.int16).tobytes()

    buf.add_audio(pcm_tail)
    cuts = buf.add_audio(pcm_silence)

    # Tail discard guard should discard lingering tail rather than emitting a cut
    assert len(cuts) == 0
    assert len(buf.audio_buffer) == 0


def test_chronos_cuts_at_acoustic_dip():
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 3.2s speech + 0.1s pause (dip) + 0.9s speech = 4.2s total (> 4.0s MAX_CONTINUOUS_BYTES)
    t1 = np.linspace(0, 3.2, int(3.2 * sr), dtype=np.float32)
    s1 = (0.3 * np.sin(2 * np.pi * 300 * t1) * 32767).astype(np.int16)
    dip = np.zeros(int(0.1 * sr), dtype=np.int16)
    t2 = np.linspace(0, 0.9, int(0.9 * sr), dtype=np.float32)
    s2 = (0.3 * np.sin(2 * np.pi * 300 * t2) * 32767).astype(np.int16)

    full_audio = np.concatenate([s1, dip, s2]).tobytes()
    cuts = buf.add_audio(full_audio)

    # Should cut at the acoustic dip rather than cutting mid-syllable at 4.0s
    assert len(cuts) >= 1
    cut = cuts[0]
    assert cut.cut_type == ChronosCutType.SOFT_CUT
    assert 96000 <= len(cut.pcm_data) <= 112000
