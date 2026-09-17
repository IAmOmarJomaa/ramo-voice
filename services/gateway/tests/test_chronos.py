"""
services.gateway.tests.test_chronos
===================================
TDD Unit tests for ChronosBuffer:
- 10.0s Rolling Sliding Window (WINDOW_MAX_SECS = 10, WINDOW_MAX_BYTES = 320,000)
- 1.0s Emit Pace (EMIT_EVERY_BYTES = 32,000)
- LocalAgreement (n=2) consensus stabilization
- Punctuation-based buffer trimming on [.!?]
- No premature 7.0s monologue hard cutoff
- Flush on EOS commits tentative tokens and purges buffer
"""

import pytest
import numpy as np
from ramo_gateway.chronos import ChronosBuffer, ChronosCutType, LocalAgreement


def test_chronos_emit_pace_1s():
    """Verify that cuts are emitted every 1.0s (32,000 bytes) of new audio."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # 0.5s audio (16,000 bytes) -> Not enough to trigger 1.0s emit pace
    pcm_500ms = np.zeros(int(0.5 * sr), dtype=np.int16).tobytes()
    cuts = buf.add_audio(pcm_500ms)
    assert len(cuts) == 0

    # Additional 0.5s audio (total 32,000 bytes = 1.0s) -> Should emit sliding window cut
    cuts2 = buf.add_audio(pcm_500ms)
    assert len(cuts2) == 1
    assert cuts2[0].cut_type == ChronosCutType.SLIDING_WINDOW
    assert cuts2[0].is_final is False
    assert len(cuts2[0].pcm_data) == 32000


def test_chronos_sliding_window_capped_at_10s():
    """Verify sliding window does not exceed 10.0s (320,000 bytes) even after 15s of audio."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # Feed 15 seconds of audio in 1-second chunks
    one_sec_pcm = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32)) * 32767).astype(np.int16).tobytes()
    last_cut = None
    for _ in range(15):
        cuts = buf.add_audio(one_sec_pcm)
        if cuts:
            last_cut = cuts[-1]

    assert last_cut is not None
    # Maximum window must be capped at 10.0s (320,000 bytes)
    assert len(last_cut.pcm_data) == 320000
    # Buffer contains full uncommitted history until trimmed
    assert len(buf.audio_buffer) >= 320000


def test_local_agreement_n2_consensus():
    """Verify words are committed only after appearing identically in 2 consecutive hypotheses."""
    la = LocalAgreement(n_agreement=2)

    # Window 1
    res1 = la.step("so it is the sec meaning")
    assert res1.committed == ""
    assert res1.tentative == "so it is the sec meaning"

    # Window 2: Prefix "so it is the sec" matches
    res2 = la.step("so it is the sec meaning secure and govern")
    assert res2.committed == "so it is the sec meaning"
    assert res2.tentative == "secure and govern"

    # Window 3: More words match
    res3 = la.step("so it is the sec meaning secure and govern growth")
    assert res3.committed == "so it is the sec meaning secure and govern"
    assert res3.tentative == "growth"


def test_chronos_punctuation_trimming():
    """Verify that when committed text ends with [.!?], Chronos trims the confirmed audio."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # Feed 5 seconds of audio
    five_sec_pcm = np.ones(int(5.0 * sr), dtype=np.int16).tobytes()
    buf.add_audio(five_sec_pcm)
    initial_len = len(buf.audio_buffer)
    assert initial_len == int(5.0 * sr * 2)

    # Simulate STT committing a sentence ending in '.' at timestamp 3.5s
    words = [
        {"word": "Hello", "start": 0.5, "end": 1.0},
        {"word": "world.", "start": 1.5, "end": 3.5},
    ]
    trimmed_bytes = buf.trim_on_punctuation(text="Hello world.", words=words)

    assert trimmed_bytes > 0
    assert len(buf.audio_buffer) < initial_len
    # Audio buffer should now retain only the remaining ~1.5s (+ small safety overlap)
    expected_remaining = int((5.0 - 3.5) * sr * 2)
    assert abs(len(buf.audio_buffer) - expected_remaining) <= int(0.3 * sr * 2)


def test_monologue_no_hard_7s_cutoff():
    """Verify unbroken monologue is NOT chopped into hard cuts at 7.0 seconds."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    # Feed 8.5 seconds of unbroken audio
    t = np.linspace(0, 8.5, int(8.5 * sr), dtype=np.float32)
    speech_pcm = (0.3 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16).tobytes()

    # Feed in 0.5s chunks
    chunk_size = int(0.5 * sr * 2)
    all_cuts = []
    for i in range(0, len(speech_pcm), chunk_size):
        chunk = speech_pcm[i:i + chunk_size]
        cuts = buf.add_audio(chunk)
        all_cuts.extend(cuts)

    # Should have emitted 8 sliding window cuts (every 1.0s)
    # NONE of them should be a HARD_CUT or premature FINAL cut!
    assert len(all_cuts) == 8
    for cut in all_cuts:
        assert cut.cut_type == ChronosCutType.SLIDING_WINDOW
        assert cut.is_final is False


def test_chronos_flush_on_eos():
    """Verify flush on EOS produces a final flush and clears buffer."""
    buf = ChronosBuffer(sample_rate=16000)
    sr = 16000

    pcm = np.ones(int(2.5 * sr), dtype=np.int16).tobytes()
    buf.add_audio(pcm)

    cut = buf.flush()
    assert cut is not None
    assert cut.cut_type == ChronosCutType.FORCED_FLUSH
    assert cut.is_final is True
    assert len(buf.audio_buffer) == 0
