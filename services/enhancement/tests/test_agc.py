import numpy as np
import pytest
from ramo_clean.agc import AGCLeveler


def test_agc_boosts_quiet_speech():
    """Verify that a quiet audio signal (-40 dBFS) is amplified towards target (-20 dBFS)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # -40 dBFS amplitude is 10^(-40/20) = 0.01
    quiet_signal = (0.01 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    agc = AGCLeveler(target_dbfs=-20.0, max_gain_db=18.0)
    leveled = agc.process(quiet_signal)

    in_rms = np.sqrt(np.mean(quiet_signal**2))
    out_rms = np.sqrt(np.mean(leveled**2))

    assert out_rms > in_rms * 2.0, "AGC failed to boost quiet signal"
    assert np.max(np.abs(leveled)) <= 1.0, "AGC allowed clipping"


def test_agc_prevents_clipping_on_loud_signal():
    """Verify that a loud signal near 0 dBFS is softly limited without digital clipping."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 0.95 amplitude sine wave
    loud_signal = (0.95 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    agc = AGCLeveler(target_dbfs=-20.0, max_gain_db=15.0)
    leveled = agc.process(loud_signal)

    assert np.max(np.abs(leveled)) <= 1.0, "Signal clipped > 1.0"
    assert leveled.dtype == np.float32
