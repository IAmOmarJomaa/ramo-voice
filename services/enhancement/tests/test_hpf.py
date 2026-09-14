import numpy as np
import pytest
from ramo_clean.hpf import HPFFilter


def test_hpf_attenuates_sub_bass():
    """Verify that a 40Hz sub-bass frequency is heavily attenuated (>18 dB reduction)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 40Hz sub-bass sine wave (e.g. HVAC / desk thump)
    sub_bass = np.sin(2 * np.pi * 40.0 * t).astype(np.float32)

    hpf = HPFFilter(cutoff_hz=80.0, sample_rate=sr, order=4)
    filtered = hpf.filter(sub_bass)

    input_rms = np.sqrt(np.mean(sub_bass**2))
    output_rms = np.sqrt(np.mean(filtered**2))
    
    # 20 * log10(output_rms / input_rms) should be < -18 dB
    reduction_db = 20 * np.log10(output_rms / input_rms)
    assert reduction_db < -18.0, f"40Hz signal was not sufficiently attenuated: {reduction_db} dB"


def test_hpf_passes_vocal_fundamental():
    """Verify that 440Hz vocal tone passes through with negligible attenuation (< 0.5 dB)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 440Hz vocal sine wave
    vocal = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    hpf = HPFFilter(cutoff_hz=80.0, sample_rate=sr, order=4)
    filtered = hpf.filter(vocal)

    # Discard filter transient at the first 100 samples
    steady_input = vocal[100:]
    steady_output = filtered[100:]

    input_rms = np.sqrt(np.mean(steady_input**2))
    output_rms = np.sqrt(np.mean(steady_output**2))
    loss_db = abs(20 * np.log10(output_rms / input_rms))
    assert loss_db < 0.5, f"440Hz vocal signal suffered excessive attenuation: {loss_db} dB"


def test_hpf_preserves_shape_and_dtype():
    sr = 16000
    audio = np.random.randn(2048).astype(np.float32) * 0.1
    hpf = HPFFilter(cutoff_hz=80.0, sample_rate=sr)
    out = hpf.filter(audio)

    assert out.shape == audio.shape
    assert out.dtype == np.float32
