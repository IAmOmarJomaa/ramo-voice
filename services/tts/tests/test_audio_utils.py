import numpy as np
import pytest
from ramo_voice.audio_utils import has_tts_runaway, trim_tts_output, normalize_audio


def test_has_tts_runaway_clean_speech():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert has_tts_runaway(audio, sample_rate=sr) is False


def test_has_tts_runaway_hallucination_pattern():
    sr = 24000
    speech = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5)))).astype(np.float32)
    silence = np.zeros(int(sr * 2.5), dtype=np.float32)
    noise = (0.2 * np.random.randn(int(sr * 0.5))).astype(np.float32)

    audio = np.concatenate([speech, silence, noise])
    assert has_tts_runaway(audio, sample_rate=sr, max_internal_silence_ms=2000) is True


def test_trim_tts_output():
    sr = 24000
    speech = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))).astype(np.float32)
    trailing_silence = np.zeros(int(sr * 1.5), dtype=np.float32)
    audio = np.concatenate([speech, trailing_silence])

    trimmed = trim_tts_output(audio, sample_rate=sr, min_silence_ms=200)
    assert len(trimmed) < len(audio)
    assert len(trimmed) >= sr


def test_normalize_audio():
    audio = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    norm = normalize_audio(audio, target_db=0.0)
    assert np.isclose(np.max(np.abs(norm)), 1.0, atol=1e-5)
