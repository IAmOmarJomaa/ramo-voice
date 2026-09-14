import time
import numpy as np
import pytest
from ramo_clean.pipeline import AudioPreconditioner


def test_pipeline_end_to_end():
    """Verify that AudioPreconditioner executes full 5-stage cleaning and outputs valid audio."""
    preconditioner = AudioPreconditioner(sample_rate=16000)

    # 1 second of audio: sub-bass rumble (40Hz) + quiet speech tone (400Hz) + stationary white noise
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sub_bass = 0.4 * np.sin(2 * np.pi * 40 * t)
    speech = 0.05 * np.sin(2 * np.pi * 400 * t)
    noise = np.random.randn(sr) * 0.01
    dirty_audio = (sub_bass + speech + noise).astype(np.float32)

    start = time.perf_counter()
    result = preconditioner.process_chunk(dirty_audio)
    elapsed = time.perf_counter() - start

    assert result.audio is not None
    assert len(result.audio) == len(dirty_audio)
    assert result.audio.dtype == np.float32
    # Benchmark: 1 second audio should process in < 25ms on CPU (actual < 2ms)
    assert elapsed < 0.100, f"Processing too slow: {elapsed*1000:.1f}ms"
    assert np.max(np.abs(result.audio)) <= 1.0
