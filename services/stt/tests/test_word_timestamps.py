import pytest
import numpy as np
from ramo_listen.engines.sensevoice_engine import SenseVoiceEngine


@pytest.mark.asyncio
async def test_transcribe_returns_word_timestamps():
    engine = SenseVoiceEngine(sample_rate=16000)
    await engine.load()

    # 1.0 second speech signal
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    result = await engine.transcribe(audio, sample_rate=sr)

    assert "words" in result, "STT result missing 'words' list"
    assert isinstance(result["words"], list)
    if result["words"]:
        first_word = result["words"][0]
        assert "word" in first_word
        assert "start" in first_word
        assert "end" in first_word
        assert "confidence" in first_word
        assert first_word["start"] <= first_word["end"]
