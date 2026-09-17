"""
tests.test_silence_hallucination
================================
TDD test verifying that silence, near-silence, and low-energy background noise
do not trigger autoregressive hallucination loops even when given a poisoned initial_prompt.
"""

import pytest
import numpy as np
from ramo_listen.engines.whisper_engine import WhisperSTTEngine
from ramo_gateway.chronos import LiveUtterancePipeline


@pytest.mark.asyncio
async def test_whisper_rejects_silence_with_poisoned_prompt():
    """Digital silence must return empty text even if initial_prompt is populated with past turns."""
    engine = WhisperSTTEngine(sample_rate=16000)
    await engine.load()

    sr = 16000
    # 3.0s of digital silence (zeros)
    silence = np.zeros(int(3.0 * sr), dtype=np.float32)
    poisoned_prompt = "time. Based on feedback, I'm going to work to do more summarized communications to improve the"

    res = await engine.transcribe(silence, sample_rate=sr, initial_prompt=poisoned_prompt)
    assert res["text"] == "", f"Expected empty transcript for silence, got: '{res['text']}'"
    assert res["raw_text"] == "", f"Expected empty raw_text for silence, got: '{res['raw_text']}'"
    assert len(res["words"]) == 0


@pytest.mark.asyncio
async def test_whisper_rejects_low_energy_noise_with_poisoned_prompt():
    """Low-energy ambient noise (RMS < 0.008) must be gated out without invoking autoregressive Whisper decoding."""
    engine = WhisperSTTEngine(sample_rate=16000)
    await engine.load()

    sr = 16000
    # 2.5s of low-amplitude white noise (RMS ~ 0.004)
    rng = np.random.default_rng(42)
    low_noise = (rng.standard_normal(int(2.5 * sr)) * 0.004).astype(np.float32)
    poisoned_prompt = "time. Based on feedback, I'm going to work to do more summarized communications to improve the"

    res = await engine.transcribe(low_noise, sample_rate=sr, initial_prompt=poisoned_prompt)
    assert res["text"] == "", f"Expected empty transcript for low noise, got: '{res['text']}'"
    assert res["raw_text"] == "", f"Expected empty raw_text for low noise, got: '{res['raw_text']}'"


def test_chronos_pipeline_does_not_pollute_context_with_dialogue():
    """LiveUtterancePipeline context_prompt must only contain technical vocabulary hints, not rolling dialogue."""
    pipeline = LiveUtterancePipeline(sample_rate=16000)
    # Default context prompt should be clean
    assert "Based on feedback" not in pipeline.context_prompt
