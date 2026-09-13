"""
Tests for Flow Matching Zero-Shot Cloning Engine (ramo_voice.engines.cloning_engine).
Verifies that:
1. Cloning engine initializes and loads properly.
2. Direct acoustic conditioning latents are ingested without ASR transcription.
3. Audio generation returns valid float32 waveforms with no NaNs/Infs.
4. Async streaming yields expected chunks.
5. Cloned voice profiles route to the cloning engine in the server pipeline.
"""

import pytest
import numpy as np
from ramo_voice.profiles import VoiceProfile, VoiceProfileStore
from ramo_voice.engines.cloning_engine import FlowMatchingCloningEngine


@pytest.mark.asyncio
async def test_cloning_engine_initialization_and_load():
    engine = FlowMatchingCloningEngine(sample_rate=24000)
    assert not engine.is_loaded
    await engine.load()
    assert engine.is_loaded
    assert engine.engine_id == "flow-matching-zero-shot"


@pytest.mark.asyncio
async def test_cloning_engine_generate_chunk_with_audio_latents():
    engine = FlowMatchingCloningEngine(sample_rate=24000)
    await engine.load()

    # Synthetic 3-second reference voice sample (sine sweep / vocal harmonics)
    ref_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 3, 3 * 24000, dtype=np.float32))

    profile = VoiceProfile(
        voice_id="speaker_test_01",
        name="Cloned Test Speaker",
        voice_type="cloned",
        sample_rate=24000,
        language="en",
        conditioning_latents=ref_audio
    )

    audio, sr = await engine.generate_chunk("This is a synthesized test sentence.", profile)
    assert sr == 24000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert not np.isnan(audio).any()
    assert not np.isinf(audio).any()
    assert np.max(np.abs(audio)) <= 1.0


@pytest.mark.asyncio
async def test_cloning_engine_streaming():
    engine = FlowMatchingCloningEngine(sample_rate=24000)
    await engine.load()

    ref_audio = np.random.uniform(-0.5, 0.5, 24000 * 3).astype(np.float32)
    profile = VoiceProfile(
        voice_id="speaker_stream_01",
        name="Streaming Speaker",
        voice_type="cloned",
        sample_rate=24000,
        language="en",
        conditioning_latents=ref_audio
    )

    chunks = []
    async for chunk in engine.generate_stream("Stream test phrase.", profile):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_audio = np.concatenate(chunks)
    assert not np.isnan(full_audio).any()
    assert len(full_audio) > 0


@pytest.mark.asyncio
async def test_engine_dispatcher_routing():
    from ramo_voice.server import get_engine_for_profile
    from ramo_voice.engines.supertonic_engine import SupertonicEngine

    store = VoiceProfileStore()
    preset_p = VoiceProfile(voice_id="preset_1", name="Preset", voice_type="preset")
    cloned_p = VoiceProfile(
        voice_id="cloned_1",
        name="Cloned",
        voice_type="cloned",
        conditioning_latents=np.zeros(16000, dtype=np.float32)
    )
    store.register(preset_p)
    store.register(cloned_p)

    engine_preset = get_engine_for_profile(preset_p)
    engine_cloned = get_engine_for_profile(cloned_p)

    assert isinstance(engine_preset, SupertonicEngine)
    assert isinstance(engine_cloned, FlowMatchingCloningEngine)
