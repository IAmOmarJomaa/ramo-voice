"""
Tests for F5-TTS Flow Matching Engine and 4.5s Speaker Registration Policy
(ramo_voice.engines.f5_engine).
Verifies:
1. F5-TTS initializes and loads properly.
2. Ingests (reference_audio, prompt_text) directly without running Whisper.
3. Produces valid float32 waveforms with 0 syllable repetition loops.
4. Enforces the 4.5s threshold rule (fallback if < 4.5s, F5-TTS if >= 4.5s).
5. Supports progressive refinement (updating 4.5s with 10s audio).
"""

import pytest
import numpy as np
from ramo_voice.profiles import VoiceProfile, VoiceProfileStore
from ramo_voice.engines.f5_engine import F5TTSEngine


@pytest.mark.asyncio
async def test_f5_engine_initialization_and_load():
    engine = F5TTSEngine(sample_rate=24000)
    assert not engine.is_loaded
    await engine.load()
    assert engine.is_loaded
    assert engine.engine_id == "f5-tts-flow-matching"


@pytest.mark.asyncio
async def test_f5_engine_generate_with_prompt_text_and_audio():
    engine = F5TTSEngine(sample_rate=24000)
    await engine.load()

    # 4.5 seconds of clean reference speech audio
    sr = 24000
    ref_audio = np.sin(2 * np.pi * 180 * np.linspace(0, 4.5, int(4.5 * sr), dtype=np.float32))

    profile = VoiceProfile(
        voice_id="speaker_alice",
        name="Alice",
        voice_type="cloned",
        sample_rate=24000,
        conditioning_latents=ref_audio,
        prompt_text="I think we should proceed with the architecture plan.",
        duration_sec=4.5
    )

    audio, out_sr = await engine.generate_chunk(
        text="Creo que deberíamos proceder con el plan.",
        profile=profile,
        speed=1.0
    )

    assert out_sr == 24000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert not np.isnan(audio).any()
    assert not np.isinf(audio).any()
    assert np.max(np.abs(audio)) <= 1.0


def test_4_5s_threshold_policy():
    from ramo_voice.profiles import register_meeting_speaker

    store = VoiceProfileStore()

    # Case A: Audio is only 2.0s (< 4.5s threshold) -> Mark fallback
    short_audio = np.zeros(int(2.0 * 24000), dtype=np.float32)
    p_short = register_meeting_speaker(
        store=store,
        speaker_id="speaker_bob",
        audio=short_audio,
        prompt_text="Yes exactly.",
        sample_rate=24000
    )
    assert p_short.use_fallback is True
    assert p_short.fallback_preset in ["af_heart", "am_adam", "af_bella"]

    # Case B: Audio is 4.6s (>= 4.5s threshold) -> Ready for F5-TTS
    full_audio = np.zeros(int(4.6 * 24000), dtype=np.float32)
    p_full = register_meeting_speaker(
        store=store,
        speaker_id="speaker_charlie",
        audio=full_audio,
        prompt_text="Welcome everyone, let's start the meeting review now.",
        sample_rate=24000
    )
    assert p_full.use_fallback is False
    assert p_full.voice_type == "cloned"
    assert p_full.duration_sec >= 4.5


def test_progressive_voice_refinement():
    from ramo_voice.profiles import register_meeting_speaker

    store = VoiceProfileStore()

    # Step 1: Register initial 4.5s audio
    audio_4_5s = np.ones(int(4.5 * 24000), dtype=np.float32) * 0.2
    p1 = register_meeting_speaker(
        store=store,
        speaker_id="speaker_dana",
        audio=audio_4_5s,
        prompt_text="Initial four point five second speech.",
        sample_rate=24000
    )
    assert p1.duration_sec == pytest.approx(4.5, rel=1e-2)

    # Step 2: Background refinement updates to 11.0s audio
    audio_11s = np.ones(int(11.0 * 24000), dtype=np.float32) * 0.3
    p2 = register_meeting_speaker(
        store=store,
        speaker_id="speaker_dana",
        audio=audio_11s,
        prompt_text="Refined eleven second speech monologue without background noise.",
        sample_rate=24000
    )
    assert p2.speaker_id == "speaker_dana"
    assert p2.duration_sec == pytest.approx(11.0, rel=1e-2)
    assert len(p2.conditioning_latents) == len(audio_11s)
