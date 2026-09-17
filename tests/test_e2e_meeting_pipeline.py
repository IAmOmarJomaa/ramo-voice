"""
test_e2e_meeting_pipeline.py
============================
End-to-End simulation of the sovereign 4-stage ramO Meeting Translation Pipeline:
1. Audio Preconditioning: 80Hz HPF + Spectral Gating + AGC + VAD (ramo_clean)
2. Speech-to-Text: SenseVoice transcription with word timestamps and emotion tags (ramo_listen)
3. Diarization & Harvester: 2-Tier voiceprint accumulation (4.5s threshold) & crosstalk rejection (ramo_speaker)
4. Translation simulation: Speaker attribution & translated text generation
5. Voice Cloning & Synthesis: F5-TTS Flow-Matching synthesis with pre-transcribed text or Supertonic fallback (ramo_voice)
"""

import asyncio
import pytest
import numpy as np

from ramo_clean.pipeline import AudioPreconditioner
from ramo_listen.engines.whisper_engine import WhisperSTTEngine
from ramo_speaker.cluster import SpeakerClusterer
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn
from ramo_voice.profiles import VoiceProfileStore, register_meeting_speaker
from ramo_voice.engines.supertonic_engine import SupertonicEngine
from ramo_voice.engines.f5_engine import F5TTSEngine
from ramo_voice.server import get_engine_for_profile
from ramo_translate.router import TranslationRouter


@pytest.mark.asyncio
async def test_end_to_end_meeting_translation_flow():
    sr = 16000

    # 1. Initialize Pipeline Microservice Components
    cleaner = AudioPreconditioner(sample_rate=sr)
    stt_engine = WhisperSTTEngine(sample_rate=sr)
    await stt_engine.load()

    clusterer = SpeakerClusterer(similarity_threshold=0.75, momentum=0.85)
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, target_sr=sr)
    profile_store = VoiceProfileStore()
    fallback_tts = SupertonicEngine()
    cloning_tts = F5TTSEngine()
    await fallback_tts.load()
    await cloning_tts.load()

    # 2. Simulate Speaker Alice speaking for 5.0 seconds with background AC noise
    t_alice = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False, dtype=np.float32)
    # 40Hz HVAC rumble + 400Hz Alice vocal tone + white noise
    raw_alice_audio = (
        0.3 * np.sin(2 * np.pi * 40 * t_alice)
        + 0.4 * np.sin(2 * np.pi * 400 * t_alice)
        + np.random.randn(len(t_alice)) * 0.02
    ).astype(np.float32)

    # Step 1: Precondition Audio (Clean rumble, normalize, gate)
    clean_alice = cleaner.process_chunk(raw_alice_audio).audio
    assert clean_alice is not None
    assert np.max(np.abs(clean_alice)) <= 1.0

    # Step 2: STT Transcribe with word timestamps & emotion
    stt_result = await stt_engine.transcribe(clean_alice, sample_rate=sr)
    assert "words" in stt_result
    assert len(stt_result["words"]) > 0
    alice_transcript = stt_result["raw_text"]
    assert len(alice_transcript) > 0

    # Step 3: Diarization & Embedding Clustering
    # Create distinct synthetic embedding for Alice
    alice_emb = np.array([0.9, 0.1, 0.05, 0.0], dtype=np.float32)
    alice_spk_id = clusterer.assign_or_update(alice_emb, is_overlap=False)
    assert alice_spk_id == "SPEAKER_00"

    # Step 4: Harvester Accumulation (Tier 1 reached: 5.0s >= 4.5s)
    harvester.add_turn(SpeakerTurn(
        speaker_id=alice_spk_id,
        start_sec=0.0,
        end_sec=5.0,
        audio=clean_alice,
        transcript=alice_transcript,
        is_overlap=False,
    ))

    profile = harvester.get_profile(alice_spk_id)
    assert profile is not None
    assert profile.tier == 1
    assert profile.ready_for_cloning
    assert profile.duration_sec >= 4.5
    # Critical: Transcript is permanently preserved (zero-ASR rule)
    assert profile.transcript == alice_transcript

    # Step 5: Register Speaker Alice with Microservice 4 (ramo_voice)
    alice_profile = register_meeting_speaker(
        store=profile_store,
        speaker_id=alice_spk_id,
        audio=profile.audio,
        prompt_text=profile.transcript,
        sample_rate=sr,
    )
    assert alice_profile.voice_type == "cloned"
    assert not alice_profile.use_fallback

    # Step 6: Real LLM Translation using TranslationRouter (Microservice 3)
    trans_router = TranslationRouter()
    alice_trans = await trans_router.translate(
        text="Hello everyone.",
        source_lang="en",
        target_lang="fr",
        session_id="meeting_e2e",
        speaker_id=alice_spk_id,
    )
    assert alice_trans.translated_text == "Bonjour tout le monde."
    assert alice_trans.speaker_symbol == "[S_A]"

    # Test conversational fast-bypass in meeting context (0 ms)
    fast_turn = await trans_router.translate(
        text="Okay.",
        source_lang="en",
        target_lang="fr",
        session_id="meeting_e2e",
        speaker_id=alice_spk_id,
    )
    assert fast_turn.is_bypass
    assert fast_turn.translated_text == "D'accord."

    # Step 7: Verify router selects F5-TTS for Alice (because >= 4.5s and prompt_text present)
    alice_engine = get_engine_for_profile(alice_profile)
    assert isinstance(alice_engine, F5TTSEngine)

    # Synthesize translated speech in Alice's cloned voice via F5-TTS
    synth_audio, out_sr = await alice_engine.generate_chunk(
        text=alice_trans.translated_text,
        profile=alice_profile,
    )
    assert synth_audio is not None
    assert len(synth_audio) > 0
    assert out_sr == alice_engine.sample_rate

    # Step 8: Simulate Speaker Bob with < 4.5s speech (Fallback to distinct preset)
    bob_turn = SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=6.0,
        end_sec=8.0,
        audio=np.ones(int(2.0 * sr), dtype=np.float32) * 0.3,
        transcript="I agree with Alice.",
    )
    harvester.add_turn(bob_turn)

    # Bob has only 2.0s (< 4.5s)
    bob_profile_harvested = harvester.get_profile("SPEAKER_01")
    assert not bob_profile_harvested.ready_for_cloning

    # Bob registered in profile store -> maps to instant fallback preset
    bob_profile = register_meeting_speaker(
        store=profile_store,
        speaker_id="SPEAKER_01",
        audio=bob_profile_harvested.audio,
        prompt_text=bob_profile_harvested.transcript,
        sample_rate=sr,
    )
    assert bob_profile.use_fallback
    assert bob_profile.voice_type == "preset"
    assert bob_profile.fallback_preset is not None

    # Bob's speech translated via TranslationRouter
    bob_trans = await trans_router.translate(
        text="I agree with Alice.",
        source_lang="en",
        target_lang="fr",
        session_id="meeting_e2e",
        speaker_id="SPEAKER_01",
    )
    assert bob_trans.translated_text == "Je suis d'accord avec Alice."

    # Verify router selects Supertonic for Bob (fallback)
    bob_engine = get_engine_for_profile(bob_profile)
    assert isinstance(bob_engine, SupertonicEngine)

    # Bob's translated speech synthesized via instant Supertonic preset
    bob_synth_audio, bob_sr = await bob_engine.generate_chunk(
        text=bob_trans.translated_text,
        profile=bob_profile,
    )
    assert bob_synth_audio is not None
    assert len(bob_synth_audio) > 0
    assert bob_sr == 44100
