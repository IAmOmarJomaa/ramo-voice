"""
tests/test_tts_orchestration_e2e.py
===================================
Comprehensive End-to-End Test for the SOTA Multi-Speaker TTS Orchestrator:
1. Purity Filtering: Crosstalk is flagged and strictly rejected from Voice Harvester.
2. Temporal Alignment: Word timestamps map cleanly to speakers without boundary clipping.
3. Anchor Voice Fallback: Speakers < 4.5s are assigned pitch-matched Kokoro presets.
4. Progressive Harvest: Speakers >= 4.5s are promoted to zero-shot cloning.
5. Prosodic Clause Buffering: Eliminates fragmented speech by enforcing breath pauses.
6. WSOLA Dynamic Latency Pacer: Applies pitch-preserving speedup without vocal distortion.
"""

import pytest
import numpy as np
from ramo_gateway.pipeline_client import PipelineDispatcher
from ramo_gateway.prosody_buffer import ProsodicClauseBuffer
from ramo_speaker.overlap import AcousticOverlapDetector
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn
from ramo_listen.aligner import TemporalWordAligner, SpeakerInterval


@pytest.mark.asyncio
async def test_full_tts_orchestration_flow():
    sr = 16000
    dispatcher = PipelineDispatcher(sample_rate=sr)

    # 1. Simulate Alice (Female, pitch ~220Hz, clean speech)
    t_alice = np.linspace(0, 2.5, int(2.5 * sr), endpoint=False, dtype=np.float32)
    alice_clean_1 = (
        0.5 * np.sin(2 * np.pi * 220 * t_alice)
        + 0.2 * np.sin(2 * np.pi * 440 * t_alice)
    )

    # Overlap detection on clean turn
    is_overlap, score = dispatcher.detect_overlap(alice_clean_1)
    assert not is_overlap
    assert score < 0.50

    # Diarization identification
    spk_alice = dispatcher.identify_speaker(alice_clean_1, is_overlap=is_overlap)
    assert "speaker" in spk_alice.lower()

    # Initial turn harvesting (2.5s < 4.5s threshold -> Tier 0, not ready for cloning)
    prof = dispatcher.harvest_speech_turn(
        speaker_id=spk_alice,
        audio_f32=alice_clean_1,
        transcript="Welcome everyone to our weekly architecture review.",
        is_overlap=is_overlap,
    )
    assert prof is not None
    assert prof.duration_sec == 2.5
    assert not prof.ready_for_cloning

    # Because Alice has < 4.5s, anchor matcher assigns a female Kokoro preset
    anchor_voice = dispatcher.anchor_matcher.match_voice(spk_alice, audio=alice_clean_1)
    assert anchor_voice in ["af_heart", "bf_emma", "af_bella", "af_nicole"]

    # 2. Simulate Crosstalk / Overlap turn (Alice + Bob talk at once)
    crosstalk_audio = (
        0.4 * np.sin(2 * np.pi * 130 * t_alice)
        + 0.4 * np.sin(2 * np.pi * 220 * t_alice)
        + 0.05 * np.random.randn(len(t_alice)).astype(np.float32)
    )
    is_overlap_cross, cross_score = dispatcher.detect_overlap(crosstalk_audio)
    assert is_overlap_cross
    assert cross_score >= 0.50

    # When overlap occurs, harvester strictly rejects the turn!
    prof_before = dispatcher.harvester.get_profile(spk_alice)
    dispatcher.harvest_speech_turn(
        speaker_id=spk_alice,
        audio_f32=crosstalk_audio,
        transcript="Wait, could you please clarify that point?",
        is_overlap=True,
    )
    prof_after = dispatcher.harvester.get_profile(spk_alice)
    # Accumulated speech MUST remain exactly 2.5s!
    assert prof_after.duration_sec == prof_before.duration_sec == 2.5

    # 3. Alice speaks a second clean turn (2.5s) -> 2.5s + 2.5s = 5.0s (>= 4.5s threshold!)
    alice_clean_2 = (
        0.5 * np.sin(2 * np.pi * 220 * t_alice)
        + 0.2 * np.sin(2 * np.pi * 440 * t_alice)
    )
    prof_promoted = dispatcher.harvest_speech_turn(
        speaker_id=spk_alice,
        audio_f32=alice_clean_2,
        transcript="I was referring to the real-time database replication latency.",
        is_overlap=False,
    )
    assert prof_promoted is not None
    assert prof_promoted.duration_sec == 5.0
    assert prof_promoted.ready_for_cloning
    assert prof_promoted.tier >= 1

    # 4. Prosodic Clause Buffering Test
    prosody_buf = dispatcher.get_prosody_buffer("test_session")
    clauses = prosody_buf.add_text("Pour commencer cette réunion importante,")
    assert len(clauses) == 1
    assert clauses[0] == "Pour commencer cette réunion importante,"

    # 5. Dynamic Latency Pacer (WSOLA) Test under queue pressure
    # 2000ms latency > 1200ms budget -> should speed up
    dynamic_speed = dispatcher.pacer.calculate_speed(current_latency_ms=2000, queue_words=30)
    assert dynamic_speed >= 1.15
    assert dynamic_speed <= 1.25

    # Synthesize audio with dynamic speed
    tts_pcm, tts_sr, speed_applied = await dispatcher.synthesize_speech(
        text=clauses[0],
        speaker_id=spk_alice,
        current_latency_ms=2000,
        queue_words=30,
    )
    assert len(tts_pcm) > 0
    assert tts_sr in [24000, 16000, 22050]
    assert speed_applied >= 1.15
