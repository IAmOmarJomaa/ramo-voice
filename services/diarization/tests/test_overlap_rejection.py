"""
services/diarization/tests/test_overlap_rejection.py
===================================================
TDD tests for Acoustic Overlap Detection and strict Voice Harvester crosstalk rejection.
"""

import pytest
import numpy as np
from ramo_speaker.overlap import AcousticOverlapDetector
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn


def test_overlap_detector_clean_vs_crosstalk():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    detector = AcousticOverlapDetector(sample_rate=sr)

    # 1. Clean single speaker (dominant pitch at 150 Hz + harmonics)
    clean_audio = (
        0.5 * np.sin(2 * np.pi * 150 * t)
        + 0.25 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
    )
    is_overlap, score = detector.detect(clean_audio)
    assert not is_overlap
    assert score < 0.50

    # 2. Crosstalk / overlapping speakers (two independent pitches 130 Hz and 210 Hz at equal power)
    crosstalk_audio = (
        0.4 * np.sin(2 * np.pi * 130 * t)
        + 0.4 * np.sin(2 * np.pi * 210 * t)
        + 0.2 * np.random.randn(len(t)).astype(np.float32) * 0.1
    )
    is_overlap_cross, cross_score = detector.detect(crosstalk_audio)
    assert is_overlap_cross
    assert cross_score >= 0.50


def test_harvester_rejects_overlap_and_low_snr():
    sr = 16000
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, tier2_threshold_sec=10.0, min_snr_db=12.0)

    audio_clean_3s = np.sin(2 * np.pi * 200 * np.linspace(0, 3.0, int(sr * 3.0), endpoint=False, dtype=np.float32))

    # Turn 1: Clean turn -> Accepted
    turn1 = SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=0.0,
        end_sec=3.0,
        audio=audio_clean_3s,
        transcript="Hello and welcome to the team meeting today.",
        is_overlap=False,
    )
    harvester.add_turn(turn1)
    prof = harvester.get_profile("SPEAKER_01")
    assert prof is not None
    assert prof.duration_sec == 3.0
    assert not prof.ready_for_cloning

    # Turn 2: Overlapping crosstalk turn -> Strictly REJECTED
    turn2_overlap = SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=3.0,
        end_sec=5.5,
        audio=np.random.randn(int(sr * 2.5)).astype(np.float32) * 0.5,
        transcript="Yeah I agree with that completely.",
        is_overlap=True,
    )
    harvester.add_turn(turn2_overlap)
    # Duration must remain unchanged at 3.0s!
    prof = harvester.get_profile("SPEAKER_01")
    assert prof.duration_sec == 3.0

    # Turn 3: Clean turn (2.0s) -> Accepted, total 5.0s -> Reaches Tier 1 ready_for_cloning!
    audio_clean_2s = np.sin(2 * np.pi * 200 * np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32))
    turn3 = SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=5.5,
        end_sec=7.5,
        audio=audio_clean_2s,
        transcript="Let's review the financial numbers next.",
        is_overlap=False,
    )
    harvester.add_turn(turn3)
    prof = harvester.get_profile("SPEAKER_01")
    assert prof.duration_sec == 5.0
    assert prof.ready_for_cloning
    assert prof.tier == 1
