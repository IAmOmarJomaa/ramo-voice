import numpy as np
import pytest
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn, HarvestedProfile


def test_harvester_tier1_activation_and_transcript_retention():
    """Verify that reaching 4.5s activates Tier 1 and retains transcript."""
    sr = 16000
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, tier2_threshold_sec=10.0, target_sr=sr)

    # 1. Add 2.0s turn with transcript
    harvester.add_turn(SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=0.0,
        end_sec=2.0,
        audio=np.ones(int(2.0 * sr), dtype=np.float32) * 0.4,
        transcript="Good morning everyone.",
        is_overlap=False,
    ))

    profile = harvester.get_profile("SPEAKER_01")
    assert profile is not None
    assert profile.tier == 0
    assert not profile.ready_for_cloning  # < 4.5s

    # 2. Add 2.8s turn with transcript -> total 4.8s >= 4.5s Tier 1
    harvester.add_turn(SpeakerTurn(
        speaker_id="SPEAKER_01",
        start_sec=3.0,
        end_sec=5.8,
        audio=np.ones(int(2.8 * sr), dtype=np.float32) * 0.4,
        transcript="Let's begin the review.",
        is_overlap=False,
    ))

    profile = harvester.get_profile("SPEAKER_01")
    assert profile.tier == 1
    assert profile.ready_for_cloning
    assert profile.duration_sec >= 4.5
    assert "Good morning everyone" in profile.transcript
    assert "Let's begin the review" in profile.transcript


def test_harvester_tier2_progressive_upgrade():
    """Verify that accumulating 10.0s upgrades profile to Tier 2."""
    sr = 16000
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, tier2_threshold_sec=10.0, target_sr=sr)

    # Add 5.0s turn (Tier 1)
    harvester.add_turn(SpeakerTurn(
        speaker_id="SPEAKER_02",
        start_sec=0.0,
        end_sec=5.0,
        audio=np.ones(int(5.0 * sr), dtype=np.float32) * 0.3,
        transcript="First part of discussion.",
    ))
    assert harvester.get_profile("SPEAKER_02").tier == 1

    # Add 5.5s turn -> total 10.5s >= 10.0s (Tier 2)
    harvester.add_turn(SpeakerTurn(
        speaker_id="SPEAKER_02",
        start_sec=6.0,
        end_sec=11.5,
        audio=np.ones(int(5.5 * sr), dtype=np.float32) * 0.3,
        transcript="Second part of discussion.",
    ))
    p2 = harvester.get_profile("SPEAKER_02")
    assert p2.tier == 2
    assert p2.ready_for_cloning
    assert p2.duration_sec >= 10.0
    assert "First part" in p2.transcript and "Second part" in p2.transcript


def test_harvester_excludes_overlapping_turns():
    """Verify that overlapping crosstalk turns are ignored by harvester."""
    sr = 16000
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, target_sr=sr)

    # Overlapping 5.0s turn
    harvester.add_turn(SpeakerTurn(
        speaker_id="SPEAKER_03",
        start_sec=0.0,
        end_sec=5.0,
        audio=np.ones(int(5.0 * sr), dtype=np.float32) * 0.5,
        transcript="Crosstalk chatter.",
        is_overlap=True,
    ))

    assert harvester.get_profile("SPEAKER_03") is None
