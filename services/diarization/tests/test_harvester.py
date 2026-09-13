"""
Tests for VoiceprintHarvester in ramo_speaker.harvester.
Extracts clean, non-overlapping reference samples for zero-shot cloning.
"""

import pytest
import numpy as np
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn


def test_harvest_clean_voice_sample():
    harvester = VoiceprintHarvester(min_duration_sec=2.5, target_sr=16000)

    sr = 16000
    # Turn 1: Speaker 0 speaks for 1 second (too short)
    turn1 = SpeakerTurn(
        speaker_id="SPEAKER_00",
        start_sec=0.0,
        end_sec=1.0,
        audio=np.ones(16000, dtype=np.float32)
    )

    # Turn 2: Speaker 0 speaks for 3.5 seconds (qualifies for voiceprint harvest)
    turn2 = SpeakerTurn(
        speaker_id="SPEAKER_00",
        start_sec=2.0,
        end_sec=5.5,
        audio=np.ones(int(3.5 * sr), dtype=np.float32) * 0.5
    )

    harvester.add_turn(turn1)
    harvester.add_turn(turn2)

    harvested = harvester.get_best_sample("SPEAKER_00")
    assert harvested is not None
    assert len(harvested) >= int(2.5 * sr)
    assert harvester.has_candidate("SPEAKER_00")
