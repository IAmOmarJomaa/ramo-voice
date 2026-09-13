"""
ramo_speaker.harvester
======================
Voiceprint Harvester: extracts isolated clean speech monologues (>=3s)
suitable for immediate, zero-ASR zero-shot voice cloning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SpeakerTurn:
    speaker_id: str
    start_sec: float
    end_sec: float
    audio: np.ndarray
    is_overlap: bool = False

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class VoiceprintHarvester:
    """
    Tracks and filters speech turns to harvest the cleanest, longest isolated monologue
    for each speaker in a multi-party conversation.
    """

    def __init__(self, min_duration_sec: float = 2.5, target_sr: int = 16000):
        self.min_duration_sec = min_duration_sec
        self.target_sr = target_sr
        # Mapping: speaker_id -> list of candidate monologue audio arrays
        self._candidates: Dict[str, List[np.ndarray]] = {}

    def add_turn(self, turn: SpeakerTurn) -> None:
        """Add a speech turn. Excludes turns marked as crosstalk/overlap."""
        if turn.is_overlap:
            # Overlapping turns cannot be used for clean voice cloning
            return

        if turn.duration_sec >= self.min_duration_sec:
            if turn.speaker_id not in self._candidates:
                self._candidates[turn.speaker_id] = []
            
            # Normalize audio waveform
            audio = turn.audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = (audio / max_abs) * 0.95

            self._candidates[turn.speaker_id].append(audio)

    def has_candidate(self, speaker_id: str) -> bool:
        return speaker_id in self._candidates and len(self._candidates[speaker_id]) > 0

    def get_best_sample(self, speaker_id: str) -> Optional[np.ndarray]:
        """
        Return the longest and cleanest candidate sample for the specified speaker.
        """
        samples = self._candidates.get(speaker_id, [])
        if not samples:
            return None
        # Return longest isolated monologue
        return max(samples, key=len)

    def list_harvested_speakers(self) -> List[str]:
        return [spk for spk, samples in self._candidates.items() if len(samples) > 0]
