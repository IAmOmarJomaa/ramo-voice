"""
ramo_speaker.cluster
====================
Online speaker clustering with StenoAI crosstalk exclusion and rolling centroid tracking.
Prevents speaker voiceprint corruption caused by overlapping speech intervals.
"""

from typing import Dict, Optional
import numpy as np


class SpeakerClusterer:
    """
    Online cosine-distance speaker clusterer.
    Applies StenoAI's crosstalk rejection: overlapping segments are NEVER used
    to update the rolling speaker centroid.
    """

    def __init__(self, similarity_threshold: float = 0.75, momentum: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.momentum = momentum
        self._centroids: Dict[str, np.ndarray] = {}

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v) + 1e-9
        return (v / norm).astype(np.float32)

    def assign_or_update(self, embedding: np.ndarray, is_overlap: bool = False) -> str:
        """
        Assign an embedding to an existing speaker or initialize a new speaker.
        If is_overlap is True, assigns to closest speaker but skips centroid updating.
        """
        vec = self._normalize(embedding)

        if not self._centroids:
            spk_id = "SPEAKER_00"
            self._centroids[spk_id] = vec
            return spk_id

        # Compute cosine similarity with all known speaker centroids
        best_spk: Optional[str] = None
        best_sim = -1.0

        for spk_id, centroid in self._centroids.items():
            sim = float(np.dot(centroid, vec))
            if sim > best_sim:
                best_sim = sim
                best_spk = spk_id

        if best_sim >= self.similarity_threshold and best_spk is not None:
            # Match existing speaker
            if not is_overlap:
                # Update rolling centroid with exponential moving average
                updated = self.momentum * self._centroids[best_spk] + (1.0 - self.momentum) * vec
                self._centroids[best_spk] = self._normalize(updated)
            return best_spk
        else:
            # Initialize new speaker
            new_id = f"SPEAKER_{len(self._centroids):02d}"
            if not is_overlap:
                self._centroids[new_id] = vec
            return new_id

    def get_centroid(self, speaker_id: str) -> Optional[np.ndarray]:
        """Return the current unit-normalized centroid of a speaker."""
        return self._centroids.get(speaker_id)

    def get_speakers(self) -> Dict[str, np.ndarray]:
        return dict(self._centroids)
