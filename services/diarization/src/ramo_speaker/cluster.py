"""
ramo_speaker.cluster
====================
Online speaker clustering with StenoAI crosstalk exclusion and rolling centroid tracking.
Prevents speaker voiceprint corruption caused by overlapping speech intervals.
Default threshold 0.62 and momentum 0.70 provide optimal speaker discrimination for CampPlus embeddings.
"""

import logging
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger("ramo_speaker.cluster")


class SpeakerClusterer:
    """
    Online cosine-distance speaker clusterer.
    Applies StenoAI's crosstalk rejection: overlapping segments are NEVER used
    to update the rolling speaker centroid.
    """

    def __init__(self, similarity_threshold: float = 0.62, momentum: float = 0.70):
        self.similarity_threshold = similarity_threshold
        self.momentum = momentum
        self._centroids: Dict[str, np.ndarray] = {}

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v)) + 1e-9
        return (v / norm).astype(np.float32)

    def assign_or_update(self, embedding: np.ndarray, is_overlap: bool = False) -> str:
        """
        Assign an embedding to an existing speaker or initialize a new speaker.
        If is_overlap is True, assigns to closest speaker but skips centroid updating.
        Emits transparent diagnostic logs detailing each centroid comparison.
        """
        vec = self._normalize(embedding)

        if not self._centroids:
            spk_id = "SPEAKER_00"
            self._centroids[spk_id] = vec
            logger.info(
                f"[DIAR_CLUSTER] 🌟 First speaker registered: '{spk_id}' (Embedding dim: {len(vec)}, Centroid initialized)"
            )
            return spk_id

        # Compute cosine similarity with all known speaker centroids
        best_spk: Optional[str] = None
        best_sim = -1.0
        comparisons: List[str] = []

        for spk_id, centroid in self._centroids.items():
            sim = float(np.dot(centroid, vec))
            match_str = "MATCH!" if sim >= self.similarity_threshold else "NO MATCH"
            comparisons.append(f"'{spk_id}': cos_sim={sim:.4f} [{match_str}]")
            if sim > best_sim:
                best_sim = sim
                best_spk = spk_id

        logger.info(
            f"[DIAR_CLUSTER] 👥 Evaluating turn vector against {len(self._centroids)} centroid(s) (Threshold: {self.similarity_threshold:.2f}):\n"
            + "\n".join(f"  -> Centroid {c}" for c in comparisons)
        )

        if best_sim >= self.similarity_threshold and best_spk is not None:
            # Match existing speaker
            if is_overlap:
                logger.info(
                    f"[DIAR_CLUSTER] ⚠️ Overlapping crosstalk turn assigned to '{best_spk}' (sim={best_sim:.4f}), centroid preserved without update."
                )
            else:
                old_c = self._centroids[best_spk].copy()
                updated = self.momentum * old_c + (1.0 - self.momentum) * vec
                new_c = self._normalize(updated)
                drift = float(np.linalg.norm(new_c - old_c))
                self._centroids[best_spk] = new_c
                logger.info(
                    f"[DIAR_CLUSTER] ✅ MATCH -> Assigned to '{best_spk}' (sim={best_sim:.4f} >= {self.similarity_threshold:.2f}) | "
                    f"EMA centroid updated (momentum={self.momentum:.2f}, drift={drift:.4f})"
                )
            return best_spk
        else:
            # Initialize new speaker
            new_id = f"SPEAKER_{len(self._centroids):02d}"
            if not is_overlap:
                self._centroids[new_id] = vec
            active_list = list(self._centroids.keys())
            logger.info(
                f"[DIAR_CLUSTER] 🌟 Max similarity {best_sim:.4f} < {self.similarity_threshold:.2f} -> INITIALIZING NEW SPEAKER: '{new_id}' | "
                f"Active centroids ({len(active_list)}): {active_list}"
            )
            return new_id

    def get_centroid(self, speaker_id: str) -> Optional[np.ndarray]:
        """Return the current unit-normalized centroid of a speaker."""
        return self._centroids.get(speaker_id)

    def get_speakers(self) -> Dict[str, np.ndarray]:
        return dict(self._centroids)
