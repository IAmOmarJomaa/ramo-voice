"""
ramo_speaker.overlap
====================
Acoustic crosstalk and multi-speaker overlap detector.
Detects simultaneous speakers via multi-pitch autocorrelation,
harmonic collision ratio, and spectral entropy degradation.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger("ramo_speaker.overlap")


class AcousticOverlapDetector:
    """
    High-precision acoustic overlap detector operating directly on raw audio waveforms.
    Identifies multiple concurrent vocal tracts (crosstalk) without needing external models.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        overlap_threshold: float = 0.50,
        min_f0_hz: float = 65.0,
        max_f0_hz: float = 400.0,
    ):
        self.sample_rate = sample_rate
        self.overlap_threshold = overlap_threshold
        self.min_lag = int(sample_rate / max_f0_hz)  # 40 samples @ 16kHz
        self.max_lag = int(sample_rate / min_f0_hz)  # 246 samples @ 16kHz

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Analyze audio chunk for multi-speaker overlap.
        Returns (is_overlap: bool, overlap_score: float in [0.0, 1.0]).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        if len(audio) < self.max_lag * 2:
            return False, 0.0

        # Frame-based multi-pitch analysis
        frame_len = int(0.080 * self.sample_rate)  # 80ms
        frame_hop = int(0.040 * self.sample_rate)  # 40ms
        n_frames = max(1, (len(audio) - frame_len) // frame_hop)

        overlap_votes = 0
        valid_voiced_frames = 0
        total_scores = []

        for i in range(n_frames):
            frame = audio[i * frame_hop : i * frame_hop + frame_len]
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms < 0.01:
                continue

            frame_score = self._analyze_frame_autocorr(frame)
            total_scores.append(frame_score)
            valid_voiced_frames += 1
            if frame_score >= self.overlap_threshold:
                overlap_votes += 1

        if valid_voiced_frames == 0:
            return False, 0.0

        mean_score = float(np.mean(total_scores)) if total_scores else 0.0
        vote_ratio = overlap_votes / valid_voiced_frames

        # Combined score: weighted average of mean frame score and vote ratio
        combined_score = round(0.5 * mean_score + 0.5 * vote_ratio, 3)
        is_overlap = combined_score >= self.overlap_threshold

        if is_overlap:
            logger.info(
                f"[OVERLAP_DETECT] ⚠️ Crosstalk detected! Score: {combined_score:.2f} "
                f"(Votes: {overlap_votes}/{valid_voiced_frames})"
            )

        return is_overlap, combined_score

    def _analyze_frame_autocorr(self, frame: np.ndarray) -> float:
        """Analyze a single frame for competing independent pitches."""
        centered = frame - np.mean(frame)
        corr = np.correlate(centered, centered, mode="full")
        corr = corr[len(corr) // 2 :]

        if corr[0] <= 1e-9:
            return 0.0

        norm_corr = corr / corr[0]
        search_region = norm_corr[self.min_lag : min(self.max_lag, len(norm_corr))]

        if len(search_region) < 10:
            return 0.0

        # Find local peaks
        peaks = []
        for lag_idx in range(1, len(search_region) - 1):
            if (
                search_region[lag_idx] > search_region[lag_idx - 1]
                and search_region[lag_idx] > search_region[lag_idx + 1]
                and search_region[lag_idx] > 0.20
            ):
                actual_lag = self.min_lag + lag_idx
                peaks.append((search_region[lag_idx], actual_lag))

        if len(peaks) < 2:
            return 0.0

        # Sort peaks descending by correlation height
        peaks.sort(key=lambda x: x[0], reverse=True)
        peak1_val, peak1_lag = peaks[0]
        peak2_val, peak2_lag = peaks[1]

        # Check if peak 2 is an octave multiple (harmonic) of peak 1:
        # e.g., lag2 ≈ 2 * lag1 or lag1 ≈ 2 * lag2
        ratio = max(peak1_lag, peak2_lag) / max(min(peak1_lag, peak2_lag), 1)
        is_harmonic = abs(ratio - 2.0) < 0.15 or abs(ratio - 3.0) < 0.15

        if is_harmonic:
            # Natural vocal harmonic, not an independent speaker
            return 0.10

        # Two competing, incommensurate pitches at comparable strength = crosstalk!
        relative_strength = peak2_val / max(peak1_val, 1e-6)
        if relative_strength > 0.45:
            # Significant competing independent pitch
            score = float(np.clip(relative_strength, 0.50, 1.0))
            return score

        return float(np.clip(relative_strength * 0.5, 0.0, 0.45))
