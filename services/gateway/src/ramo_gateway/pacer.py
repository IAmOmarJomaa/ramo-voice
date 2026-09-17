"""
ramo_gateway.pacer
==================
Dynamic Latency Pacer: Waveform Similarity Overlap-Add (WSOLA) time-scale modification.
Prevents translation queue lag from bilingual syllable expansion (e.g. EN -> FR/ES +25%)
while strictly preserving natural vocal pitch.
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("ramo_gateway.pacer")


class DynamicLatencyPacer:
    """
    WSOLA time-scale modification engine and latency regulator.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        target_latency_ms: int = 1200,
        speed_min: float = 1.0,
        speed_max: float = 1.25,
        queue_threshold_words: int = 20,
    ):
        self.sample_rate = sample_rate
        self.target_latency_ms = target_latency_ms
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.queue_threshold_words = queue_threshold_words

    def estimate_pitch(self, audio: np.ndarray) -> float:
        """Estimate fundamental frequency (F0) using autocorrelation."""
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) < 512:
            return 0.0

        centered = audio - np.mean(audio)
        corr = np.correlate(centered, centered, mode="full")
        corr = corr[len(corr) // 2 :]

        min_lag = int(self.sample_rate / 400.0)
        max_lag = int(self.sample_rate / 60.0)

        if max_lag >= len(corr):
            return 0.0

        peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[peak_lag] > 0.15 * corr[0]:
            return float(self.sample_rate / peak_lag)
        return 0.0

    def calculate_speed(self, current_latency_ms: float, queue_words: int = 0) -> float:
        """
        Calculate the optimal WSOLA playback speed multiplier.
        Ramps smoothly between speed_min and speed_max when queue lags.
        """
        if current_latency_ms <= self.target_latency_ms and queue_words < self.queue_threshold_words:
            return self.speed_min

        # Scale based on latency overshoot and queue size
        overshoot = max(0.0, current_latency_ms - self.target_latency_ms) / 1000.0
        queue_factor = max(0.0, queue_words - self.queue_threshold_words) / 20.0
        pressure = min(1.0, 0.6 * overshoot + 0.4 * queue_factor)

        target_speed = self.speed_min + pressure * (self.speed_max - self.speed_min)
        return round(float(np.clip(target_speed, self.speed_min, self.speed_max)), 2)

    def apply_wsola(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Apply Synchronous Overlap-Add (SOLA) time-scale modification.
        Stretches or compresses audio duration by 1 / speed without altering pitch.
        """
        if abs(speed - 1.0) < 0.02 or len(audio) < int(0.1 * self.sample_rate):
            return audio.copy()

        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        win_len = int(0.030 * self.sample_rate)  # 30ms window
        hop_out = int(0.010 * self.sample_rate)  # 10ms synthesis hop
        hop_in = int(round(hop_out * speed))     # scaled analysis hop
        search_range = int(0.015 * self.sample_rate)  # 15ms search range

        win = np.hanning(win_len).astype(np.float32)
        out_len = int(len(audio) / speed)
        out = np.zeros(out_len + win_len, dtype=np.float32)
        norm = np.zeros(out_len + win_len, dtype=np.float32)

        # Initialize first frame
        out[:win_len] = audio[:win_len] * win
        norm[:win_len] = win**2

        syn_pos = hop_out
        ana_pos = hop_in

        while (syn_pos + win_len) <= len(out) and (ana_pos + win_len + search_range) <= len(audio):
            template = out[syn_pos : syn_pos + win_len]
            s_min = max(-search_range, -ana_pos)
            s_max = min(search_range, len(audio) - ana_pos - win_len)

            best_delta = 0
            best_corr = -1e9

            for d in range(s_min, s_max, 2):
                cand = audio[ana_pos + d : ana_pos + d + win_len]
                c = float(np.dot(template, cand))
                if c > best_corr:
                    best_corr = c
                    best_delta = d

            chosen = audio[ana_pos + best_delta : ana_pos + best_delta + win_len] * win
            out[syn_pos : syn_pos + win_len] += chosen
            norm[syn_pos : syn_pos + win_len] += win**2

            syn_pos += hop_out
            ana_pos += hop_in

        mask = norm > 1e-4
        out[mask] /= norm[mask]
        trimmed = out[:out_len]
        logger.debug(
            f"[WSOLA] Sped {len(audio)/self.sample_rate:.2f}s -> {len(trimmed)/self.sample_rate:.2f}s (speed: {speed}x)"
        )
        return trimmed
