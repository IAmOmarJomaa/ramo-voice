"""
ramo_clean.agc
==============
Adaptive Gain Control (AGC) & Soft-Knee Limiter.
Normalizes incoming speech chunks dynamically towards -20 dBFS
while applying soft-saturation limiting to prevent digital clipping (> 1.0).
"""

import numpy as np


class AGCLeveler:
    """
    Adaptive Gain Control normalizer with soft-knee limiter.
    """

    def __init__(
        self,
        target_dbfs: float = -20.0,
        max_gain_db: float = 18.0,
        silence_threshold_dbfs: float = -55.0,
    ):
        self.target_dbfs = target_dbfs
        self.target_rms = 10.0 ** (target_dbfs / 20.0)
        self.max_gain_linear = 10.0 ** (max_gain_db / 20.0)
        self.silence_threshold_rms = 10.0 ** (silence_threshold_dbfs / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Adjust gain of audio array to match target RMS and softly limit peaks.
        """
        if len(audio) == 0:
            return audio.astype(np.float32)

        data = audio.astype(np.float32)
        current_rms = np.sqrt(np.mean(data**2))

        # Do not boost pure silence/ambient room noise below floor
        if current_rms < self.silence_threshold_rms or current_rms == 0:
            return data

        # Desired gain factor
        gain = self.target_rms / current_rms
        gain = min(gain, self.max_gain_linear)

        scaled = data * gain

        # Soft-knee limiting using hyperbolic tangent when peaks exceed 0.90
        # For |x| <= 0.85, linear passthrough. Above 0.85, smooth compression.
        threshold = 0.85
        over = np.abs(scaled) > threshold
        if np.any(over):
            signs = np.sign(scaled)
            excess = np.abs(scaled) - threshold
            # Smoothly compress excess using tanh
            compressed = threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
            scaled = np.where(over, signs * compressed, scaled)

        return np.clip(scaled, -1.0, 1.0).astype(np.float32)
