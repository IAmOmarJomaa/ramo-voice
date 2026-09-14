"""
ramo_clean.spectral_gate
========================
Real-time frequency-domain spectral gating and noise floor suppressor.
Uses STFT spectral subtraction with dynamic noise floor tracking to attenuate
stationary background noise without speech distortion.
"""

from typing import Optional
import numpy as np
from scipy import signal


class SpectralGate:
    """
    Lightweight STFT spectral noise gate.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        noise_reduction_factor: float = 0.65,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_reduction_factor = noise_reduction_factor
        self._noise_floor: Optional[np.ndarray] = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral gating to audio array.
        """
        if len(audio) < self.n_fft:
            return audio.astype(np.float32)

        data = audio.astype(np.float32)

        # Compute STFT
        _, _, Zxx = signal.stft(
            data,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            padded=False,
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Track noise floor estimate from bottom 15th percentile of frame magnitudes
        frame_floor = np.percentile(magnitude, 15, axis=-1, keepdims=True)
        if self._noise_floor is None:
            self._noise_floor = frame_floor
        else:
            # Smoothly adapt noise floor
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * frame_floor

        # Spectral subtraction
        gain = 1.0 - (self.noise_reduction_factor * self._noise_floor / (magnitude + 1e-8))
        gain = np.clip(gain, 0.1, 1.0)

        # Reconstruct filtered STFT
        cleaned_Zxx = gain * magnitude * np.exp(1j * phase)

        # Inverse STFT
        _, cleaned = signal.istft(
            cleaned_Zxx,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )

        # Match output length with input length
        if len(cleaned) < len(data):
            cleaned = np.pad(cleaned, (0, len(data) - len(cleaned)), mode="constant")
        else:
            cleaned = cleaned[:len(data)]

        return cleaned.astype(np.float32)
