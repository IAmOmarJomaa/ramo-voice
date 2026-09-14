"""
ramo_clean.hpf
==============
4th-order Butterworth High-Pass Filter (HPF) at 80Hz.
Eliminates HVAC rumble, desk vibrations, microphone pops, and sub-bass noise
without attenuating vocal fundamental frequencies (100Hz - 300Hz).
"""

from typing import Optional
import numpy as np
from scipy import signal


class HPFFilter:
    """
    Butterworth High-Pass Filter designed to strip sub-80Hz acoustic rumble.
    """

    def __init__(
        self,
        cutoff_hz: float = 80.0,
        sample_rate: int = 16000,
        order: int = 4,
    ):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.order = order

        nyquist = 0.5 * sample_rate
        normalized_cutoff = cutoff_hz / nyquist
        # Design Butterworth high-pass filter
        self.b, self.a = signal.butter(order, normalized_cutoff, btype="highpass")
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal filter state."""
        self._zi = None

    def filter(self, audio: np.ndarray, stream: bool = False) -> np.ndarray:
        """
        Apply HPF to 1D float32 audio array.
        If stream=True, maintains filter initial conditions across sequential chunks.
        """
        if len(audio) == 0:
            return audio.astype(np.float32)

        data = audio.astype(np.float32)
        if stream:
            if self._zi is None:
                self._zi = signal.lfilter_zi(self.b, self.a) * data[0]
            filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        else:
            filtered = signal.lfilter(self.b, self.a, data)

        return filtered.astype(np.float32)
