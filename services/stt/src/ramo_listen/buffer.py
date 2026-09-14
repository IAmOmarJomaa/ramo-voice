"""
ramo_listen.buffer
==================
Rolling audio ring buffer with zero-latency resampling to 16kHz.
Ensures uniform acoustic input for whisper_streaming and SenseVoice engines.
"""

from typing import Optional
import numpy as np
from scipy import signal


try:
    from ramo_clean.pipeline import AudioPreconditioner
    RAMO_CLEAN_AVAILABLE = True
except ImportError:
    RAMO_CLEAN_AVAILABLE = False


class AudioRingBuffer:
    """
    Circular/rolling in-memory buffer storing float32 PCM samples at target_sr (default 16000).
    Automatically resamples incoming audio if sample rates differ.
    Optionally applies ramo_clean 5-stage preconditioning.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        max_duration_sec: float = 30.0,
        enable_preconditioner: bool = False,
    ):
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * target_sr)
        self._buffer = np.zeros(0, dtype=np.float32)
        self.preconditioner = (
            AudioPreconditioner(sample_rate=target_sr)
            if enable_preconditioner and RAMO_CLEAN_AVAILABLE
            else None
        )

    @property
    def duration_sec(self) -> float:
        return len(self._buffer) / self.target_sr

    def push(self, samples: np.ndarray, input_sr: Optional[int] = None) -> None:
        """Add new samples to buffer, resampling if input_sr != target_sr, and preconditioning."""
        if input_sr is None:
            input_sr = self.target_sr

        audio = samples.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Resample to target_sr if necessary
        if input_sr != self.target_sr and len(audio) > 0:
            target_num_samples = int(round(len(audio) * float(self.target_sr) / float(input_sr)))
            if target_num_samples > 0:
                audio = signal.resample(audio, target_num_samples).astype(np.float32)

        # Apply in-memory preconditioning if enabled
        if self.preconditioner is not None and len(audio) > 0:
            audio = self.preconditioner.process_chunk(audio, stream=True).audio

        # Append to buffer
        if len(self._buffer) == 0:
            self._buffer = audio
        else:
            self._buffer = np.concatenate([self._buffer, audio])

        # Clamp to max capacity
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples:]

    def get_window(self, duration_sec: float) -> np.ndarray:
        """Extract the most recent window of audio."""
        num_samples = int(round(duration_sec * self.target_sr))
        if num_samples >= len(self._buffer):
            return self._buffer.copy()
        return self._buffer[-num_samples:].copy()

    def get_all(self) -> np.ndarray:
        """Extract entire active buffer."""
        return self._buffer.copy()

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer = np.zeros(0, dtype=np.float32)
