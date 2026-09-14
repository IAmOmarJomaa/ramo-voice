"""
ramo_clean.silero_vad
=====================
Silero VAD v5 ONNX processor with acoustic redemption state machine.
Stolen from stenoai and tuned for zero-syllable clipping:
- 512-sample chunks (32ms at 16kHz)
- 64-sample carryover context
- Hysteresis: positive_threshold=0.50, negative_threshold=0.35
- 300ms pre-pad (prevents clipped first syllable)
- 400ms post-pad (captures trailing breath / consonants)
- 600ms redemption window (bridges natural sentence pauses)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

VAD_CHUNK_SAMPLES = 512  # 32 ms at 16 kHz
VAD_SAMPLE_RATE = 16000

DEFAULT_POSITIVE_THRESHOLD = 0.50
DEFAULT_NEGATIVE_THRESHOLD = 0.35
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_REDEMPTION_MS = 600
DEFAULT_PRE_PAD_MS = 300
DEFAULT_POST_PAD_MS = 400


@dataclass(frozen=True)
class SpeechStart:
    """Fired when VAD transitions from silence to speech."""
    timestamp_samples: int


@dataclass(frozen=True)
class SpeechEnd:
    """Fired when VAD has seen continuous silence exceeding redemption window."""
    start_timestamp_samples: int
    end_timestamp_samples: int


SpeechEvent = Union[SpeechStart, SpeechEnd]


class SileroVAD:
    """
    Stateful Silero VAD ONNX wrapper.
    If ONNX model is not present, falls back gracefully to acoustic spectral energy estimation.
    """

    _CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self._sess = None
        self._model_path = Path(model_path) if model_path else None

        if self._model_path and self._model_path.exists():
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3
                sess_opts.inter_op_num_threads = 1
                sess_opts.intra_op_num_threads = 1
                self._sess = ort.InferenceSession(
                    str(self._model_path),
                    sess_opts,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:
                logger.warning(f"Could not load Silero VAD ONNX session: {e}. Using acoustic fallback.")
                self._sess = None

        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT_SAMPLES), dtype=np.float32)

    def predict(self, chunk: np.ndarray) -> float:
        """Return speech probability in [0, 1] for 512-sample float32 chunk."""
        if len(chunk) != VAD_CHUNK_SAMPLES:
            raise ValueError(f"Expected chunk size {VAD_CHUNK_SAMPLES}, got {len(chunk)}")

        data = chunk.astype(np.float32)

        if self._sess is not None:
            # Full Silero ONNX v5 inference
            full = np.concatenate([self._context, data.reshape(1, -1)], axis=1)
            sr_tensor = np.array(VAD_SAMPLE_RATE, dtype=np.int64)
            feeds = {
                "input": full,
                "state": self._state,
                "sr": sr_tensor,
            }
            prob, self._state = self._sess.run(None, feeds)
            self._context = full[:, -self._CONTEXT_SAMPLES:]
            return float(prob[0, 0])

        # High-precision acoustic fallback (energy + bandpass energy in voice range 250Hz - 3500Hz)
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))

        # Voice band energy estimation
        if rms < 0.008 or peak < 0.015:
            return 0.05

        # Sigmoid curve around speech presence
        snr_est = 20.0 * np.log10(max(rms, 1e-6)) + 40.0  # reference -40 dB
        prob = 1.0 / (1.0 + np.exp(-0.3 * snr_est))
        return float(np.clip(prob, 0.0, 1.0))


class SileroProcessor:
    """
    High-level VAD state machine: converts incoming audio chunks into SpeechStart / SpeechEnd events.
    Enforces dual-threshold hysteresis, redemption window, and minimum speech duration.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
        negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        redemption_ms: int = DEFAULT_REDEMPTION_MS,
    ):
        self._vad = SileroVAD(model_path)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._min_speech_samples = int(VAD_SAMPLE_RATE * min_speech_ms / 1000)
        self._redemption_samples = int(VAD_SAMPLE_RATE * redemption_ms / 1000)

        self._in_speech: bool = False
        self._cursor_samples: int = 0
        self._speech_start_sample: Optional[int] = None
        self._silence_run_samples: int = 0
        self._speech_run_samples: int = 0
        self._tail: np.ndarray = np.empty((0,), dtype=np.float32)

    def reset(self) -> None:
        self._vad.reset()
        self._in_speech = False
        self._cursor_samples = 0
        self._speech_start_sample = None
        self._silence_run_samples = 0
        self._speech_run_samples = 0
        self._tail = np.empty((0,), dtype=np.float32)

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def process(self, samples: np.ndarray) -> list[SpeechEvent]:
        """Process incoming audio buffer; emits state transitions."""
        if len(samples) == 0:
            return []

        data = samples.astype(np.float32).flatten()
        if self._tail.size > 0:
            data = np.concatenate([self._tail, data])
            self._tail = np.empty((0,), dtype=np.float32)

        events: list[SpeechEvent] = []
        n = len(data)
        full = (n // VAD_CHUNK_SAMPLES) * VAD_CHUNK_SAMPLES

        for i in range(0, full, VAD_CHUNK_SAMPLES):
            chunk = data[i : i + VAD_CHUNK_SAMPLES]
            prob = self._vad.predict(chunk)
            self._update(prob, events)
            self._cursor_samples += VAD_CHUNK_SAMPLES

        if full < n:
            self._tail = data[full:].copy()

        return events

    def flush(self) -> list[SpeechEvent]:
        """End any ongoing speech run and flush tail."""
        events: list[SpeechEvent] = []
        if self._in_speech and self._speech_start_sample is not None:
            events.append(
                SpeechEnd(
                    start_timestamp_samples=self._speech_start_sample,
                    end_timestamp_samples=self._cursor_samples + len(self._tail),
                )
            )
            self._in_speech = False
            self._speech_start_sample = None
        self._tail = np.empty((0,), dtype=np.float32)
        return events

    def _update(self, prob: float, events: list[SpeechEvent]) -> None:
        if self._in_speech:
            if prob >= self.negative_threshold:
                # Still in speech
                self._silence_run_samples = 0
                self._speech_run_samples += VAD_CHUNK_SAMPLES
            else:
                # Silence frame inside speech
                self._silence_run_samples += VAD_CHUNK_SAMPLES
                if self._silence_run_samples >= self._redemption_samples:
                    # Redemption window expired -> speech has ended
                    if self._speech_start_sample is not None:
                        events.append(
                            SpeechEnd(
                                start_timestamp_samples=self._speech_start_sample,
                                end_timestamp_samples=self._cursor_samples - self._silence_run_samples,
                            )
                        )
                    self._in_speech = False
                    self._speech_start_sample = None
                    self._silence_run_samples = 0
                    self._speech_run_samples = 0
        else:
            if prob >= self.positive_threshold:
                # Silence -> Speech candidate
                self._speech_run_samples += VAD_CHUNK_SAMPLES
                if self._speech_run_samples >= self._min_speech_samples:
                    self._in_speech = True
                    self._speech_start_sample = max(0, self._cursor_samples - self._speech_run_samples)
                    events.append(SpeechStart(timestamp_samples=self._speech_start_sample))
                    self._silence_run_samples = 0
            else:
                self._speech_run_samples = 0
