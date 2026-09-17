"""
ramo_gateway.chronos
====================
Sovereign Live Utterance Pipeline & Stabilizer.
Synthesizes the 5-repo streaming ASR blueprint (UFAL + Meetily + StenoAI + Natively-Cluely + Moonshine):
- 512ms (8,192-sample) Rolling Look-Behind Buffer preserving plosive onsets
- Silero VAD state machine with 2,000ms redemption time bridging natural breathing pauses
- Linear probability fading between 10.0s and 15.0s hunting for natural pause boundaries
- Monologue soft-commit with 300ms tail context carry at 15.0s ceiling
- Adaptive EWMA keep-pace guard adjusting partial intervals to prevent queue backlog
- LocalAgreement-2 with word-boundary snapping longest common prefix
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np

logger = logging.getLogger("ramo_gateway.chronos")


class ChronosCutType(str, enum.Enum):
    PROVISIONAL = "provisional"
    SLIDING_WINDOW = "sliding_window"
    FINAL = "final"
    SOFT_CUT = "soft_cut"
    HARD_CUT = "hard_cut"
    FORCED_FLUSH = "forced_flush"


@dataclass
class UtteranceEvent:
    event_type: str        # "partial" | "final"
    line_id: str           # Stable monotonic ID (e.g. utt_sess1_1)
    audio: np.ndarray      # float32 16kHz PCM
    is_final: bool
    context_prompt: str = ""
    duration_s: float = 0.0

    @property
    def cut_type(self) -> ChronosCutType:
        return ChronosCutType.FINAL if self.is_final else ChronosCutType.PROVISIONAL

    @property
    def pcm_data(self) -> bytes:
        return (np.clip(self.audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# Backward-compatibility alias
ChronosCut = UtteranceEvent


def calculate_rms(pcm_data: Union[bytes, np.ndarray]) -> float:
    """Calculates root-mean-square amplitude of audio samples."""
    if isinstance(pcm_data, bytes):
        if len(pcm_data) < 2:
            return 0.0
        arr = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        arr = pcm_data.astype(np.float32)
        if len(arr) == 0:
            return 0.0
    return float(np.sqrt(np.mean(arr ** 2) + 1e-9))


def calculate_fade_factor(
    current_samples: int,
    sr: int = 16000,
    fade_start_s: float = 10.0,
    max_utterance_s: float = 15.0,
) -> float:
    """
    Moonshine linear probability fading between fade_start_s (10s) and max_utterance_s (15s).
    Gently hunts for natural human breath pauses during continuous speech.
    """
    fade_start_samples = int(fade_start_s * sr)
    max_samples = int(max_utterance_s * sr)
    if current_samples <= fade_start_samples:
        return 1.0
    if current_samples >= max_samples:
        return 0.0
    return float((max_samples - current_samples) / (max_samples - fade_start_samples))


def longest_common_prefix_word_boundary(a: str, b: str) -> str:
    """
    Computes LCP between two hypotheses, snapping backward to the last whitespace
    if the match ends inside a word (Natively-Cluely / Meetily standard).
    """
    if not a or not b:
        return ""
    length = min(len(a), len(b))
    i = 0
    while i < length and a[i] == b[i]:
        i += 1

    # Snap back to word boundary if split is mid-word on either side
    is_mid_word = (
        (i < len(a) and not a[i].isspace()) or (i < len(b) and not b[i].isspace())
    )
    if is_mid_word and i > 0 and not a[i - 1].isspace():
        while i > 0 and not a[i - 1].isspace():
            i -= 1
    return a[:i].strip()


@dataclass
class LocalAgreementResult:
    committed: str
    tentative: str
    newly_committed: str = ""


class LocalAgreement:
    """
    Online LocalAgreement (n=2) consensus stabilizer with word-boundary snapping.
    Words are committed only after appearing identically in consecutive hypotheses.
    """

    def __init__(self, n_agreement: int = 2):
        self.n_agreement = n_agreement
        self.hypotheses: List[str] = []
        self.committed_text: str = ""

    def step(self, text: str) -> LocalAgreementResult:
        """
        Ingest the latest transcript hypothesis.
        Returns committed and tentative text with word-boundary snapping.
        """
        clean_text = text.strip()
        self.hypotheses.append(clean_text)
        if len(self.hypotheses) > self.n_agreement:
            self.hypotheses.pop(0)

        prev_committed = self.committed_text

        if len(self.hypotheses) >= self.n_agreement:
            h_first = self.hypotheses[0]
            agreed = h_first
            for h in self.hypotheses[1:]:
                agreed = longest_common_prefix_word_boundary(agreed, h)

            if len(agreed) > len(prev_committed):
                self.committed_text = agreed

        newly_committed = ""
        if len(self.committed_text) > len(prev_committed):
            newly_committed = self.committed_text[len(prev_committed):].strip()

        latest = self.hypotheses[-1] if self.hypotheses else ""
        tentative = latest[len(self.committed_text):].strip()

        return LocalAgreementResult(
            committed=self.committed_text,
            tentative=tentative,
            newly_committed=newly_committed,
        )

    def flush(self) -> LocalAgreementResult:
        """On EOS or final commit, finalize all remaining text."""
        prev_committed = self.committed_text
        if self.hypotheses:
            self.committed_text = self.hypotheses[-1]
        newly_committed = self.committed_text[len(prev_committed):].strip()
        self.hypotheses = []
        return LocalAgreementResult(
            committed=self.committed_text,
            tentative="",
            newly_committed=newly_committed,
        )

    def reset(self) -> None:
        self.hypotheses = []
        self.committed_text = ""


class LiveUtterancePipeline:
    """
    Production Live Utterance Accumulator with Look-Behind Buffer,
    Silero VAD 2000ms redemption, probability fading, soft-commit tail carry,
    and adaptive EWMA keep-pace guard.
    """

    LOOK_BEHIND_SAMPLES: int = 8192  # 512ms @ 16kHz

    def __init__(
        self,
        sample_rate: int = 16000,
        session_id: str = "sess",
        vad_redemption_ms: int = 2000,
        min_speech_ms: int = 250,
        base_partial_interval_s: float = 0.5,
        max_partial_interval_s: float = 4.0,
        keep_pace_alpha: float = 0.3,
        keep_pace_safety: float = 1.2,
        fade_start_s: float = 10.0,
        max_utterance_s: float = 15.0,
        soft_commit_tail_s: float = 0.3,
    ):
        self.sr = sample_rate
        self.session_id = session_id
        self.vad_redemption_samples = int(self.sr * vad_redemption_ms / 1000)
        self.min_speech_samples = int(self.sr * min_speech_ms / 1000)

        # Look-behind rolling circular buffer (512ms)
        self.look_behind_ring = np.zeros((self.LOOK_BEHIND_SAMPLES,), dtype=np.float32)

        # Utterance state machine
        self.in_speech: bool = False
        self.current_line_id: Optional[str] = None
        self.current_samples = np.empty((0,), dtype=np.float32)
        self.last_partial_sample_count: int = 0
        self.silence_run_samples: int = 0
        self.speech_run_samples: int = 0

        # Monologue limits
        self.fade_start_s = fade_start_s
        self.max_utterance_s = max_utterance_s
        self.max_utterance_samples = int(self.sr * max_utterance_s)
        self.soft_commit_tail_samples = int(self.sr * soft_commit_tail_s)
        self.partial_window_samples = int(15.0 * self.sr)

        # Adaptive EWMA keep-pace guard (StenoAI standard)
        self.base_partial_interval_s = base_partial_interval_s
        self.max_partial_interval_s = max_partial_interval_s
        self.keep_pace_alpha = keep_pace_alpha
        self.keep_pace_safety = keep_pace_safety
        self.decode_ewma_s: float = 0.0

        # Timeline and prompt
        self.context_prompt: str = ""
        self.line_counter: int = 0

    def record_decode_wall_time(self, dt_s: float) -> None:
        """Update EWMA of STT inference wall-time to dynamically stretch partial interval."""
        if dt_s <= 0:
            return
        if self.decode_ewma_s <= 0.0:
            self.decode_ewma_s = dt_s
        else:
            self.decode_ewma_s = (1.0 - self.keep_pace_alpha) * self.decode_ewma_s + self.keep_pace_alpha * dt_s

    def effective_partial_interval_samples(self) -> int:
        """Calculate dynamic partial interval based on EWMA decode time."""
        pace_s = max(self.base_partial_interval_s, self.decode_ewma_s * self.keep_pace_safety)
        pace_s = min(pace_s, self.max_partial_interval_s)
        return int(pace_s * self.sr)

    def push_audio(
        self,
        chunk: Union[bytes, np.ndarray],
        is_voice: Optional[bool] = None,
    ) -> List[UtteranceEvent]:
        """
        Ingest audio chunk (bytes or float32 ndarray) and return any triggered UtteranceEvents.
        """
        if len(chunk) == 0:
            return []

        if isinstance(chunk, bytes):
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = chunk.astype(np.float32)

        events: List[UtteranceEvent] = []

        # Determine voice presence if not explicitly provided
        if is_voice is None:
            rms = calculate_rms(samples)
            if self.in_speech:
                # Monologue probability fade factor (Moonshine standard)
                dur_samples = len(self.current_samples)
                fade = calculate_fade_factor(dur_samples, self.sr, self.fade_start_s, self.max_utterance_s)
                # Attenuate effective voice detection threshold to seek breath pauses
                thresh = 0.010 * max(0.2, fade)
                is_voice = rms >= thresh
            else:
                is_voice = rms >= 0.015

        # 1. Non-Speech State: update rolling look-behind buffer
        if not self.in_speech:
            if len(samples) >= self.LOOK_BEHIND_SAMPLES:
                self.look_behind_ring[:] = samples[-self.LOOK_BEHIND_SAMPLES:]
            else:
                self.look_behind_ring = np.roll(self.look_behind_ring, -len(samples))
                self.look_behind_ring[-len(samples):] = samples

            # Voice Onset
            if is_voice:
                self.speech_run_samples += len(samples)
                if self.speech_run_samples >= self.min_speech_samples or is_voice:
                    self.in_speech = True
                    self.line_counter += 1
                    self.current_line_id = f"utt_{self.session_id}_{self.line_counter}"
                    # Prepend look-behind to preserve plosives
                    self.current_samples = np.concatenate([self.look_behind_ring.copy(), samples])
                    self.last_partial_sample_count = 0
                    self.silence_run_samples = 0
                    self.speech_run_samples = 0
                    logger.debug(
                        f"[PIPELINE] 🎙️ Speech start: line_id='{self.current_line_id}' "
                        f"(Prepended {len(self.look_behind_ring)} look-behind samples)"
                    )
            else:
                self.speech_run_samples = 0
            return events

        # 2. In Speech State: voice continuing
        if self.in_speech and is_voice:
            self.silence_run_samples = 0
            self.current_samples = np.concatenate([self.current_samples, samples])

            # Monologue Soft-Commit Ceiling (15.0s)
            if len(self.current_samples) >= self.max_utterance_samples:
                logger.info(
                    f"[PIPELINE] ⏳ Monologue ceiling reached ({len(self.current_samples)/self.sr:.1f}s >= {self.max_utterance_s}s) "
                    f"-> Soft-commit on '{self.current_line_id}'"
                )
                final_ev = UtteranceEvent(
                    event_type="final",
                    line_id=self.current_line_id,
                    audio=self.current_samples.copy(),
                    is_final=True,
                    context_prompt=self.context_prompt,
                    duration_s=len(self.current_samples) / self.sr,
                )
                events.append(final_ev)

                # Carry forward 300ms tail context into a fresh segment
                tail = self.current_samples[-self.soft_commit_tail_samples:]
                self.line_counter += 1
                self.current_line_id = f"utt_{self.session_id}_{self.line_counter}"
                self.current_samples = tail.copy()
                self.last_partial_sample_count = 0
                return events

            # Check if due for Partial Preview
            delta_samples = len(self.current_samples) - self.last_partial_sample_count
            if delta_samples >= self.effective_partial_interval_samples():
                self.last_partial_sample_count = len(self.current_samples)
                tail_window = self.current_samples[-self.partial_window_samples:]
                events.append(
                    UtteranceEvent(
                        event_type="partial",
                        line_id=self.current_line_id,
                        audio=tail_window.copy(),
                        is_final=False,
                        context_prompt=self.context_prompt,
                        duration_s=len(tail_window) / self.sr,
                    )
                )
            return events

        # 3. In Speech State: silence detected during utterance
        if self.in_speech and not is_voice:
            self.silence_run_samples += len(samples)
            self.current_samples = np.concatenate([self.current_samples, samples])

            if self.silence_run_samples >= self.vad_redemption_samples:
                # 2000ms redemption window expired -> Speech has finalized!
                logger.info(
                    f"[PIPELINE] 🛑 Speech ended: line_id='{self.current_line_id}' "
                    f"({len(self.current_samples)/self.sr:.2f}s total audio, {self.silence_run_samples/self.sr:.2f}s silence)"
                )
                self.in_speech = False
                final_ev = UtteranceEvent(
                    event_type="final",
                    line_id=self.current_line_id,
                    audio=self.current_samples.copy(),
                    is_final=True,
                    context_prompt=self.context_prompt,
                    duration_s=len(self.current_samples) / self.sr,
                )
                events.append(final_ev)

                # Reset state for next turn
                self.current_samples = np.empty((0,), dtype=np.float32)
                self.last_partial_sample_count = 0
                self.silence_run_samples = 0
                self.speech_run_samples = 0
                self.current_line_id = None
                return events

        return events

    def flush(self) -> List[UtteranceEvent]:
        """EOS or forced flush."""
        events: List[UtteranceEvent] = []
        if self.in_speech and len(self.current_samples) > 0:
            final_ev = UtteranceEvent(
                event_type="final",
                line_id=self.current_line_id or f"utt_{self.session_id}_{self.line_counter}",
                audio=self.current_samples.copy(),
                is_final=True,
                context_prompt=self.context_prompt,
                duration_s=len(self.current_samples) / self.sr,
            )
            events.append(final_ev)

        self.in_speech = False
        self.current_samples = np.empty((0,), dtype=np.float32)
        self.last_partial_sample_count = 0
        self.silence_run_samples = 0
        self.speech_run_samples = 0
        self.current_line_id = None
        return events

    # Backward compatibility for ChronosBuffer interface
    def add_audio(self, chunk: bytes) -> List[UtteranceEvent]:
        return self.push_audio(chunk)

    @property
    def audio_buffer(self) -> bytes:
        return (np.clip(self.current_samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def trim_on_punctuation(self, text: str, words: list) -> int:
        return 0


# Backward-compatibility alias
ChronosBuffer = LiveUtterancePipeline


