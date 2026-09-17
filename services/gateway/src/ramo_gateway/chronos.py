"""
ramo_gateway.chronos
====================
Chronos Dynamic Streaming Buffer with Sliding Window & LocalAgreement (n=2).
Combines:
- 10.0s Rolling Sliding Window (WINDOW_MAX_SECS = 10, WINDOW_MAX_BYTES = 320,000)
- 1.0s Emit Pace (EMIT_EVERY_BYTES = 32,000)
- Online LocalAgreement (n=2) consensus stabilizer
- Sentence boundary buffer trimming on [.!?]
- Elimination of premature 7.0s monologue hard ceilings & micro-pause splits
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


class ChronosCutType(str, enum.Enum):
    PROVISIONAL = "provisional"
    SLIDING_WINDOW = "sliding_window"
    SOFT_CUT = "soft_cut"
    HARD_CUT = "hard_cut"
    FORCED_FLUSH = "forced_flush"


@dataclass
class ChronosCut:
    cut_type: ChronosCutType
    pcm_data: bytes
    is_final: bool
    overlap_secs: float = 0.0


@dataclass
class LocalAgreementResult:
    committed: str
    tentative: str
    newly_committed: str = ""


class LocalAgreement:
    """
    Online LocalAgreement (n=2) consensus stabilizer.
    Words are committed only after appearing identically in consecutive hypotheses.
    """

    def __init__(self, n_agreement: int = 2):
        self.n_agreement = n_agreement
        self.hypotheses: List[List[str]] = []
        self.committed_words: List[str] = []

    def step(self, text: str) -> LocalAgreementResult:
        """
        Ingest the transcript hypothesis of the current sliding window.
        Returns committed and tentative text.
        """
        words = [w for w in text.strip().split() if w]
        self.hypotheses.append(words)
        if len(self.hypotheses) > self.n_agreement:
            self.hypotheses.pop(0)

        prev_committed_len = len(self.committed_words)

        if len(self.hypotheses) >= self.n_agreement:
            # Find longest common prefix across all stored hypotheses
            h_first = self.hypotheses[0]
            common_len = len(h_first)
            for h in self.hypotheses[1:]:
                cur_match = 0
                while cur_match < common_len and cur_match < len(h) and h_first[cur_match] == h[cur_match]:
                    cur_match += 1
                common_len = cur_match

            # Can only commit forward
            if common_len > prev_committed_len:
                self.committed_words = list(h_first[:common_len])

        committed_str = " ".join(self.committed_words)
        newly_committed_str = " ".join(self.committed_words[prev_committed_len:])

        latest_words = self.hypotheses[-1] if self.hypotheses else []
        tentative_words = latest_words[len(self.committed_words):]
        tentative_str = " ".join(tentative_words)

        return LocalAgreementResult(
            committed=committed_str,
            tentative=tentative_str,
            newly_committed=newly_committed_str,
        )

    def flush(self) -> LocalAgreementResult:
        """On EOS or forced flush, commit all remaining tentative words."""
        prev_committed_len = len(self.committed_words)
        if self.hypotheses:
            self.committed_words = list(self.hypotheses[-1])
        committed_str = " ".join(self.committed_words)
        newly_committed_str = " ".join(self.committed_words[prev_committed_len:])
        self.hypotheses = []
        return LocalAgreementResult(
            committed=committed_str,
            tentative="",
            newly_committed=newly_committed_str,
        )

    def reset(self) -> None:
        self.hypotheses = []
        self.committed_words = []


def calculate_rms(pcm_bytes: bytes) -> float:
    """Calculates root-mean-square amplitude of 16-bit PCM samples."""
    if len(pcm_bytes) < 2:
        return 0.0
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(arr ** 2) + 1e-9))


class ChronosBuffer:
    """
    Stateful audio accumulation buffer with 10.0s rolling sliding window and 1.0s emit pace.
    Eliminates arbitrary 120ms/300ms pause chopping and 7.0s monologue ceiling.
    """

    # Audio format: 16kHz, 16-bit Mono = 32,000 bytes/second
    SAMPLE_RATE: int = 16000
    BYTES_PER_SAMPLE: int = 2
    BYTES_PER_SEC: int = 32000

    # Operational constants
    WINDOW_MAX_SECS: int = 10
    WINDOW_MAX_BYTES: int = 32000 * 10      # 320,000 bytes (10.0s cap)
    EMIT_EVERY_BYTES: int = 32000          # Emit sliding window cut every 1.0s of new audio
    PROVISIONAL_TICK_BYTES: int = 19200    # 600ms provisional preview interval
    MIN_PROCESS_BYTES: int = 3200          # 100ms minimum audio before processing
    OVERLAP_SAFETY_BYTES: int = 6400       # 200ms safety overlap retained on punctuation trim
    SILENCE_RMS_CEILING: float = 0.015

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = bytearray()
        self.bytes_since_last_emit = 0
        self.bytes_since_provisional = 0
        self.silence_duration_bytes = 0

    def add_audio(self, chunk: bytes) -> List[ChronosCut]:
        """
        Ingest audio bytes and return cuts triggered by the 1.0s emit pace.
        Maintains the rolling history for STT and LocalAgreement.
        """
        cuts: List[ChronosCut] = []
        if not chunk:
            return cuts

        self.audio_buffer.extend(chunk)
        self.bytes_since_last_emit += len(chunk)
        self.bytes_since_provisional += len(chunk)

        # Monitor silence envelope for diagnostics / idle detection
        rms = calculate_rms(chunk)
        if rms < self.SILENCE_RMS_CEILING:
            self.silence_duration_bytes += len(chunk)
        else:
            self.silence_duration_bytes = 0

        # Emit 10.0s sliding window every 1.0s of new audio
        if self.bytes_since_last_emit >= self.EMIT_EVERY_BYTES:
            self.bytes_since_last_emit = 0
            window = bytes(self.audio_buffer[-self.WINDOW_MAX_BYTES:])
            cuts.append(
                ChronosCut(
                    cut_type=ChronosCutType.SLIDING_WINDOW,
                    pcm_data=window,
                    is_final=False,
                )
            )

        return cuts

    def trim_on_punctuation(self, text: str, words: List[Dict[str, Any]]) -> int:
        """
        Trim confirmed audio from the buffer when committed text ends with [.!?].
        Prevents unbounded buffer growth during extended monologues.
        Returns the number of bytes trimmed from the start of the buffer.
        """
        if not text or not words:
            return 0

        if not re.search(r'[.!?]$', text.strip()):
            return 0

        trim_time = words[-1].get("end")
        if trim_time is None or trim_time <= 0:
            return 0

        trim_bytes = int(trim_time * self.SAMPLE_RATE * self.BYTES_PER_SAMPLE)
        trim_bytes = max(0, trim_bytes - self.OVERLAP_SAFETY_BYTES)

        window_start_bytes = max(0, len(self.audio_buffer) - self.WINDOW_MAX_BYTES)
        trim_idx = window_start_bytes + trim_bytes
        trim_idx = max(0, min(trim_idx, len(self.audio_buffer)))

        if trim_idx > 0:
            self.audio_buffer = bytearray(self.audio_buffer[trim_idx:])
            self.bytes_since_last_emit = max(0, self.bytes_since_last_emit - trim_idx)
            self.bytes_since_provisional = max(0, self.bytes_since_provisional - trim_idx)
            return trim_idx

        return 0

    def should_trigger_provisional(self) -> bool:
        """Returns True if provisional tick interval (600ms) reached."""
        if (
            self.bytes_since_provisional >= self.PROVISIONAL_TICK_BYTES
            and len(self.audio_buffer) >= self.MIN_PROCESS_BYTES
        ):
            self.bytes_since_provisional = 0
            return True
        return False

    def get_provisional_snapshot(self) -> Optional[bytes]:
        """Snapshot current buffer for provisional transcription without flushing."""
        if len(self.audio_buffer) < self.MIN_PROCESS_BYTES:
            return None
        return bytes(self.audio_buffer[-self.WINDOW_MAX_BYTES:])

    def truncate_at_offset(self, byte_offset: int) -> None:
        """Cleanly truncate audio buffer at exact byte offset."""
        if 0 < byte_offset <= len(self.audio_buffer):
            self.audio_buffer = bytearray(self.audio_buffer[byte_offset:])
            self.silence_duration_bytes = 0
            self.bytes_since_last_emit = max(0, self.bytes_since_last_emit - byte_offset)
            self.bytes_since_provisional = max(0, self.bytes_since_provisional - byte_offset)

    def flush(self) -> Optional[ChronosCut]:
        """Forced flush on EOS or stream close."""
        if len(self.audio_buffer) >= self.MIN_PROCESS_BYTES:
            pcm = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            self.bytes_since_last_emit = 0
            self.bytes_since_provisional = 0
            self.silence_duration_bytes = 0
            return ChronosCut(cut_type=ChronosCutType.FORCED_FLUSH, pcm_data=pcm, is_final=True)
        self.audio_buffer = bytearray()
        self.bytes_since_last_emit = 0
        self.bytes_since_provisional = 0
        self.silence_duration_bytes = 0
        return None

