"""
ramo_gateway.chronos
====================
Chronos Dynamic Auto-Cut Buffer (PFE Section 3.5.3).
Handles:
- 4.0s hard cut (128,000 bytes at 16kHz mono)
- 0.5s overlap tail retention across hard cuts
- 500ms conversational silence detection (RMS < 0.015) soft cut
- Soft-cut overlap tail discard guard
- 500ms provisional tick for live streaming STT preview
"""

import enum
import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


class ChronosCutType(str, enum.Enum):
    PROVISIONAL = "provisional"
    SOFT_CUT = "soft_cut"
    HARD_CUT = "hard_cut"
    FORCED_FLUSH = "forced_flush"


@dataclass
class ChronosCut:
    cut_type: ChronosCutType
    pcm_data: bytes
    is_final: bool


def calculate_rms(pcm_bytes: bytes) -> float:
    """Calculates root-mean-square amplitude of 16-bit PCM samples."""
    if len(pcm_bytes) < 2:
        return 0.0
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(arr ** 2) + 1e-9))


class ChronosBuffer:
    """
    Stateful audio accumulation buffer with intelligent dynamic segmentation.
    """

    MAX_CONTINUOUS_BYTES: int = 128000     # 4.0 seconds (64,000 samples @ 16kHz mono)
    OVERLAP_TAIL_BYTES: int = 16000        # 0.5 seconds (8,000 samples)
    PROVISIONAL_TICK_BYTES: int = 16000    # 0.5 seconds interval
    SILENCE_THRESHOLD_BYTES: int = 16000   # 500ms silence duration threshold
    SILENCE_RMS_CEILING: float = 0.015     # RMS below this is considered conversational pause
    MIN_PROCESS_BYTES: int = 3200          # 100ms minimum audio before processing

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_buffer = bytearray()
        self.silence_duration_bytes = 0
        self.bytes_since_provisional = 0

    def _find_best_acoustic_dip(
        self, search_start_bytes: int, search_end_bytes: int, frame_size: int = 1600
    ) -> Optional[int]:
        """
        Scans frames of size `frame_size` (50ms @ 16kHz 16-bit mono) within [search_start_bytes, search_end_bytes].
        Returns the byte offset of the lowest RMS energy dip if below threshold.
        """
        if search_end_bytes <= search_start_bytes or search_end_bytes > len(self.audio_buffer):
            return None

        best_offset = None
        min_rms = float("inf")

        step_size = max(2, frame_size // 2)
        step_size = step_size - (step_size % 2)

        for offset in range(search_start_bytes, search_end_bytes - frame_size + 1, step_size):
            frame = bytes(self.audio_buffer[offset : offset + frame_size])
            rms = calculate_rms(frame)
            if rms < min_rms:
                min_rms = rms
                best_offset = offset + (frame_size // 2)
                best_offset = best_offset - (best_offset % 2)

        if best_offset is not None and min_rms <= 0.025:
            return best_offset

        return None

    def add_audio(self, chunk: bytes) -> List[ChronosCut]:
        """
        Ingest audio bytes and return any cuts triggered (hard cut or soft cut).
        """
        cuts: List[ChronosCut] = []
        if not chunk:
            return cuts

        self.audio_buffer.extend(chunk)
        self.bytes_since_provisional += len(chunk)

        # Check silence
        rms = calculate_rms(chunk)
        if rms < self.SILENCE_RMS_CEILING:
            self.silence_duration_bytes += len(chunk)
        else:
            self.silence_duration_bytes = 0

        # 1. Continuous Speech Segmentation (Between 3.0s and 4.0s)
        if len(self.audio_buffer) >= self.MAX_CONTINUOUS_BYTES:
            # First, check if there is an acoustic energy dip (micro-pause between words) in the last 1.0s
            search_start = max(0, self.MAX_CONTINUOUS_BYTES - 32000)
            dip_offset = self._find_best_acoustic_dip(
                search_start_bytes=search_start,
                search_end_bytes=len(self.audio_buffer),
            )

            if dip_offset and dip_offset > self.MIN_PROCESS_BYTES:
                pcm_to_process = bytes(self.audio_buffer[:dip_offset])
                cuts.append(ChronosCut(cut_type=ChronosCutType.SOFT_CUT, pcm_data=pcm_to_process, is_final=True))
                self.audio_buffer = bytearray(self.audio_buffer[dip_offset:])
                self.silence_duration_bytes = 0
                self.bytes_since_provisional = len(self.audio_buffer)
                return cuts

            # Fallback Hard Cut (No acoustic dip found)
            pcm_to_process = bytes(self.audio_buffer)
            cuts.append(ChronosCut(cut_type=ChronosCutType.HARD_CUT, pcm_data=pcm_to_process, is_final=True))

            # Retain overlap tail so boundaries between words are not severed
            tail = bytes(self.audio_buffer[-self.OVERLAP_TAIL_BYTES:])
            if calculate_rms(tail) > self.SILENCE_RMS_CEILING:
                self.audio_buffer = bytearray(tail)
            else:
                self.audio_buffer = bytearray()

            self.silence_duration_bytes = 0
            self.bytes_since_provisional = len(self.audio_buffer)
            return cuts

        # 2. Soft Cut (Conversational pause >= 500ms)
        if self.silence_duration_bytes >= self.SILENCE_THRESHOLD_BYTES and len(self.audio_buffer) > self.MIN_PROCESS_BYTES:
            speech_bytes = len(self.audio_buffer) - self.silence_duration_bytes
            # Guard: If buffer only contains lingering overlap tail, discard rather than re-transcribing
            if speech_bytes <= self.OVERLAP_TAIL_BYTES:
                self.audio_buffer = bytearray()
                self.silence_duration_bytes = 0
                self.bytes_since_provisional = 0
                return cuts

            pcm_to_process = bytes(self.audio_buffer[:speech_bytes])
            cuts.append(ChronosCut(cut_type=ChronosCutType.SOFT_CUT, pcm_data=pcm_to_process, is_final=True))
            self.audio_buffer = bytearray()
            self.silence_duration_bytes = 0
            self.bytes_since_provisional = 0
            return cuts

        return cuts

    def should_trigger_provisional(self) -> bool:
        """Returns True if provisional tick interval reached."""
        if self.bytes_since_provisional >= self.PROVISIONAL_TICK_BYTES and len(self.audio_buffer) >= self.MIN_PROCESS_BYTES:
            self.bytes_since_provisional = 0
            return True
        return False

    def get_provisional_snapshot(self) -> Optional[bytes]:
        """Snapshot current buffer for provisional transcription without flushing."""
        if len(self.audio_buffer) < self.MIN_PROCESS_BYTES:
            return None
        return bytes(self.audio_buffer)

    def flush(self) -> Optional[ChronosCut]:
        """Forced flush on EOS."""
        if len(self.audio_buffer) >= self.MIN_PROCESS_BYTES:
            pcm = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            self.silence_duration_bytes = 0
            self.bytes_since_provisional = 0
            return ChronosCut(cut_type=ChronosCutType.FORCED_FLUSH, pcm_data=pcm, is_final=True)
        self.audio_buffer = bytearray()
        return None
