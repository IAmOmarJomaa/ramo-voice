"""
ramo_gateway.chronos
====================
Chronos Dynamic Auto-Cut Buffer (SOTA Dual-Path VAD & Zero-Overlap Architecture).
Handles:
- Dual-Path VAD:
    * Normal conversational pause: 300ms silence (<4.0s accumulated audio)
    * Relaxed breath pause: 120ms silence (4.0s - 7.0s accumulated audio)
- 7.0s Monologue Hard Ceiling for Align-then-Commit (ATC)
- Acoustic dip boundary detection (50ms micro-pause)
- Zero overlap tail retention (OVERLAP_TAIL_BYTES = 0)
- 600ms provisional tick for live streaming STT preview
"""

import enum
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
    Enforces SOTA Dual-Path VAD, Zero Acoustic Overlap, and Monologue Ceiling.
    """

    # Audio format: 16kHz, 16-bit Mono = 32,000 bytes/second
    SAMPLE_RATE: int = 16000
    BYTES_PER_SEC: int = 32000

    # Operational thresholds
    SOFT_CEILING_BYTES: int = 128000       # 4.0 seconds
    HARD_CEILING_BYTES: int = 224000       # 7.0 seconds (Monologue Hard Ceiling)
    NORMAL_PAUSE_BYTES: int = 9600         # 300ms normal conversational pause
    RELAXED_PAUSE_BYTES: int = 3840        # 120ms relaxed breath pause
    PROVISIONAL_TICK_BYTES: int = 19200    # 600ms provisional preview interval
    SILENCE_RMS_CEILING: float = 0.015     # RMS below this is considered conversational pause
    MIN_PROCESS_BYTES: int = 3200          # 100ms minimum audio before processing
    OVERLAP_TAIL_BYTES: int = 0            # SOTA Zero Overlap invariant (NO repeated audio)

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
        Ingest audio bytes and return any cuts triggered (soft cut, dip cut, or hard cut).
        Enforces Dual-Path VAD and ZERO acoustic overlap.
        """
        cuts: List[ChronosCut] = []
        if not chunk:
            return cuts

        self.audio_buffer.extend(chunk)
        self.bytes_since_provisional += len(chunk)

        # Check silence envelope
        rms = calculate_rms(chunk)
        if rms < self.SILENCE_RMS_CEILING:
            self.silence_duration_bytes += len(chunk)
        else:
            self.silence_duration_bytes = 0

        buffer_len = len(self.audio_buffer)

        # Determine active silence threshold via Dual-Path VAD
        active_silence_threshold = (
            self.NORMAL_PAUSE_BYTES
            if buffer_len < self.SOFT_CEILING_BYTES
            else self.RELAXED_PAUSE_BYTES
        )

        # Path 1: Conversational / Breath Pause Detected (Soft Cut)
        if self.silence_duration_bytes >= active_silence_threshold and buffer_len >= self.MIN_PROCESS_BYTES:
            speech_bytes = buffer_len - self.silence_duration_bytes
            if speech_bytes < self.MIN_PROCESS_BYTES:
                # Buffer is essentially pure silence; purge cleanly
                self.audio_buffer = bytearray()
                self.silence_duration_bytes = 0
                self.bytes_since_provisional = 0
                return cuts

            pcm_to_process = bytes(self.audio_buffer[:speech_bytes])
            cuts.append(ChronosCut(cut_type=ChronosCutType.SOFT_CUT, pcm_data=pcm_to_process, is_final=True))

            # Zero overlap tail: truncate audio buffer cleanly without retaining previous speech!
            remaining = bytes(self.audio_buffer[speech_bytes:])
            if calculate_rms(remaining) < self.SILENCE_RMS_CEILING:
                self.audio_buffer = bytearray()
                self.silence_duration_bytes = 0
            else:
                self.audio_buffer = bytearray(remaining)

            self.bytes_since_provisional = len(self.audio_buffer)
            return cuts

        # Path 2: Monologue Hard Ceiling (7.0s) reached without pause
        if buffer_len >= self.HARD_CEILING_BYTES:
            # First, search for an acoustic energy dip (micro-pause between words) in the last 2.0s
            search_start = max(0, self.HARD_CEILING_BYTES - 64000)
            dip_offset = self._find_best_acoustic_dip(
                search_start_bytes=search_start,
                search_end_bytes=buffer_len,
            )

            if dip_offset and dip_offset > self.MIN_PROCESS_BYTES:
                pcm_to_process = bytes(self.audio_buffer[:dip_offset])
                cuts.append(ChronosCut(cut_type=ChronosCutType.SOFT_CUT, pcm_data=pcm_to_process, is_final=True))
                self.audio_buffer = bytearray(self.audio_buffer[dip_offset:])
                self.silence_duration_bytes = 0
                self.bytes_since_provisional = len(self.audio_buffer)
                return cuts

            # Fallback Hard Cut: Clean truncation with ZERO overlap tail
            pcm_to_process = bytes(self.audio_buffer)
            cuts.append(ChronosCut(cut_type=ChronosCutType.HARD_CUT, pcm_data=pcm_to_process, is_final=True))
            self.audio_buffer = bytearray()
            self.silence_duration_bytes = 0
            self.bytes_since_provisional = 0
            return cuts

        return cuts

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
        return bytes(self.audio_buffer)

    def truncate_at_offset(self, byte_offset: int) -> None:
        """Cleanly truncate audio buffer at exact byte offset for Align-then-Commit."""
        if 0 < byte_offset <= len(self.audio_buffer):
            self.audio_buffer = bytearray(self.audio_buffer[byte_offset:])
            self.silence_duration_bytes = 0
            self.bytes_since_provisional = len(self.audio_buffer)

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

