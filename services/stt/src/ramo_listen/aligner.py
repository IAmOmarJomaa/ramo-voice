"""
ramo_listen.aligner
===================
Temporal Word Aligner: assigns Whisper word-level timestamps to speaker intervals.
Uses midpoint temporal intersection to deterministically prevent boundary-edge ambiguities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ramo_listen.aligner")


@dataclass
class SpeakerInterval:
    speaker_id: str
    start_sec: float
    end_sec: float


@dataclass
class SpeakerAlignedSegment:
    speaker_id: str
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)
    start_sec: float = 0.0
    end_sec: float = 0.0


class TemporalWordAligner:
    """
    Maps Whisper word tokens (with start & end timestamps) to speaker intervals.
    Assigns each word based on its temporal midpoint (start + end) / 2.
    """

    def align_words_to_speakers(
        self,
        words: List[Dict[str, Any]],
        intervals: List[SpeakerInterval],
    ) -> List[SpeakerAlignedSegment]:
        if not words:
            return []

        if not intervals:
            # Default to single unknown speaker if intervals not provided
            text = " ".join(w["word"] for w in words)
            return [
                SpeakerAlignedSegment(
                    speaker_id="Speaker 1",
                    text=text,
                    words=words,
                    start_sec=words[0].get("start", 0.0),
                    end_sec=words[-1].get("end", 0.0),
                )
            ]

        # Group words by matched interval
        assigned_words: List[tuple[str, Dict[str, Any]]] = []

        for w in words:
            w_start = w.get("start", 0.0)
            w_end = w.get("end", w_start + 0.1)
            midpoint = (w_start + w_end) / 2.0

            matched_spk: Optional[str] = None
            min_dist = float("inf")

            # 1. Direct containment check
            for interval in intervals:
                if interval.start_sec <= midpoint <= interval.end_sec:
                    matched_spk = interval.speaker_id
                    break

            # 2. Nearest boundary fallback if slightly outside intervals
            if matched_spk is None:
                for interval in intervals:
                    dist = min(abs(midpoint - interval.start_sec), abs(midpoint - interval.end_sec))
                    if dist < min_dist:
                        min_dist = dist
                        matched_spk = interval.speaker_id

            assigned_words.append((matched_spk or intervals[0].speaker_id, w))

        # Collapse contiguous runs of the same speaker into aligned segments
        segments: List[SpeakerAlignedSegment] = []
        current_spk: Optional[str] = None
        current_words: List[Dict[str, Any]] = []

        for spk_id, w in assigned_words:
            if current_spk is None:
                current_spk = spk_id
                current_words = [w]
            elif spk_id == current_spk:
                current_words.append(w)
            else:
                # Flush previous segment
                seg_text = " ".join(item["word"] for item in current_words)
                segments.append(
                    SpeakerAlignedSegment(
                        speaker_id=current_spk,
                        text=seg_text,
                        words=current_words,
                        start_sec=current_words[0].get("start", 0.0),
                        end_sec=current_words[-1].get("end", 0.0),
                    )
                )
                current_spk = spk_id
                current_words = [w]

        if current_words and current_spk is not None:
            seg_text = " ".join(item["word"] for item in current_words)
            segments.append(
                SpeakerAlignedSegment(
                    speaker_id=current_spk,
                    text=seg_text,
                    words=current_words,
                    start_sec=current_words[0].get("start", 0.0),
                    end_sec=current_words[-1].get("end", 0.0),
                )
            )

        logger.debug(
            f"[ALIGNER] Aligned {len(words)} words across {len(intervals)} intervals into {len(segments)} segment(s)"
        )
        return segments
