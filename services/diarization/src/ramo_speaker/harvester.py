"""
ramo_speaker.harvester
======================
Voiceprint Harvester: extracts isolated clean speech monologues and transcripts.
Features:
- 2-Tier Progressive Accumulator:
    * Tier 1 (4.5s): Fast-path instant zero-shot voice cloning threshold.
    * Tier 2 (10.0s): High-fidelity progressive upgrade.
- Zero-ASR Guarantee: Retains exact transcript permanently attached to audio,
  preventing the legacy bug where deleting text caused a 4.5s Whisper freeze.
- Crosstalk & Overlap Rejection: Overlapping turns are strictly excluded.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import soundfile as sf

logger = logging.getLogger("ramo_speaker.harvester")


@dataclass
class SpeakerTurn:
    speaker_id: str
    start_sec: float
    end_sec: float
    audio: np.ndarray
    transcript: str = ""
    is_overlap: bool = False

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class HarvestedProfile:
    speaker_id: str
    audio: np.ndarray
    transcript: str
    duration_sec: float
    tier: int  # 0: <4.5s, 1: >=4.5s, 2: >=10.0s
    ready_for_cloning: bool


class VoiceprintHarvester:
    """
    2-Tier Progressive Voiceprint Harvester for real-time meeting diarization.
    """

    def __init__(
        self,
        tier1_threshold_sec: float = 4.5,
        tier2_threshold_sec: float = 10.0,
        min_duration_sec: float = 1.0,
        target_sr: int = 16000,
    ):
        self.tier1_threshold_sec = tier1_threshold_sec
        self.tier2_threshold_sec = tier2_threshold_sec
        self.min_duration_sec = min_duration_sec
        self.target_sr = target_sr

        # Storage per speaker: list of (audio_chunk, transcript_chunk)
        self._speaker_audio: Dict[str, List[np.ndarray]] = {}
        self._speaker_transcripts: Dict[str, List[str]] = {}

    def add_turn(self, turn: SpeakerTurn) -> None:
        """Add a speech turn. Excludes turns marked as crosstalk/overlap."""
        if turn.is_overlap:
            logger.info(
                f"[HARVESTER] ⚠️ Excluded overlapping turn from voiceprint harvesting for '{turn.speaker_id}' ({turn.duration_sec:.2f}s)"
            )
            return

        audio = turn.audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Normalize audio peak
        max_abs = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        if max_abs > 0:
            audio = (audio / max_abs) * 0.95

        spk_id = turn.speaker_id
        if spk_id not in self._speaker_audio:
            self._speaker_audio[spk_id] = []
            self._speaker_transcripts[spk_id] = []

        self._speaker_audio[spk_id].append(audio)
        if turn.transcript:
            self._speaker_transcripts[spk_id].append(turn.transcript.strip())

        total_samples = sum(len(c) for c in self._speaker_audio[spk_id])
        total_sec = total_samples / self.target_sr
        turns_count = len(self._speaker_audio[spk_id])

        logger.info(
            f"[HARVESTER] 🎙️ Audio turn added for '{spk_id}': +{turn.duration_sec:.2f}s | "
            f"Total accumulated: {total_sec:.2f}s / {self.tier2_threshold_sec:.1f}s ({turns_count} turns)"
        )

        if total_sec >= self.tier2_threshold_sec:
            logger.info(
                f"[HARVESTER] 🚀 '{spk_id}' achieved TIER 2 ({total_sec:.2f}s >= {self.tier2_threshold_sec:.1f}s)! High-Fidelity Profile Ready!"
            )
        elif total_sec >= self.tier1_threshold_sec:
            logger.info(
                f"[HARVESTER] 🚀 '{spk_id}' achieved TIER 1 ({total_sec:.2f}s >= {self.tier1_threshold_sec:.1f}s)! INSTANT ZERO-SHOT CLONING READY!"
            )

    def has_candidate(self, speaker_id: str) -> bool:
        return speaker_id in self._speaker_audio and len(self._speaker_audio[speaker_id]) > 0

    def get_profile(self, speaker_id: str) -> Optional[HarvestedProfile]:
        """
        Return the 2-Tier HarvestedProfile with concatenated audio and transcript.
        """
        if not self.has_candidate(speaker_id):
            return None

        chunks = self._speaker_audio[speaker_id]
        transcripts = self._speaker_transcripts[speaker_id]

        concatenated_audio = np.concatenate(chunks)
        combined_transcript = " ".join(transcripts).strip()
        duration_sec = len(concatenated_audio) / self.target_sr

        if duration_sec >= self.tier2_threshold_sec:
            tier = 2
            ready = True
        elif duration_sec >= self.tier1_threshold_sec:
            tier = 1
            ready = True
        else:
            tier = 0
            ready = False

        return HarvestedProfile(
            speaker_id=speaker_id,
            audio=concatenated_audio,
            transcript=combined_transcript,
            duration_sec=round(duration_sec, 2),
            tier=tier,
            ready_for_cloning=ready,
        )

    def get_best_sample(self, speaker_id: str) -> Optional[np.ndarray]:
        """Return the concatenated audio sample for the specified speaker."""
        profile = self.get_profile(speaker_id)
        return profile.audio if profile else None

    def list_harvested_speakers(self) -> List[str]:
        return [spk for spk, chunks in self._speaker_audio.items() if len(chunks) > 0]

    def to_tts_registration_payload(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """
        Package harvested profile into Microservice 4 registration payload:
        POST /v1/voices/register_speaker
        """
        profile = self.get_profile(speaker_id)
        if not profile or not profile.ready_for_cloning:
            return None

        # Encode audio as base64 WAV
        buf = io.BytesIO()
        sf.write(buf, profile.audio, self.target_sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "speaker_id": profile.speaker_id,
            "name": profile.speaker_id,
            "audio_base64": audio_b64,
            "prompt_text": profile.transcript,
            "duration_sec": profile.duration_sec,
            "confidence": 0.98,
        }
