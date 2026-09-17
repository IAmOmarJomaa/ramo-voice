"""
ramo_gateway.anchor_matcher
===========================
Anchor Voice Matcher: maps speaker pitch (F0) and acoustic embeddings to Kokoro voice presets.
Provides instant 50ms fallback synthesis for speakers who have not yet reached the 4.5s
zero-shot cloning threshold, with deterministic multi-speaker voice variety.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger("ramo_gateway.anchor_matcher")

FEMALE_PRESETS = ["af_heart", "bf_emma", "af_bella", "af_nicole"]
MALE_PRESETS = ["am_adam", "bm_george", "am_michael", "bm_lewis"]


class AnchorVoiceMatcher:
    """
    Selects and caches pitch-matched Kokoro voice presets per speaker ID.
    """

    def __init__(self, pitch_split_hz: float = 175.0):
        self.pitch_split_hz = pitch_split_hz
        self._cache: Dict[str, str] = {}
        self._female_idx: int = 0
        self._male_idx: int = 0

    def match_voice(
        self,
        speaker_id: str,
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> str:
        """
        Return the assigned Kokoro preset for speaker_id.
        If not yet cached, estimates pitch F0 from audio to assign female vs male preset.
        """
        if speaker_id in self._cache:
            return self._cache[speaker_id]

        f0 = self._estimate_f0(audio, sample_rate) if audio is not None else 0.0

        if f0 > self.pitch_split_hz:
            voice = FEMALE_PRESETS[self._female_idx % len(FEMALE_PRESETS)]
            self._female_idx += 1
            gender = "female"
        else:
            voice = MALE_PRESETS[self._male_idx % len(MALE_PRESETS)]
            self._male_idx += 1
            gender = "male"

        self._cache[speaker_id] = voice
        logger.info(
            f"[ANCHOR_VOICE] 🎭 Assigned '{voice}' ({gender}, F0: {f0:.1f}Hz) to '{speaker_id}'"
        )
        return voice

    def get_assigned_voice(self, speaker_id: str) -> Optional[str]:
        return self._cache.get(speaker_id)

    def _estimate_f0(self, audio: np.ndarray, sample_rate: int) -> float:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if len(audio) < 512:
            return 0.0

        centered = audio - np.mean(audio)
        corr = np.correlate(centered, centered, mode="full")
        corr = corr[len(corr) // 2 :]

        min_lag = int(sample_rate / 400.0)
        max_lag = int(sample_rate / 60.0)
        if max_lag >= len(corr):
            return 0.0

        peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[peak_lag] > 0.20 * corr[0]:
            return float(sample_rate / peak_lag)
        return 0.0
