"""
ramo_voice.profiles
===================
In-memory zero-IO Voice Profile & Speaker Latent Cache.
Adapted from Coqui XTTS-v2 and Voicebox's profile architecture.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger("ramo_voice.profiles")


@dataclass
class VoiceProfile:
    voice_id: str
    name: str
    voice_type: str = "preset"  # "preset" or "cloned"
    sample_rate: int = 44100
    conditioning_latents: Optional[np.ndarray] = None
    speaker_embedding: Optional[np.ndarray] = None
    reference_wav_path: Optional[str] = None
    language: str = "en"


class VoiceProfileStore:
    """
    In-memory thread-safe registry for preset and cloned voice profiles.
    Ensures zero disk I/O on hot inference paths.
    """
    def __init__(self):
        self._profiles: Dict[str, VoiceProfile] = {}

    def register(self, profile: VoiceProfile) -> None:
        self._profiles[profile.voice_id] = profile
        logger.info(f"Registered voice profile: {profile.voice_id} ({profile.voice_type}, {profile.sample_rate}Hz)")

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(voice_id)

    def list_profiles(self) -> Dict[str, VoiceProfile]:
        return dict(self._profiles)

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._profiles


# Global singleton instance
default_store = VoiceProfileStore()
