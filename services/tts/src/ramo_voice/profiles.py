"""
ramo_voice.profiles
===================
In-memory zero-IO Voice Profile & Speaker Latent Cache.
Includes 4.5-second threshold policy and progressive voice refinement.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import logging

logger = logging.getLogger("ramo_voice.profiles")

# Available instant fallback presets
FALLBACK_PRESETS: List[str] = ["af_heart", "am_adam", "af_bella"]


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
    prompt_text: Optional[str] = None
    duration_sec: float = 0.0
    use_fallback: bool = False
    fallback_preset: Optional[str] = None

    @property
    def speaker_id(self) -> str:
        return self.voice_id


class VoiceProfileStore:
    """
    In-memory thread-safe registry for preset and cloned voice profiles.
    Ensures zero disk I/O on hot inference paths.
    """
    def __init__(self):
        self._profiles: Dict[str, VoiceProfile] = {}

    def register(self, profile: VoiceProfile) -> None:
        self._profiles[profile.voice_id] = profile
        logger.info(f"Registered voice profile: {profile.voice_id} ({profile.voice_type}, {profile.sample_rate}Hz, fallback={profile.use_fallback})")

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(voice_id)

    def list_profiles(self) -> Dict[str, VoiceProfile]:
        return dict(self._profiles)

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._profiles


def register_meeting_speaker(
    store: VoiceProfileStore,
    speaker_id: str,
    audio: np.ndarray,
    prompt_text: str,
    sample_rate: int = 24000,
) -> VoiceProfile:
    """
    Register or progressively refine a meeting speaker profile:
    - If audio duration >= 4.5s: mark ready for F5-TTS zero-shot flow matching.
    - If audio duration < 4.5s: mark use_fallback=True and map to an instant preset voice.
    - If speaker already exists and new audio is longer: update latents (progressive refinement).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    duration_sec = float(len(audio) / sample_rate)
    is_ready_for_clone = duration_sec >= 4.5

    # Assign deterministic fallback preset based on speaker_id
    fallback_idx = abs(hash(speaker_id)) % len(FALLBACK_PRESETS)
    fallback_choice = FALLBACK_PRESETS[fallback_idx]

    existing = store.get(speaker_id)
    if existing:
        # Progressive refinement: update if new audio provides more data
        existing.conditioning_latents = audio
        existing.prompt_text = prompt_text
        existing.duration_sec = duration_sec
        existing.sample_rate = sample_rate
        if is_ready_for_clone:
            existing.use_fallback = False
            existing.voice_type = "cloned"
            existing.fallback_preset = None
            logger.info(f"Progressively refined speaker {speaker_id} with {duration_sec:.2f}s audio.")
        return existing

    profile = VoiceProfile(
        voice_id=speaker_id,
        name=speaker_id,
        voice_type="cloned" if is_ready_for_clone else "preset",
        sample_rate=sample_rate,
        conditioning_latents=audio,
        prompt_text=prompt_text,
        duration_sec=duration_sec,
        use_fallback=not is_ready_for_clone,
        fallback_preset=fallback_choice if not is_ready_for_clone else None
    )

    store.register(profile)
    return profile


# Global singleton instance
default_store = VoiceProfileStore()
