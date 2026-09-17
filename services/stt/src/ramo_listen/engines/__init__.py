"""
ramo_listen.engines
===================
STT Engine adapters.
"""
from .base import BaseSTTEngine
from .whisper_engine import WhisperSTTEngine
from .sensevoice_engine import SenseVoiceEngine

__all__ = ["BaseSTTEngine", "WhisperSTTEngine", "SenseVoiceEngine"]
