"""
ramo_listen.engines
===================
STT Engine adapters.
"""
from .base import BaseSTTEngine
from .sensevoice_engine import SenseVoiceEngine

__all__ = ["BaseSTTEngine", "SenseVoiceEngine"]
