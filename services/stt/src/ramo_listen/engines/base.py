"""
ramo_listen.engines.base
========================
Abstract base class for streaming and batch speech-to-text engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator
import numpy as np


class BaseSTTEngine(ABC):
    def __init__(self, engine_id: str, sample_rate: int = 16000, device: str = "cpu"):
        self.engine_id = engine_id
        self.sample_rate = sample_rate
        self.device = device
        self.is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load ASR acoustic model weights."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe an audio buffer, returning text, emotion, and confidence."""
        pass
