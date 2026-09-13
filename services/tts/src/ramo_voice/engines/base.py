"""
ramo_voice.engines.base
=======================
Abstract Base Class for all sovereign voice engines.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple
import numpy as np
from ..profiles import VoiceProfile


class BaseTTSEngine(ABC):
    def __init__(self, engine_id: str, sample_rate: int = 44100, device: str = "cpu"):
        self.engine_id = engine_id
        self.sample_rate = sample_rate
        self.device = device
        self.is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load model checkpoints or ONNX runtime sessions."""
        pass

    @abstractmethod
    async def generate_chunk(
        self,
        text: str,
        profile: VoiceProfile,
        seed: Optional[int] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize a single text chunk into audio waveform (float32, 1D)."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        text: str,
        profile: VoiceProfile,
        seed: Optional[int] = None,
        speed: float = 1.0,
    ) -> AsyncIterator[np.ndarray]:
        """Stream synthesized audio chunks for real-time WebSocket playback."""
        pass
