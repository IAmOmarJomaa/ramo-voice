"""
ramo_translate.engines.base
===========================
Base abstract interface for translation engines.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class BaseTranslationEngine(ABC):
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load model weights or initialize API clients."""
        pass

    @abstractmethod
    async def generate_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context_str: str = "",
        speaker_label: str = "",
    ) -> str:
        """Translate text with conversational context."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context_str: str = "",
        speaker_label: str = "",
    ) -> AsyncIterator[str]:
        """Stream translation tokens."""
        pass
