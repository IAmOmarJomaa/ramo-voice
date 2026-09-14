"""
ramo_translate.router
=====================
Unified Translation Router.
Coordinates:
1. Same-language short-circuit (0 ms)
2. Fast-Bypass Conversational Dictionary (0 ms)
3. 3-Tier Meeting Context & Sliding Window Tracking
4. BaseTranslationEngine dispatch (Local LLM or Universal API)
5. Action Item Detection
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .fast_bypass import check_fast_bypass, normalize_target_language
from .context_tracker import MeetingContextTracker
from .action_detector import detect_action_item, ActionItemType
from .engines.base import BaseTranslationEngine
from .engines.local_engine import LocalLLMEngine

logger = logging.getLogger("ramo_translate.router")


@dataclass
class TranslationResult:
    translated_text: str
    source_language: str
    target_language: str
    is_bypass: bool = False
    detected_action: Optional[ActionItemType] = None
    speaker_symbol: Optional[str] = None


class TranslationRouter:
    """
    Central orchestration router for real-time speech translation.
    """

    def __init__(
        self,
        engine: Optional[BaseTranslationEngine] = None,
        context_tracker: Optional[MeetingContextTracker] = None,
    ):
        self.engine = engine or LocalLLMEngine()
        self.context_tracker = context_tracker or MeetingContextTracker()

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        session_id: str = "default",
        speaker_id: str = "default",
    ) -> TranslationResult:
        clean_text = text.strip()
        if not clean_text:
            return TranslationResult(
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                is_bypass=True,
            )

        norm_target = normalize_target_language(target_lang)
        norm_source = normalize_target_language(source_lang)

        # 1. Same-language short circuit
        if norm_target == norm_source:
            return TranslationResult(
                translated_text=clean_text,
                source_language=source_lang,
                target_language=target_lang,
                is_bypass=True,
            )

        # 2. Fast-bypass conversational dictionary (0 ms)
        bypass_hit = check_fast_bypass(clean_text, target_lang)
        if bypass_hit is not None:
            logger.info(f"⚡ [Fast-Bypass Hit] '{clean_text}' -> '{bypass_hit}' ({target_lang})")
            # Still update context tracker with spoken turn
            self.context_tracker.add_utterance(session_id, speaker_id, clean_text)
            action = detect_action_item(clean_text)
            symbol = self.context_tracker.get_speaker_symbol(session_id, speaker_id)
            return TranslationResult(
                translated_text=bypass_hit,
                source_language=source_lang,
                target_language=target_lang,
                is_bypass=True,
                detected_action=action,
                speaker_symbol=symbol,
            )

        # 3. Retrieve 3-tier meeting context
        speaker_symbol = self.context_tracker.get_speaker_symbol(session_id, speaker_id)
        window = self.context_tracker.get_sliding_window(session_id)
        meeting_ctx = self.context_tracker.get_meeting_context(session_id)

        context_blocks = []
        if meeting_ctx:
            context_blocks.append(f"Meeting Context: {meeting_ctx}")
        if window:
            context_blocks.append("Recent Dialogue:\n" + "\n".join(window))
        full_context = "\n\n".join(context_blocks)

        # 4. Engine dispatch
        translated = await self.engine.generate_translation(
            text=clean_text,
            source_lang=norm_source,
            target_lang=norm_target,
            context_str=full_context,
            speaker_label=speaker_symbol,
        )

        # 5. Update sliding window
        self.context_tracker.add_utterance(session_id, speaker_id, clean_text)

        # 6. Action item detection
        action = detect_action_item(clean_text)

        return TranslationResult(
            translated_text=translated,
            source_language=source_lang,
            target_language=target_lang,
            is_bypass=False,
            detected_action=action,
            speaker_symbol=speaker_symbol,
        )
