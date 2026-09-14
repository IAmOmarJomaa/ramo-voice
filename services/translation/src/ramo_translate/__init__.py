"""
ramo_translate
==============
Sovereign Real-Time Meeting Translation & LLM Intelligence Microservice.
Features:
- Fast-Bypass 0ms Conversational Dictionary (EN, FR, ES, DE, AR)
- 3-Tier Context Architecture with Speaker Symbol Masking ([S_A], [S_B])
- Meeting Action Item Detection (task_delegation, schedule_change, fact_check)
- Safe Memory Footprint on Colab T4 (~2.0 GB VRAM)
- Universal OpenAI/Groq/Cloud API Fallback
"""

from .fast_bypass import check_fast_bypass, FAST_BYPASS_DICT
from .context_tracker import MeetingContextTracker
from .action_detector import detect_action_item
from .router import TranslationRouter, TranslationResult

__all__ = [
    "check_fast_bypass",
    "FAST_BYPASS_DICT",
    "MeetingContextTracker",
    "detect_action_item",
    "TranslationRouter",
    "TranslationResult",
]
