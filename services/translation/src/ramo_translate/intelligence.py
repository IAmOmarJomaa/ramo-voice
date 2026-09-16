"""
ramo_translate.intelligence
===========================
Node 2: Executive Meeting Intelligence Synthesizer.
Asynchronously analyzes multi-turn conversational dialogue to extract:
1. Action items (task, assignee, assigned_by, deadline)
2. Direct orders and triggers
3. Statements/claims requiring fact-checking
4. High-level key meeting notes
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ramo_translate.intelligence")

INTELLIGENCE_SYSTEM_PROMPT = (
    "You are an executive meeting intelligence assistant.\n"
    "Analyze the recent conversation between participants.\n"
    "Extract actionable information and summarize key context.\n\n"
    "RULES:\n"
    "1. Extract 'action_items': work commitments, tasks assigned, or deadlines (who does what by when).\n"
    "2. Extract 'direct_orders': urgent commands, immediate requests, or operational triggers.\n"
    "3. Extract 'verification_claims': specific metrics, percentages, revenue figures, or factual claims.\n"
    "4. Extract 'key_notes': significant decisions, consensus, or important discussion points.\n"
    "5. If nothing actionable occurred, return empty arrays.\n\n"
    "Output ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "action_items": [{"task": "...", "assignee": "...", "assigned_by": "...", "deadline": "..."}],\n'
    '  "direct_orders": [{"order": "...", "target": "..."}],\n'
    '  "verification_claims": ["..."],\n'
    '  "key_notes": ["..."]\n'
    "}"
)


class MeetingIntelligenceSynthesizer:
    """
    Dedicated intelligence synthesis engine for multi-turn meeting context.
    """

    def __init__(self, system_prompt: str = INTELLIGENCE_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def format_prompt(self, dialogue_turns: List[str]) -> str:
        """
        Format the intelligence extraction prompt using XML sandboxed dialogue.
        """
        dialogue_text = "\n".join(t.strip() for t in dialogue_turns if t.strip())
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<recent_dialogue>\n{dialogue_text}\n</recent_dialogue>\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def parse_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured meeting intelligence dictionary.
        """
        fallback = {
            "action_items": [],
            "direct_orders": [],
            "verification_claims": [],
            "key_notes": [],
        }

        if not raw_output or not raw_output.strip():
            return fallback

        # 1. Direct JSON extraction
        try:
            bracket_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}')
            if bracket_idx != -1 and end_idx > bracket_idx:
                parsed = json.loads(raw_output[bracket_idx : end_idx + 1])
                return {
                    "action_items": parsed.get("action_items", []),
                    "direct_orders": parsed.get("direct_orders", []),
                    "verification_claims": parsed.get("verification_claims", []),
                    "key_notes": parsed.get("key_notes", []),
                }
        except Exception as e:
            logger.debug(f"JSON parse failed on intelligence output: {e}")

        return fallback

    async def analyze(self, dialogue_turns: List[str], engine) -> Dict[str, Any]:
        """
        Run intelligence analysis against an LLM engine.
        """
        if not dialogue_turns:
            return self.parse_output("")

        prompt = self.format_prompt(dialogue_turns)
        try:
            raw = await engine.generate_raw(prompt, max_new_tokens=256)
            return self.parse_output(raw)
        except Exception as e:
            logger.error(f"Intelligence synthesis failed: {e}", exc_info=True)
            return self.parse_output("")
