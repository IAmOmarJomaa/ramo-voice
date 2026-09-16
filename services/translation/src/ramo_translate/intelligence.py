"""
ramo_translate.intelligence
===========================
Node 2: Sovereign Enterprise Meeting Intelligence Synthesizer (The Scout).
Implements the SOTA 7-Category Enterprise Taxonomy, Temporal Deduplication,
and XML-Sandboxed Rolling Buffer Observation:

1. ACTION_ITEM: Tasks, assignees, assigners, deadlines, dependencies, urgency.
2. DECISION_ADR: Ratified decisions, rationale, rejected alternatives, dissents.
3. SCHEDULE_DYNAMICS: Proposed meetings, rescheduling, calendar checks.
4. VERIFICATION_CLAIM: Hard metrics, numbers, SLA promises, regulatory facts.
5. BLOCKER_RISK: Impediments, affected workflows, severity levels.
6. UNANSWERED_QUESTION: Inquiries that got bypassed or derailed without answer.
7. SALES_SIGNAL: Competitor mentions, pricing pushback, churn threats.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("ramo_translate.intelligence")

COMMANDER_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class SemanticEvent:
    event_id: str
    event_category: str
    confidence_score: float
    source_speakers: List[str]
    trigger_quote: str
    structured_payload: Dict[str, Any]
    requires_commander_routing: bool = field(init=False)

    def __post_init__(self):
        self.requires_commander_routing = self.confidence_score >= COMMANDER_CONFIDENCE_THRESHOLD


@dataclass
class DeduplicationResult:
    is_update: bool
    canonical_event_id: str
    event: SemanticEvent


class TemporalDeduplicator:
    """
    In-memory semantic cache preventing duplicate triggers when participants
    deliberate on the same task or decision across continuous rolling turns.
    """

    def __init__(self, similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold
        self._active_events: Dict[str, SemanticEvent] = {}

    @property
    def active_event_count(self) -> int:
        return len(self._active_events)

    def get_event(self, event_id: str) -> Optional[SemanticEvent]:
        return self._active_events.get(event_id)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        words = re.findall(r"\b\w{3,}\b", text.lower())
        return set(words)

    def _compute_similarity(self, a_text: str, b_text: str) -> float:
        tokens_a = self._tokenize(a_text)
        tokens_b = self._tokenize(b_text)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        jaccard = float(intersection / union) if union > 0 else 0.0
        min_len = min(len(tokens_a), len(tokens_b))
        containment = float(intersection / min_len) if min_len > 0 else 0.0
        return max(jaccard, containment)

    def process_event(self, event: SemanticEvent) -> DeduplicationResult:
        """
        Evaluate incoming event against active cache.
        If a similar event exists in the same category, merges payload and marks as update.
        """
        best_match_id: Optional[str] = None
        best_sim = 0.0

        for existing_id, existing_evt in self._active_events.items():
            if existing_evt.event_category != event.event_category:
                continue

            # 1. Direct primary key equality or containment
            for pk in ("task", "decision", "blocker", "claim", "question"):
                if pk in event.structured_payload and pk in existing_evt.structured_payload:
                    va = str(event.structured_payload[pk]).lower().strip()
                    vb = str(existing_evt.structured_payload[pk]).lower().strip()
                    if va and vb and (va == vb or va in vb or vb in va):
                        best_sim = 1.0
                        best_match_id = existing_id
                        break

            if best_match_id is not None and best_sim >= 1.0:
                break

            # 2. Text quote & summary similarity
            target_summary = event.trigger_quote
            existing_summary = existing_evt.trigger_quote

            sim = self._compute_similarity(target_summary, existing_summary)
            if sim > best_sim:
                best_sim = sim
                best_match_id = existing_id

        if best_match_id is not None and best_sim >= self.similarity_threshold:
            canonical = self._active_events[best_match_id]
            # Merge payload updates
            for k, v in event.structured_payload.items():
                if v and (v not in ("Not specified", "None specified", None)):
                    canonical.structured_payload[k] = v
            # Update trigger quote if new one is richer
            if len(event.trigger_quote) > len(canonical.trigger_quote):
                canonical.trigger_quote = event.trigger_quote
            canonical.confidence_score = max(canonical.confidence_score, event.confidence_score)
            return DeduplicationResult(
                is_update=True,
                canonical_event_id=canonical.event_id,
                event=canonical,
            )

        # Register new event
        self._active_events[event.event_id] = event
        return DeduplicationResult(
            is_update=False,
            canonical_event_id=event.event_id,
            event=event,
        )


SOTA_INTELLIGENCE_SYSTEM_PROMPT = (
    "You are the Scout, an ambient corporate meeting intelligence filter.\n"
    "Your exclusive directive is to analyze the conversation in <recent_dialogue> and extract key enterprise semantic events.\n\n"
    "CATEGORIES TO EXTRACT:\n"
    "1. 'action_items': tasks, commitments, assignees, deadlines, dependencies, urgency (high/medium/low).\n"
    "2. 'decisions': formal consensus, rationale, rejected alternatives, dissenting views.\n"
    "3. 'schedule_dynamics': rescheduling proposals, meeting invites, proposed dates/times, calendar checks.\n"
    "4. 'verification_claims': statistics, metrics, SLA promises, numerical claims to fact-check.\n"
    "5. 'blockers_and_risks': impediments, affected areas, severity (critical/high/medium).\n"
    "6. 'unanswered_questions': critical questions asked that got derailed or ignored.\n"
    "7. 'key_notes': significant context, announcements, or attendees.\n\n"
    "RULES:\n"
    "- If an event occurs, compute a confidence_score (0.0 to 1.0).\n"
    "- If no events occur, return empty arrays.\n"
    "- Output ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "action_items": [{"task": "...", "assignee": "...", "assigned_by": "...", "deadline": "...", "dependencies": "...", "urgency": "..."}],\n'
    '  "decisions": [{"decision": "...", "rationale": "...", "rejected_alternatives": [], "dissenting_views": "..."}],\n'
    '  "schedule_dynamics": [{"event_type": "...", "proposed_time": "...", "participants": [], "check_availability": true}],\n'
    '  "verification_claims": [{"claim": "...", "speaker": "...", "metric_value": "..."}],\n'
    '  "blockers_and_risks": [{"blocker": "...", "affected_area": "...", "severity": "..."}],\n'
    '  "unanswered_questions": [{"question": "...", "asked_by": "...", "target_person": "..."}],\n'
    '  "key_notes": ["..."],\n'
    '  "events": [\n'
    '    {\n'
    '      "event_id": "...",\n'
    '      "event_category": "ACTION_ITEM|DECISION_ADR|SCHEDULE_DYNAMICS|VERIFICATION_CLAIM|BLOCKER_RISK|UNANSWERED_QUESTION",\n'
    '      "confidence_score": 0.95,\n'
    '      "source_speakers": ["..."],\n'
    '      "trigger_quote": "...",\n'
    '      "structured_payload": {}\n'
    '    }\n'
    '  ]\n'
    "}"
)


class MeetingIntelligenceSynthesizer:
    """
    Dedicated intelligence synthesis engine for multi-turn meeting context.
    """

    def __init__(self, system_prompt: str = SOTA_INTELLIGENCE_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.deduplicator = TemporalDeduplicator(similarity_threshold=0.80)

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
        fallback: Dict[str, Any] = {
            "action_items": [],
            "direct_orders": [],
            "decisions": [],
            "schedule_dynamics": [],
            "verification_claims": [],
            "blockers_and_risks": [],
            "unanswered_questions": [],
            "key_notes": [],
            "events": [],
        }

        if not raw_output or not raw_output.strip():
            return fallback

        try:
            bracket_idx = raw_output.find("{")
            end_idx = raw_output.rfind("}")
            if bracket_idx != -1 and end_idx > bracket_idx:
                parsed = json.loads(raw_output[bracket_idx : end_idx + 1])

                events_list: List[SemanticEvent] = []
                raw_events = parsed.get("events", [])
                for ev in raw_events:
                    try:
                        events_list.append(
                            SemanticEvent(
                                event_id=ev.get("event_id") or f"evt-{uuid.uuid4().hex[:8]}",
                                event_category=ev.get("event_category", "ACTION_ITEM"),
                                confidence_score=float(ev.get("confidence_score", 0.90)),
                                source_speakers=ev.get("source_speakers", []),
                                trigger_quote=ev.get("trigger_quote", ""),
                                structured_payload=ev.get("structured_payload", {}),
                            )
                        )
                    except Exception:
                        pass

                # Synthesize standard verification claims into backward compatible dict/str
                raw_claims = parsed.get("verification_claims", [])
                formatted_claims = []
                for c in raw_claims:
                    if isinstance(c, dict):
                        formatted_claims.append(c)
                    elif isinstance(c, str):
                        formatted_claims.append({"claim": c, "speaker": "", "metric_value": ""})

                return {
                    "action_items": parsed.get("action_items", []),
                    "direct_orders": parsed.get("direct_orders", []),
                    "decisions": parsed.get("decisions", []),
                    "schedule_dynamics": parsed.get("schedule_dynamics", []),
                    "verification_claims": formatted_claims,
                    "blockers_and_risks": parsed.get("blockers_and_risks", []),
                    "unanswered_questions": parsed.get("unanswered_questions", []),
                    "key_notes": parsed.get("key_notes", []),
                    "events": events_list,
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
            raw = await engine.generate_raw(prompt, max_new_tokens=384)
            logger.debug(f"Raw intelligence generation: {raw}")
            parsed = self.parse_output(raw)

            # Apply temporal deduplication to any emitted events
            if parsed.get("events"):
                deduped_events = []
                for ev in parsed["events"]:
                    res = self.deduplicator.process_event(ev)
                    deduped_events.append(res.event)
                parsed["events"] = deduped_events

            return parsed
        except Exception as e:
            logger.error(f"Intelligence analysis failed: {e}", exc_info=True)
            return self.parse_output("")
