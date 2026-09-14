"""
ramo_translate.action_detector
==============================
Meeting Intelligence Action Item Classifier.
Classifies utterances into:
- 'task_delegation': Explicit commitment or work assignment
- 'schedule_change': Calendar, time, or date adjustments
- 'fact_check': Numerical, metric, or statistical assertions
"""

import re
from typing import Optional, Literal

ActionItemType = Literal["task_delegation", "schedule_change", "fact_check"]

TASK_PATTERNS = [
    re.compile(r"\b(i\s+will|i'll|i\s+am\s+going\s+to)\s+(send|deliver|finish|write|code|prepare|implement|email|share|handle|lead|deploy|fix|update|create)\b", re.IGNORECASE),
    re.compile(r"\b(can\s+you|could\s+you|please)\s+(take\s+over|follow\s+up|review|verify|check|look\s+into)\b", re.IGNORECASE),
    re.compile(r"\b(assigned\s+to|action\s+item\s+for|on\s+my\s+todo)\b", re.IGNORECASE),
    re.compile(r"\b(by\s+tomorrow|by\s+end\s+of\s+day|by\s+eod|by\s+next\s+week)\b", re.IGNORECASE),
]

SCHEDULE_PATTERNS = [
    re.compile(r"\b(reschedule|postpone|push\s+back|move\s+to|schedule\s+for)\b", re.IGNORECASE),
    re.compile(r"\b(calendar\s+invite|meeting\s+at|sync\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b", re.IGNORECASE),
    re.compile(r"\b(let's\s+meet|let's\s+reschedule|let's\s+push)\b", re.IGNORECASE),
]

FACT_PATTERNS = [
    re.compile(r"\b(\d+(\.\d+)?\s*(percent|%))\b", re.IGNORECASE),
    re.compile(r"\b(\$?\d+(\.\d+)?\s*(million|billion|thousand|k|m|b))\b", re.IGNORECASE),
    re.compile(r"\b(retention\s+was|revenue\s+was|churn\s+rate|latency\s+dropped|performance\s+increased)\b", re.IGNORECASE),
    re.compile(r"\b(according\s+to\s+the\s+metrics|data\s+shows|statistics\s+indicate)\b", re.IGNORECASE),
]


def detect_action_item(text: str) -> Optional[ActionItemType]:
    """
    Detect actionable meeting items using high-precision acoustic regex heuristics.
    """
    if not text:
        return None

    clean = text.strip()

    # Schedule changes take priority
    for pat in SCHEDULE_PATTERNS:
        if pat.search(clean):
            return "schedule_change"

    # Task commitments & work delegation
    for pat in TASK_PATTERNS:
        if pat.search(clean):
            return "task_delegation"

    # Metrics & hard statistics
    for pat in FACT_PATTERNS:
        if pat.search(clean):
            return "fact_check"

    return None
