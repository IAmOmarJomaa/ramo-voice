"""
ramo_listen.text_cleaner
========================
Post-processing filters for transcribed speech:
- Strips Whisper repetitive hallucination loops and YouTube artifacts
- Normalizes English and French contractions (e.g. "it 's" -> "it's")
- Cleans conversational filler words (e.g. "um", "uh") inspired by ghost-pepper
"""

import re
from typing import List

# Common ASR hallucinations & subtitle boilerplate
HALLUCINATION_PATTERNS = [
    re.compile(r"thank you for watching\s*[!.]*", re.IGNORECASE),
    re.compile(r"subtitles by.*", re.IGNORECASE),
    re.compile(r"please subscribe.*", re.IGNORECASE),
    re.compile(r"like and subscribe.*", re.IGNORECASE),
    re.compile(r"sous-titres réalisés par.*", re.IGNORECASE),
    re.compile(r"merci d'avoir regardé.*", re.IGNORECASE),
]

# Contraction normalization patterns
CONTRACTION_PATTERNS = [
    (re.compile(r"\b(\w+)\s+'\s*(s|m|d|ll|re|ve)\b", re.IGNORECASE), r"\1'\2"),
    (re.compile(r"\b(can|don|won|doesn|didn|isn|aren|wasn|weren|haven|hasn|hadn)\s*'\s*t\b", re.IGNORECASE), r"\1't"),
    (re.compile(r"\b(c|j|n|m|t|s|d|l|qu)\s*'\s*(\w+)", re.IGNORECASE), r"\1'\2"),
    (re.compile(r"\s+-\s+", re.IGNORECASE), r"-"),
]

# Filler words: e.g. "um ,", "uh ", etc.
FILLER_PATTERNS = [
    re.compile(r"\b(um|uh|erm|ah|hmm)\b\s*[,]*\s*", re.IGNORECASE),
]


def strip_hallucinations(text: str) -> str:
    """Remove known ASR hallucination phrases."""
    cleaned = text
    for pat in HALLUCINATION_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_contractions(text: str) -> str:
    """Normalize broken tokens around apostrophes and hyphens."""
    cleaned = text
    for pat, repl in CONTRACTION_PATTERNS:
        cleaned = pat.sub(repl, cleaned)
    return cleaned


def clean_transcript(text: str, strip_fillers: bool = True) -> str:
    """Full transcript cleaning pipeline."""
    if not text:
        return ""

    cleaned = strip_hallucinations(text)
    if strip_fillers:
        for pat in FILLER_PATTERNS:
            cleaned = pat.sub("", cleaned)

    cleaned = normalize_contractions(cleaned)
    # Fix spacing before punctuation (e.g. "ready ." -> "ready.")
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    # Strip leading/trailing stray punctuation commas, dots, or exclamation marks
    cleaned = re.sub(r"^[\s,;.-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
