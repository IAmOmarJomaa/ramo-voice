"""
ramo_listen.hypothesis_buffer
==============================
Hypothesis boundary deduplicator based on whisper_streaming.
Eliminates boundary word repetition across overlapping audio chunks (1- to 5-gram suppression).
Provides initial_prompt continuation for Faster-Whisper context tracking.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("ramo_listen.hypothesis_buffer")


def _clean_token(t: str) -> str:
    """Strip punctuation and lowercase token for robust comparison."""
    return re.sub(r"[^\w\s]", "", t).lower().strip()


class HypothesisDeduplicator:
    """
    Online hypothesis boundary deduplicator.
    Maintains a rolling window of committed tokens and matches against incoming chunk hypotheses.
    """

    def __init__(self, max_ngram: int = 5):
        self.max_ngram = max_ngram
        self.committed_tokens: List[str] = []
        self.committed_text: str = ""

    def deduplicate(
        self,
        text: str,
        words: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Deduplicates incoming text and word timestamps against recent committed tokens.
        Returns the cleaned (deduplicated_text, deduplicated_words).
        """
        if not text or not text.strip():
            return "", words or []

        new_tokens = text.strip().split()
        clean_new = [_clean_token(w) for w in new_tokens if _clean_token(w)]
        clean_committed = [_clean_token(w) for w in self.committed_tokens if _clean_token(w)]

        cn = len(clean_committed)
        nn = len(clean_new)
        drop_count = 0

        # Check n-grams from max_ngram down to 1
        for i in range(min(min(cn, nn), self.max_ngram), 0, -1):
            c_slice = clean_committed[-i:]
            n_slice = clean_new[:i]
            if c_slice == n_slice:
                drop_count = i
                logger.info(f"✂️ [DEDUP] Stripped {drop_count} overlapping boundary words: {new_tokens[:drop_count]}")
                break

        if drop_count > 0:
            remaining_tokens = new_tokens[drop_count:]
            remaining_text = " ".join(remaining_tokens)
            remaining_words = words[drop_count:] if words and len(words) >= drop_count else []
            self.committed_tokens.extend(remaining_tokens)
            self.committed_text = f"{self.committed_text} {remaining_text}".strip()
            return remaining_text, remaining_words
        else:
            self.committed_tokens.extend(new_tokens)
            self.committed_text = f"{self.committed_text} {text.strip()}".strip()
            return text.strip(), words or []

    def get_initial_prompt(self, max_chars: int = 200) -> str:
        """Returns the last max_chars of committed text for Faster-Whisper context prompt."""
        if not self.committed_text:
            return ""
        if len(self.committed_text) <= max_chars:
            return self.committed_text
        suffix = self.committed_text[-max_chars:]
        space_idx = suffix.find(" ")
        if space_idx != -1:
            return suffix[space_idx + 1 :]
        return suffix

    def reset(self) -> None:
        """Reset buffer state for a new session or speaker transition."""
        self.committed_tokens.clear()
        self.committed_text = ""
