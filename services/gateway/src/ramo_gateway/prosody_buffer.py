"""
ramo_gateway.prosody_buffer
===========================
Prosodic Clause Buffer: segments streaming translated tokens into natural breath clauses.
Combines strong/weak punctuation, conjunction hinging, and a Time-To-First-Audio (TTFA)
timeout to eliminate robotic word-by-word fragmentation while ensuring ultra-low latency.
"""

from __future__ import annotations

import re
import time
import logging
from typing import List, Optional

logger = logging.getLogger("ramo_gateway.prosody_buffer")


class ProsodicClauseBuffer:
    """
    Accumulates streaming translation text and flushes cohesive acoustic clauses.
    """

    def __init__(
        self,
        min_clause_words: int = 4,
        max_clause_words: int = 14,
        strong_terminators: Optional[List[str]] = None,
        weak_terminators: Optional[List[str]] = None,
        conjunctions: Optional[List[str]] = None,
        ttfa_timeout_ms: int = 600,
    ):
        self.min_clause_words = min_clause_words
        self.max_clause_words = max_clause_words
        self.strong_terminators = strong_terminators or [".", "!", "?", "。", "！", "？"]
        self.weak_terminators = weak_terminators or [",", ";", ":", "，", "；"]
        self.conjunctions = [
            c.lower()
            for c in (
                conjunctions
                or [
                    "and", "but", "because", "although", "or", "however",
                    "et", "mais", "parce que", "porque"
                ]
            )
        ]
        self.ttfa_timeout_ms = ttfa_timeout_ms

        self._buffer: str = ""
        self._last_token_time: float = time.time()

    def add_text(self, text: str) -> List[str]:
        """Ingest text token/chunk and emit any completed acoustic clauses."""
        if not text:
            return []

        self._buffer += text
        self._last_token_time = time.time()
        return self._extract_clauses()

    def flush(self) -> List[str]:
        """Manually flush any remaining text in the buffer."""
        pending = self._buffer.strip()
        self._buffer = ""
        if pending:
            logger.debug(f"[PROSODY_BUFFER] Manual flush: '{pending}'")
            return [pending]
        return []

    def check_timeout(self) -> List[str]:
        """Force flush if TTFA timeout elapsed with unfinalized text in buffer."""
        if not self._buffer.strip():
            return []

        elapsed_ms = (time.time() - self._last_token_time) * 1000.0
        if elapsed_ms >= self.ttfa_timeout_ms:
            clause = self._buffer.strip()
            self._buffer = ""
            logger.info(f"[PROSODY_BUFFER] ⏱️ TTFA timeout ({elapsed_ms:.0f}ms >= {self.ttfa_timeout_ms}ms) flush: '{clause}'")
            return [clause]
        return []

    def get_pending_text(self) -> str:
        """Return pending buffer text without flushing."""
        return self._buffer.strip()

    def _extract_clauses(self) -> List[str]:
        ready_clauses: List[str] = []

        while self._buffer:
            buf = self._buffer
            words = buf.strip().split()
            n_words = len(words)

            if n_words == 0:
                self._buffer = ""
                break

            # 1. Check strong terminators (. ! ? etc)
            cut_idx = -1
            for term in self.strong_terminators:
                pos = buf.find(term)
                if pos != -1 and (cut_idx == -1 or pos < cut_idx):
                    cut_idx = pos + len(term)

            if cut_idx != -1:
                clause = buf[:cut_idx].strip()
                self._buffer = buf[cut_idx:].lstrip()
                if clause:
                    ready_clauses.append(clause)
                continue

            # 2. Check weak terminators (, ; :) only if >= min_clause_words
            cut_idx = -1
            for term in self.weak_terminators:
                # Find all occurrences of the weak terminator in the buffer
                start_search = 0
                while True:
                    pos = buf.find(term, start_search)
                    if pos == -1:
                        break
                    preceding = buf[:pos].strip().split()
                    if len(preceding) >= self.min_clause_words:
                        if cut_idx == -1 or (pos + len(term)) < cut_idx:
                            cut_idx = pos + len(term)
                            break
                    start_search = pos + len(term)

            if cut_idx != -1:
                clause = buf[:cut_idx].strip()
                self._buffer = buf[cut_idx:].lstrip()
                if clause:
                    ready_clauses.append(clause)
                continue

            # 3. Check conjunction hinging (e.g. 'and', 'but', 'et', 'mais') if words >= min_clause_words + 2
            if n_words >= (self.min_clause_words + 2):
                conj_cut = -1
                for conj in self.conjunctions:
                    pattern = rf"(?<=\s){re.escape(conj)}(?=\s)"
                    matches = list(re.finditer(pattern, buf, re.IGNORECASE))
                    for m in matches:
                        preceding = buf[: m.start()].strip().split()
                        if len(preceding) >= self.min_clause_words:
                            conj_cut = m.start()
                            break
                    if conj_cut != -1:
                        break

                if conj_cut != -1:
                    clause = buf[:conj_cut].strip()
                    self._buffer = buf[conj_cut:].lstrip()
                    if clause:
                        ready_clauses.append(clause)
                    continue

            # 4. Force cut if buffer exceeds max_clause_words
            if n_words >= self.max_clause_words:
                split_point = " ".join(words[: self.max_clause_words])
                cut_len = buf.find(words[self.max_clause_words - 1]) + len(words[self.max_clause_words - 1])
                clause = buf[:cut_len].strip()
                self._buffer = buf[cut_len:].lstrip()
                ready_clauses.append(clause)
                continue

            # No clause ready yet; keep buffering
            break

        return ready_clauses
