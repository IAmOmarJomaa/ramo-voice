"""
ramo_listen.local_agreement
===========================
LocalAgreement streaming stabilizer extracted from whisper_streaming.
Eliminates hallucination cascades and premature word commits in sliding-window ASR.

A prefix of tokens is only committed once it has appeared identically across
N consecutive window predictions (default N=2).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class AgreementResult:
    committed: str
    tentative: str


class LocalAgreement:
    """
    Online LocalAgreement algorithm.
    Compares consecutive ASR hypotheses and computes stable committed prefixes.
    """

    def __init__(self, n_agreement: int = 2):
        self.n_agreement = n_agreement
        self.hypotheses: List[List[str]] = []
        self.committed_words: List[str] = []

    def step(self, hypothesis: str) -> AgreementResult:
        """
        Process a new window hypothesis and calculate committed vs tentative text.
        """
        words = hypothesis.strip().split()
        if not words:
            return AgreementResult(
                committed=" ".join(self.committed_words),
                tentative=""
            )

        self.hypotheses.append(words)
        if len(self.hypotheses) > self.n_agreement:
            self.hypotheses.pop(0)

        # Check for agreement if we have accumulated enough windows
        if len(self.hypotheses) >= self.n_agreement:
            # Find longest common prefix across all active hypotheses
            min_len = min(len(h) for h in self.hypotheses)
            prefix_len = 0
            for i in range(min_len):
                word_candidates = {h[i] for h in self.hypotheses}
                if len(word_candidates) == 1:
                    prefix_len = i + 1
                else:
                    break

            if prefix_len > len(self.committed_words):
                # We have new words that achieved consensus
                self.committed_words = self.hypotheses[-1][:prefix_len]

        committed_str = " ".join(self.committed_words)
        tentative_words = words[len(self.committed_words):]
        tentative_str = " ".join(tentative_words)

        return AgreementResult(committed=committed_str, tentative=tentative_str)

    def flush(self) -> str:
        """Flush and commit all remaining words in the latest hypothesis."""
        if self.hypotheses:
            all_words = self.hypotheses[-1]
            self.committed_words = all_words
            self.hypotheses.clear()
            return " ".join(all_words)
        res = " ".join(self.committed_words)
        self.committed_words.clear()
        return res
