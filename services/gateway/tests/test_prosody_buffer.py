"""
services/gateway/tests/test_prosody_buffer.py
=============================================
TDD tests for ProsodicClauseBuffer: clause boundary splitting, conjunction hinging,
and TTFA timeout flushing.
"""

import time
import pytest
from ramo_gateway.prosody_buffer import ProsodicClauseBuffer


def test_strong_terminator_flushing():
    buf = ProsodicClauseBuffer(min_clause_words=4, max_clause_words=14)

    # Adding words without terminator -> buffered, returns empty
    clauses = buf.add_text("We need to discuss the budget")
    assert clauses == []

    # Adding period terminator -> immediate flush
    clauses = buf.add_text(" for the upcoming quarter.")
    assert len(clauses) == 1
    assert clauses[0] == "We need to discuss the budget for the upcoming quarter."


def test_weak_terminator_with_min_words():
    buf = ProsodicClauseBuffer(min_clause_words=4, max_clause_words=14)

    # Weak terminator with fewer than min_clause_words (only 2 words) -> NOT flushed yet
    clauses = buf.add_text("In fact,")
    assert clauses == []

    # Add more words up to weak terminator -> flushed
    clauses = buf.add_text(" as we observed yesterday,")
    assert len(clauses) == 1
    assert clauses[0] == "In fact, as we observed yesterday,"


def test_conjunction_hinging():
    buf = ProsodicClauseBuffer(min_clause_words=4, max_clause_words=14)

    # Adding 7 words ending before conjunction
    clauses = buf.add_text("The initial results look very promising indeed")
    assert clauses == []

    # Conjunction 'and' hinges the clause
    clauses = buf.add_text(" and we will review them.")
    assert len(clauses) >= 1
    assert "The initial results look very promising indeed" in clauses[0]


def test_manual_flush():
    buf = ProsodicClauseBuffer(min_clause_words=4)
    buf.add_text("A short utterance")
    flushed = buf.flush()
    assert flushed == ["A short utterance"]
    # Subsequent flush should be empty
    assert buf.flush() == []


def test_ttfa_timeout_flush():
    buf = ProsodicClauseBuffer(min_clause_words=4, ttfa_timeout_ms=100)
    buf.add_text("Incomplete clause without any punctuation")
    assert buf.get_pending_text() != ""

    # Wait 150ms to exceed 100ms timeout
    time.sleep(0.15)
    timed_out_clauses = buf.check_timeout()
    assert len(timed_out_clauses) == 1
    assert timed_out_clauses[0] == "Incomplete clause without any punctuation"
    assert buf.get_pending_text() == ""
