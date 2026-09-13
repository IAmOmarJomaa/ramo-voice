"""
Tests for LocalAgreement streaming stabilizer in ramo_listen.local_agreement.
Harvested from whisper_streaming to prevent phantom hallucinations.
"""

import pytest
from ramo_listen.local_agreement import LocalAgreement


def test_local_agreement_agreement_reached():
    la = LocalAgreement(n_agreement=2)

    # Iteration 1: initial hypothesis
    res1 = la.step("hello world this is")
    # Nothing committed yet because agreement count is 1
    assert res1.committed == ""
    assert res1.tentative == "hello world this is"

    # Iteration 2: second hypothesis agreeing on prefix
    res2 = la.step("hello world this is a test")
    # "hello world this is" has appeared in 2 consecutive steps -> committed!
    assert res2.committed == "hello world this is"
    assert res2.tentative == "a test"


def test_local_agreement_hypotheses_divergence():
    la = LocalAgreement(n_agreement=2)

    la.step("the quick brown fox")
    # Hypotheses diverges completely due to acoustic noise
    res = la.step("a fast white cat")
    
    # Should not commit conflicting words prematurely
    assert res.committed == ""
    assert res.tentative == "a fast white cat"


def test_local_agreement_flush():
    la = LocalAgreement(n_agreement=2)
    la.step("sentence concluding now")
    final_text = la.flush()
    assert final_text == "sentence concluding now"
