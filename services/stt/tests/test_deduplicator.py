import pytest
from ramo_listen.hypothesis_buffer import HypothesisDeduplicator


def test_deduplicator_removes_boundary_ngram_overlap():
    dedup = HypothesisDeduplicator(max_ngram=5)

    # Chunk 1
    t1, w1 = dedup.deduplicate("So it is the second of October and this is our data science sync.")
    assert t1 == "So it is the second of October and this is our data science sync."

    # Chunk 2 has 3-word overlap ("data science sync")
    t2, w2 = dedup.deduplicate("data science sync and we have a ton of items to discuss.")
    assert t2 == "and we have a ton of items to discuss."


def test_deduplicator_single_word_overlap():
    dedup = HypothesisDeduplicator(max_ngram=5)

    t1, _ = dedup.deduplicate("Are you ready to go")
    assert t1 == "Are you ready to go"

    t2, _ = dedup.deduplicate("go ahead with the presentation.")
    assert t2 == "ahead with the presentation."


def test_deduplicator_prompt_generation():
    dedup = HypothesisDeduplicator()
    dedup.deduplicate("Hello everyone welcome to the meeting.")
    prompt = dedup.get_initial_prompt(max_chars=50)
    assert len(prompt) <= 50
    assert "meeting" in prompt
