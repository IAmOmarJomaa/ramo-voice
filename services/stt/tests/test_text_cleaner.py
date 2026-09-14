import pytest
from ramo_listen.text_cleaner import clean_transcript, normalize_contractions, strip_hallucinations


def test_strip_hallucinations():
    assert strip_hallucinations("Thank you for watching!") == ""
    assert strip_hallucinations("Subtitles by the Amara.org community") == ""
    assert strip_hallucinations("Please subscribe to my channel") == ""
    assert strip_hallucinations("We agreed on the timeline. Thank you for watching.") == "We agreed on the timeline."


def test_normalize_contractions():
    assert normalize_contractions("it 's a good plan") == "it's a good plan"
    assert normalize_contractions("don 't worry") == "don't worry"
    assert normalize_contractions("c 'est la vie") == "c'est la vie"
    assert normalize_contractions("qu 'est - ce que c 'est") == "qu'est-ce que c'est"


def test_clean_transcript_end_to_end():
    raw = "um , it 's ready . Thank you for watching !"
    cleaned = clean_transcript(raw, strip_fillers=True)
    assert cleaned == "it's ready."
