import numpy as np
import pytest
from ramo_voice.chunker import split_text_into_chunks, concatenate_audio_chunks


def test_split_short_text():
    text = "Hello world."
    chunks = split_text_into_chunks(text, max_chars=100)
    assert chunks == ["Hello world."]


def test_abbreviation_preservation():
    text = "Dr. Smith arrived at 5 p.m. to discuss the U.S. economy with Mr. Brown. Everything went smoothly."
    chunks = split_text_into_chunks(text, max_chars=80)
    assert any("Dr. Smith" in c for c in chunks)
    assert any("Mr. Brown" in c for c in chunks)


def test_decimal_preservation():
    text = "The value of pi is approximately 3.14159, which is quite interesting."
    chunks = split_text_into_chunks(text, max_chars=45)
    assert any("3.14159" in c for c in chunks)


def test_tag_preservation():
    text = "This is hilarious [laugh] and I cannot stop smiling <breath> today."
    chunks = split_text_into_chunks(text, max_chars=30)
    for c in chunks:
        if "[" in c:
            assert "[laugh]" in c
        if "<" in c:
            assert "<breath>" in c


def test_cjk_punctuation():
    text = "こんにちは。元気ですか？はい、元気です！"
    chunks = split_text_into_chunks(text, max_chars=12)
    assert len(chunks) >= 2
    assert "こんにちは。" in chunks[0]


def test_concatenate_audio_chunks_crossfade():
    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    chunk1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    chunk2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)

    concatenated = concatenate_audio_chunks([chunk1, chunk2], sample_rate=sr, crossfade_ms=20)
    expected_overlap = int(sr * 20 / 1000)
    expected_length = len(chunk1) + len(chunk2) - expected_overlap
    assert len(concatenated) == expected_length
    assert not np.isnan(concatenated).any()
