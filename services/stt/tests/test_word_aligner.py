"""
services/stt/tests/test_word_aligner.py
=======================================
TDD tests for TemporalWordAligner: mapping Whisper word timestamps to speaker intervals.
"""

import pytest
from ramo_listen.aligner import TemporalWordAligner, SpeakerInterval


def test_single_speaker_alignment():
    aligner = TemporalWordAligner()

    words = [
        {"word": "Good", "start": 0.10, "end": 0.35, "confidence": 0.95},
        {"word": "morning", "start": 0.38, "end": 0.80, "confidence": 0.98},
        {"word": "everyone", "start": 0.85, "end": 1.30, "confidence": 0.92},
    ]

    intervals = [
        SpeakerInterval(speaker_id="SPEAKER_01", start_sec=0.0, end_sec=1.5)
    ]

    segments = aligner.align_words_to_speakers(words, intervals)
    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_01"
    assert segments[0].text == "Good morning everyone"
    assert len(segments[0].words) == 3
    assert segments[0].start_sec == 0.10
    assert segments[0].end_sec == 1.30


def test_multi_speaker_midpoint_boundary_alignment():
    aligner = TemporalWordAligner()

    # Speaker 1 speaks "I think we should proceed" from 0.0s to 1.5s
    # Speaker 2 interrupts "Wait a minute" from 1.5s to 2.5s
    words = [
        {"word": "I", "start": 0.10, "end": 0.25, "confidence": 0.95},
        {"word": "think", "start": 0.28, "end": 0.50, "confidence": 0.96},
        {"word": "we", "start": 0.52, "end": 0.70, "confidence": 0.97},
        {"word": "should", "start": 0.72, "end": 1.10, "confidence": 0.94},
        {"word": "proceed", "start": 1.12, "end": 1.48, "confidence": 0.93},
        {"word": "Wait", "start": 1.55, "end": 1.80, "confidence": 0.91},
        {"word": "a", "start": 1.82, "end": 1.95, "confidence": 0.96},
        {"word": "minute", "start": 1.98, "end": 2.40, "confidence": 0.98},
    ]

    intervals = [
        SpeakerInterval(speaker_id="SPEAKER_01", start_sec=0.0, end_sec=1.5),
        SpeakerInterval(speaker_id="SPEAKER_02", start_sec=1.5, end_sec=2.6),
    ]

    segments = aligner.align_words_to_speakers(words, intervals)
    assert len(segments) == 2
    assert segments[0].speaker_id == "SPEAKER_01"
    assert segments[0].text == "I think we should proceed"
    assert len(segments[0].words) == 5

    assert segments[1].speaker_id == "SPEAKER_02"
    assert segments[1].text == "Wait a minute"
    assert len(segments[1].words) == 3


def test_word_straddling_boundary():
    """If a word straddles the 1.5s boundary (e.g. 1.40s to 1.65s), its midpoint decides."""
    aligner = TemporalWordAligner()

    # Word midpoint = (1.40 + 1.56) / 2 = 1.48s (< 1.50s) -> belongs to SPEAKER_01
    words = [
        {"word": "now", "start": 1.40, "end": 1.56, "confidence": 0.90},
    ]
    intervals = [
        SpeakerInterval(speaker_id="SPEAKER_01", start_sec=0.0, end_sec=1.5),
        SpeakerInterval(speaker_id="SPEAKER_02", start_sec=1.5, end_sec=3.0),
    ]

    segments = aligner.align_words_to_speakers(words, intervals)
    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_01"
