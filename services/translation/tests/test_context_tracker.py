import pytest
from ramo_translate.context_tracker import MeetingContextTracker


def test_speaker_symbol_masking():
    tracker = MeetingContextTracker()
    s1 = tracker.get_speaker_symbol(session_id="meet_1", speaker_id="SPEAKER_00")
    s2 = tracker.get_speaker_symbol(session_id="meet_1", speaker_id="SPEAKER_01")
    s1_repeat = tracker.get_speaker_symbol(session_id="meet_1", speaker_id="SPEAKER_00")

    assert s1 == "[S_A]"
    assert s2 == "[S_B]"
    assert s1_repeat == "[S_A]"


def test_sliding_window_buffer_clamping():
    tracker = MeetingContextTracker(window_size=4)
    session = "meet_test"

    for i in range(6):
        tracker.add_utterance(session, speaker_id="SPEAKER_00", text=f"Sentence {i}")

    window = tracker.get_sliding_window(session)
    assert len(window) == 4
    assert window[0] == "[S_A]: Sentence 2"
    assert window[-1] == "[S_A]: Sentence 5"
