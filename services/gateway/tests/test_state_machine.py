"""
Tests for ConversationStateMachine in ramo_gateway.state_machine.
Verifies VAD transitions and barge-in interruption logic.
"""

import pytest
from ramo_gateway.state_machine import ConversationStateMachine, SessionState


def test_initial_state():
    sm = ConversationStateMachine()
    assert sm.state == SessionState.LISTENING


def test_vad_speech_start_and_stop():
    sm = ConversationStateMachine(silence_timeout_sec=0.3)

    # User begins speaking
    sm.on_speech_start()
    assert sm.state == SessionState.SPEAKING

    # User pauses for silence_timeout
    sm.on_speech_stop()
    assert sm.state == SessionState.THINKING


def test_barge_in_interruption():
    sm = ConversationStateMachine()

    # Audio is playing back to user
    sm.set_state(SessionState.PLAYING)
    assert sm.state == SessionState.PLAYING

    # User interrupts by speaking
    interrupted = sm.on_speech_start()
    assert interrupted is True
    assert sm.state == SessionState.SPEAKING
