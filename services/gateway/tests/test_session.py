"""
services.gateway.tests.test_session
===================================
Unit tests for Gateway Session and TTS deduplication store.
"""

import time
import pytest
from ramo_gateway.session import GatewaySession, SessionStore


def test_session_state_and_speaker_inheritance():
    sess = GatewaySession(session_id="sess_test1")
    assert sess.session_id == "sess_test1"
    assert sess.target_language == "fr"
    assert sess.auto_tts is True

    sess.update_config(target_language="es", auto_tts=False, context_summary="Sprint planning")
    assert sess.target_language == "es"
    assert sess.auto_tts is False
    assert sess.context_summary == "Sprint planning"

    assert sess.get_last_known_speaker() == "Unknown"
    sess.set_last_known_speaker("Alice")
    assert sess.get_last_known_speaker() == "Alice"


def test_session_tts_deduplication():
    sess = GatewaySession(session_id="sess_test2")
    key = "trc_123:Hello team"

    # First attempt: not a duplicate
    assert sess.is_duplicate_tts(key) is False

    # Second attempt immediately: duplicate!
    assert sess.is_duplicate_tts(key) is True


def test_session_store_lifecycle():
    store = SessionStore()
    sess = store.create("sess_abc")
    assert store.count() == 1
    assert store.get("sess_abc") == sess

    store.remove("sess_abc")
    assert store.count() == 0
    assert store.get("sess_abc") is None


def test_session_utterance_advance():
    sess = GatewaySession(session_id="sess_seq_test")
    id1 = sess.get_current_chunk_id()
    assert id1 == "utt_sess_seq_test_0"

    id2 = sess.advance_utterance()
    assert id2 == "utt_sess_seq_test_1"
    assert sess.get_current_chunk_id() == id2

    id3 = sess.advance_utterance()
    assert id3 == "utt_sess_seq_test_2"
    assert sess.get_current_chunk_id() == id3

