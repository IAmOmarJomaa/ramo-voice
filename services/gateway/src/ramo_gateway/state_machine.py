"""
ramo_gateway.state_machine
==========================
VAD-driven turn-taking conversation state machine with barge-in interruption.
"""

import enum
import logging

logger = logging.getLogger("ramo_gateway.state_machine")


class SessionState(str, enum.Enum):
    LISTENING = "LISTENING"   # Waiting for user speech
    SPEAKING = "SPEAKING"     # User is actively speaking
    THINKING = "THINKING"     # Silence detected; waiting for LLM/TTS
    PLAYING = "PLAYING"       # Audio is currently playing back to user


class ConversationStateMachine:
    """
    Manages conversational turn-taking and handles barge-in interruptions.
    """

    def __init__(self, silence_timeout_sec: float = 0.3):
        self.silence_timeout_sec = silence_timeout_sec
        self.state = SessionState.LISTENING

    def on_speech_start(self) -> bool:
        """
        Triggered when VAD detects speech onset.
        Returns True if this speech constituted a barge-in (interrupted PLAYING or THINKING).
        """
        was_interrupted = False
        if self.state in [SessionState.PLAYING, SessionState.THINKING]:
            logger.info(f"Barge-in detected! User interrupted from state: {self.state}")
            was_interrupted = True

        self.state = SessionState.SPEAKING
        return was_interrupted

    def on_speech_stop(self) -> None:
        """
        Triggered when silence exceeds silence_timeout_sec.
        """
        if self.state == SessionState.SPEAKING:
            self.state = SessionState.THINKING

    def set_state(self, state: SessionState) -> None:
        """Explicitly transition to a state."""
        self.state = state
