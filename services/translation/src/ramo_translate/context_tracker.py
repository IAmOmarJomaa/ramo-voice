"""
ramo_translate.context_tracker
==============================
3-Tier Context Architecture with Speaker Symbol Masking.
Preserves multi-turn dialogue context and cross-turn pronoun references
while preventing speaker identity hallucination.
"""

from typing import Dict, List, Optional
import threading


class MeetingContextTracker:
    """
    Manages session-level conversational history and anonymous speaker symbol assignment.
    """

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self._lock = threading.Lock()
        # session_id -> list of "[S_A]: utterance"
        self._session_windows: Dict[str, List[str]] = {}
        # session_id -> { speaker_id: "[S_A]" }
        self._speaker_maps: Dict[str, Dict[str, str]] = {}
        # session_id -> meeting agenda / summary context
        self._meeting_contexts: Dict[str, str] = {}

    def get_speaker_symbol(self, session_id: str, speaker_id: str) -> str:
        """Return anonymous token for speaker (e.g. '[S_A]', '[S_B]')."""
        with self._lock:
            if session_id not in self._speaker_maps:
                self._speaker_maps[session_id] = {}
            spk_map = self._speaker_maps[session_id]
            if speaker_id not in spk_map:
                letter = chr(65 + len(spk_map))  # A, B, C...
                spk_map[speaker_id] = f"[S_{letter}]"
            return spk_map[speaker_id]

    def set_meeting_context(self, session_id: str, context: str) -> None:
        """Set high-level meeting agenda/topic context."""
        with self._lock:
            self._meeting_contexts[session_id] = context.strip()

    def get_meeting_context(self, session_id: str) -> str:
        with self._lock:
            return self._meeting_contexts.get(session_id, "")

    def add_utterance(self, session_id: str, speaker_id: str, text: str) -> None:
        """Add spoken turn to sliding dialogue window."""
        clean = text.strip()
        if not clean:
            return

        symbol = self.get_speaker_symbol(session_id, speaker_id)
        with self._lock:
            if session_id not in self._session_windows:
                self._session_windows[session_id] = []
            self._session_windows[session_id].append(f"{symbol}: {clean}")
            # Clamp to window_size
            if len(self._session_windows[session_id]) > self.window_size:
                self._session_windows[session_id] = self._session_windows[session_id][-self.window_size:]

    def get_sliding_window(self, session_id: str) -> List[str]:
        """Return snapshot of current sliding window for session."""
        with self._lock:
            return list(self._session_windows.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._session_windows.pop(session_id, None)
            self._speaker_maps.pop(session_id, None)
            self._meeting_contexts.pop(session_id, None)
