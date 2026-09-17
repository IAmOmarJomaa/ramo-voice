"""
ramo_gateway.session
====================
Session state tracking and duplicate TTS trigger suppression for Bridge-Tauri connections.
"""

import time
import threading
from typing import Dict, Optional, Tuple, List, Any


class GatewaySession:
    """
    State manager for a single active WebSocket client connection.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.target_language = "fr"
        self.auto_tts = True
        self.source = "mic"
        self.context_summary = ""
        self.studio_id = "default_studio"
        self.meeting_id = ""
        self.tts_engine = "kokoro"
        self.agent_mode = False
        self._last_known_speaker = "Unknown"
        self.last_ping = time.time()
        self._recent_tts: Dict[str, float] = {}
        self.dialogue_history: list = []
        self.action_items: list = []
        self.chunk_seq: int = 0
        self.utterance_seq: int = 0
        self.current_revision: int = 0
        self.last_final_transcript: str = ""
        self.last_final_time: float = 0.0
        self.dynamic_glossary: dict = {}
        self.deduplicator = None
        self._lock = threading.Lock()

    def get_current_utterance_id(self) -> str:
        with self._lock:
            return f"utt_{self.session_id}_{self.utterance_seq}"

    def get_current_chunk_id(self) -> str:
        """Alias for backward compatibility with Bridge-Tauri chunk_id."""
        with self._lock:
            return f"utt_{self.session_id}_{self.utterance_seq}"

    def next_revision(self) -> int:
        with self._lock:
            self.current_revision += 1
            return self.current_revision

    def advance_utterance(self) -> str:
        with self._lock:
            self.utterance_seq += 1
            self.chunk_seq += 1
            self.current_revision = 0
            return f"utt_{self.session_id}_{self.utterance_seq}"

    def advance_chunk_seq(self) -> str:
        """Alias for advance_utterance for backward compatibility."""
        return self.advance_utterance()

    def ping(self) -> None:
        self.last_ping = time.time()

    def update_config(
        self,
        target_language: Optional[str] = None,
        auto_tts: Optional[bool] = None,
        source: Optional[str] = None,
        context_summary: Optional[str] = None,
        studio_id: Optional[str] = None,
        meeting_id: Optional[str] = None,
        tts_engine: Optional[str] = None,
    ) -> None:
        with self._lock:
            if target_language is not None:
                self.target_language = target_language
            if auto_tts is not None:
                self.auto_tts = auto_tts
            if source is not None:
                self.source = source
            if context_summary is not None:
                self.context_summary = context_summary
            if studio_id is not None:
                self.studio_id = studio_id
            if meeting_id is not None:
                self.meeting_id = meeting_id
            if tts_engine is not None:
                self.tts_engine = tts_engine

    def set_agent_mode(self, enabled: bool) -> None:
        with self._lock:
            self.agent_mode = enabled

    def get_last_known_speaker(self) -> str:
        with self._lock:
            return self._last_known_speaker

    def set_last_known_speaker(self, speaker: str) -> None:
        with self._lock:
            if speaker and speaker.upper() != "UNKNOWN":
                self._last_known_speaker = speaker

    def is_duplicate_tts(self, key: str, window_sec: float = 1.5) -> bool:
        """
        Suppresses duplicate synthesis requests arriving within `window_sec`.
        """
        with self._lock:
            now = time.time()
            # Prune entries older than 3 seconds
            dead_keys = [k for k, ts in self._recent_tts.items() if now - ts > 3.0]
            for k in dead_keys:
                del self._recent_tts[k]

            last = self._recent_tts.get(key)
            if last is not None and (now - last) < window_sec:
                return True

            self._recent_tts[key] = now
            return False


class SessionStore:
    """
    Thread-safe registry of active GatewaySession instances.
    """

    def __init__(self):
        self._sessions: Dict[str, GatewaySession] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str) -> GatewaySession:
        with self._lock:
            sess = GatewaySession(session_id)
            self._sessions[session_id] = sess
            return sess

    def get(self, session_id: str) -> Optional[GatewaySession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
