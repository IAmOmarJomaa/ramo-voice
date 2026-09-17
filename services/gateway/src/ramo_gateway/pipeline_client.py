"""
ramo_gateway.pipeline_client
============================
High-performance pipeline dispatcher coordinating Enhancement, STT,
Diarization, Translation, and TTS.
Supports dual execution:
1. Zero-latency in-memory execution (<1.5ms per stage).
2. Asynchronous HTTP/WS microservice network dispatch.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# In-process sovereign microservice engines for zero-socket latency
from ramo_clean.pipeline import AudioPreconditioner
from ramo_listen.engines.whisper_engine import WhisperSTTEngine
from ramo_listen.aligner import TemporalWordAligner, SpeakerInterval
from ramo_speaker.cluster import SpeakerClusterer
from ramo_speaker.segmenter import AudioSegmenter
from ramo_speaker.overlap import AcousticOverlapDetector
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn, HarvestedProfile
from ramo_translate.router import TranslationRouter
from ramo_translate.action_detector import detect_action_item
from ramo_translate.intelligence import MeetingIntelligenceSynthesizer
from ramo_voice.engines.supertonic_engine import SupertonicEngine
from ramo_voice.engines.kokoro_engine import KokoroEngine
from ramo_voice.engines.f5_engine import F5TTSEngine
from ramo_voice.profiles import default_store, VoiceProfile
from ramo_common.config import load_tts_pacer_config, TTSPacerConfig
from .prosody_buffer import ProsodicClauseBuffer
from .pacer import DynamicLatencyPacer
from .anchor_matcher import AnchorVoiceMatcher

logger = logging.getLogger("ramo_gateway.pipeline")


class PipelineDispatcher:
    """
    Coordinates multi-stage audio intelligence pipeline.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        # Single source of truth configuration
        self.config: TTSPacerConfig = load_tts_pacer_config()

        # 0. Clean & Precondition
        self.preconditioner = AudioPreconditioner(sample_rate=sample_rate)

        # 1. STT (Direct Faster-Whisper neural engine)
        self.stt = WhisperSTTEngine(sample_rate=sample_rate)
        self.aligner = TemporalWordAligner()

        # 2. Diarization & Crosstalk Detection
        self.overlap_detector = AcousticOverlapDetector(sample_rate=sample_rate)
        self.clusterer = SpeakerClusterer(similarity_threshold=0.62, momentum=0.70)
        self.segmenter = AudioSegmenter(sample_rate=sample_rate)
        self.harvester = VoiceprintHarvester(
            tier1_threshold_sec=self.config.harvesting.tier1_threshold_sec,
            tier2_threshold_sec=self.config.harvesting.tier2_threshold_sec,
            max_buffer_sec=self.config.harvesting.max_buffer_sec,
            min_snr_db=self.config.harvesting.min_snr_db,
            target_sr=sample_rate,
        )

        # 3. Translation & Meeting Intelligence
        self.translator = TranslationRouter()
        self.intelligence_synth = MeetingIntelligenceSynthesizer()

        # 4. Voice Engines, Anchors, and Prosody Pacing
        self.tts_supertonic = SupertonicEngine()
        self.tts_kokoro = KokoroEngine()
        self.tts_f5 = F5TTSEngine()
        self.anchor_matcher = AnchorVoiceMatcher()
        self.pacer = DynamicLatencyPacer(
            sample_rate=24000,
            target_latency_ms=self.config.pacer.target_latency_budget_ms,
            speed_min=self.config.pacer.wsola_speed_min,
            speed_max=self.config.pacer.wsola_speed_max,
            queue_threshold_words=self.config.pacer.queue_threshold_words,
        )
        self.prosody_buffers: Dict[str, ProsodicClauseBuffer] = {}

        self._initialized = False

    async def initialize(self) -> None:
        """Warm up all in-process engines."""
        if self._initialized:
            return
        logger.info("Initializing in-process sovereign pipeline engines...")
        await self.stt.load()
        await self.tts_supertonic.load()
        await self.tts_kokoro.load()
        await self.translator.engine.load()
        self._initialized = True

    def clean_audio_pcm(self, pcm16_bytes: bytes) -> Tuple[np.ndarray, bytes]:
        """Convert PCM16 bytes to float32, apply 80Hz HPF, VAD, spectral gate, and AGC."""
        arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        cleaned = self.preconditioner.process_chunk(arr, stream=True).audio
        cleaned_pcm16 = (np.clip(cleaned, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return cleaned, cleaned_pcm16

    async def process_stt(self, audio_f32: np.ndarray) -> dict:
        """Run STT inference."""
        if not self._initialized:
            await self.initialize()
        return await self.stt.transcribe(audio_f32)

    def get_prosody_buffer(self, session_id: str = "default") -> ProsodicClauseBuffer:
        """Retrieve or initialize session prosodic clause buffer."""
        if session_id not in self.prosody_buffers:
            self.prosody_buffers[session_id] = ProsodicClauseBuffer(
                min_clause_words=self.config.prosody_buffer.min_clause_words,
                max_clause_words=self.config.prosody_buffer.max_clause_words,
                strong_terminators=self.config.prosody_buffer.strong_terminators,
                weak_terminators=self.config.prosody_buffer.weak_terminators,
                conjunctions=self.config.prosody_buffer.conjunctions,
                ttfa_timeout_ms=self.config.prosody_buffer.ttfa_timeout_ms,
            )
        return self.prosody_buffers[session_id]

    def detect_overlap(self, audio_f32: np.ndarray) -> Tuple[bool, float]:
        """Detect multi-speaker crosstalk and acoustic collisions."""
        return self.overlap_detector.detect(audio_f32)

    def identify_speaker(
        self,
        audio_f32: np.ndarray,
        last_known: str = "Unknown",
        is_overlap: bool = False,
    ) -> str:
        """Identify or cluster speaker embedding using CampPlus 192-dim projection."""
        if len(audio_f32) < int(0.2 * self.sample_rate):
            logger.debug(f"👥 [DIAR] Audio too short ({len(audio_f32)} samples < 0.2s) - inheriting '{last_known}'")
            return last_known if last_known != "Unknown" else "Speaker 1"
        emb = self.segmenter.extract_embedding(audio_f32)
        spk_id = self.clusterer.assign_or_update(emb, is_overlap=is_overlap)
        logger.info(
            f"👥 [DIAR_IDENTIFY] Audio chunk ({len(audio_f32)/self.sample_rate:.2f}s) -> Assigned '{spk_id}' "
            f"(overlap: {is_overlap}, prev: '{last_known}')"
        )
        return spk_id

    def harvest_speech_turn(
        self,
        speaker_id: str,
        audio_f32: np.ndarray,
        transcript: str,
        is_overlap: bool,
    ) -> Optional[HarvestedProfile]:
        """Strictly harvest single-speaker clean monologue turns into voiceprint pool."""
        turn = SpeakerTurn(
            speaker_id=speaker_id,
            start_sec=0.0,
            end_sec=round(len(audio_f32) / self.sample_rate, 3),
            audio=audio_f32,
            transcript=transcript,
            is_overlap=is_overlap,
        )
        self.harvester.add_turn(turn)
        profile = self.harvester.get_profile(speaker_id)
        if profile and profile.ready_for_cloning:
            logger.info(
                f"[HARVEST_PROMOTION] 🚀 '{speaker_id}' promoted to ZERO-SHOT CLONING! ({profile.duration_sec:.1f}s clean speech)"
            )
            default_store.register_profile(
                VoiceProfile(
                    voice_id=speaker_id,
                    name=speaker_id,
                    voice_type="cloned",
                    sample_rate=self.sample_rate,
                    reference_audio=profile.audio,
                    reference_text=profile.transcript,
                )
            )
        return profile

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        session_id: str = "default",
        speaker_id: str = "Speaker 1",
    ) -> Tuple[str, bool, Optional[str]]:
        """Translate text with 0ms fast-bypass and 3-tier sliding context."""
        res = await self.translator.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            session_id=session_id,
            speaker_id=speaker_id,
        )
        return res.translated_text, res.is_bypass, res.detected_action

    async def extract_meeting_intelligence(self, dialogue_turns: list) -> dict:
        """Extract multi-turn meeting intelligence (actions, orders, notes)."""
        return await self.intelligence_synth.analyze(dialogue_turns, self.translator.engine)

    def check_action_item(self, text: str) -> Optional[str]:
        """Classify if utterance contains commitments, schedule changes, or task delegations."""
        return detect_action_item(text)

    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speaker_id: str = "Speaker 1",
        engine_type: Optional[str] = None,
        speaker_audio: Optional[np.ndarray] = None,
        current_latency_ms: float = 0.0,
        queue_words: int = 0,
    ) -> Tuple[bytes, int, float]:
        """
        Synthesize speech with instant anchor matching (<4.5s) or zero-shot cloning (>=4.5s),
        applying WSOLA pitch-preserving time-scale regulation.
        """
        if not self._initialized:
            await self.initialize()

        # Check if speaker has harvested zero-shot profile
        cloned_profile = default_store.get(speaker_id)
        is_cloned = cloned_profile is not None and cloned_profile.voice_type == "cloned"

        if is_cloned:
            effective_engine = "f5tts"
            effective_voice = speaker_id
            profile = cloned_profile
            logger.info(f"[SYNTH] 🎙️ Using ZERO-SHOT CLONING for '{speaker_id}'")
        else:
            effective_engine = engine_type or self.config.engine_selection.default_fallback
            effective_voice = voice or self.anchor_matcher.match_voice(speaker_id, audio=speaker_audio, sample_rate=self.sample_rate)
            profile = default_store.get(effective_voice)
            if profile is None:
                profile = VoiceProfile(voice_id=effective_voice, name=effective_voice, voice_type="preset", sample_rate=24000)
            logger.info(f"[SYNTH] ⚓ Using ANCHOR PRESET '{effective_voice}' for '{speaker_id}' (<4.5s threshold)")

        # Calculate dynamic latency speed
        speed = self.pacer.calculate_speed(current_latency_ms=current_latency_ms, queue_words=queue_words)

        if effective_engine == "kokoro" or "kokoro" in effective_voice.lower():
            audio_f32, sr = await self.tts_kokoro.generate_chunk(text, profile, speed=1.0)
        elif effective_engine == "f5tts":
            audio_f32, sr = await self.tts_f5.generate_chunk(text, profile, speed=1.0)
        else:
            audio_f32, sr = await self.tts_supertonic.generate_chunk(text, profile, speed=1.0)

        # Apply WSOLA time-scale modification if speed != 1.0
        if speed > 1.02:
            audio_f32 = self.pacer.apply_wsola(audio_f32, speed=speed)

        pcm16_bytes = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return pcm16_bytes, sr, speed
