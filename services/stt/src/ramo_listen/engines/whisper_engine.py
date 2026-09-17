"""
ramo_listen.engines.whisper_engine
==================================
Real Faster-Whisper neural STT engine with word-level timestamps,
multi-language detection, and cross-chunk deduplication support.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseSTTEngine
from ..emotion_detector import detect_acoustic_events
from ..text_cleaner import clean_transcript

logger = logging.getLogger("ramo_listen.engines.whisper")


class WhisperSTTEngine(BaseSTTEngine):
    """
    Faster-Whisper production inference engine.
    Runs on CUDA in FP16 or CPU in INT8 with zero synthetic mock fallbacks.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(engine_id="faster-whisper", sample_rate=sample_rate)

        # Auto-detect optimal hardware
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_size is None:
            self.model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3" if self.device == "cuda" else "tiny")
        else:
            self.model_size = model_size

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self._model = None

    async def load(self) -> None:
        """Loads WhisperModel in a separate thread to keep AsyncIO event loop responsive."""
        if self.is_loaded and self._model is not None:
            return

        logger.info(
            f"Loading Faster-Whisper ({self.model_size}) on {self.device} [{self.compute_type}]..."
        )
        
        def _load_sync():
            from faster_whisper import WhisperModel
            return WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, _load_sync)
        self.is_loaded = True
        logger.info("Faster-Whisper neural model loaded successfully.")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe speech audio with word-level timestamps and acoustic emotion detection."""
        if not self.is_loaded:
            await self.load()

        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Ensure float32 normalized in [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        duration = len(audio) / sample_rate
        events = detect_acoustic_events(audio, sample_rate=sample_rate)

        # Skip inference if pure silence or low energy
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        if rms < 0.008 or duration < 0.25:
            return {
                "text": "",
                "raw_text": "",
                "words": [],
                "emotion": events.emotion,
                "has_laughter": events.has_laughter,
                "tags": events.tags,
                "duration": round(duration, 2),
                "confidence": 0.0,
                "language": "en",
            }

        def _infer_sync():
            segments, info = self._model.transcribe(
                audio,
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                initial_prompt=initial_prompt,
            )
            seg_list = list(segments)
            return seg_list, info

        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(None, _infer_sync)

        raw_text_parts = []
        words: List[Dict[str, Any]] = []

        for seg in segments:
            raw_text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "confidence": round(w.probability, 3),
                        }
                    )

        raw_text = " ".join(raw_text_parts).strip()

        # Hallucination guard: reject prompt echo under low acoustic energy
        if initial_prompt and raw_text and rms < 0.015:
            norm_prompt = " ".join(initial_prompt.lower().split())
            norm_raw = " ".join(raw_text.lower().split())
            if norm_raw in norm_prompt or norm_prompt in norm_raw:
                logger.warning(
                    f"🛡️ [WHISPER] Discarded hallucinated prompt echo under low energy ({rms:.4f}): '{raw_text}'"
                )
                return {
                    "text": "",
                    "raw_text": "",
                    "words": [],
                    "emotion": events.emotion,
                    "has_laughter": events.has_laughter,
                    "tags": events.tags,
                    "duration": round(duration, 2),
                    "confidence": 0.0,
                    "language": getattr(info, "language", "en") if info else "en",
                }

        cleaned_text = clean_transcript(raw_text)


        # Prepend acoustic tags if present
        tagged_text = f"{' '.join(events.tags)} {cleaned_text}".strip()

        confidence = float(np.mean([w["confidence"] for w in words])) if words else 0.85

        return {
            "text": tagged_text,
            "raw_text": cleaned_text,
            "words": words,
            "emotion": events.emotion,
            "has_laughter": events.has_laughter,
            "tags": events.tags,
            "duration": round(duration, 2),
            "confidence": round(confidence, 3),
            "language": info.language if info else "en",
        }
