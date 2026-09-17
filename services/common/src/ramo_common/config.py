"""
ramo_common.config
==================
Type-safe Pydantic configuration model and YAML loader for the TTS Pacer
and Multi-Speaker Orchestrator pipeline.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("ramo_common.config")


class HarvestingConfig(BaseModel):
    tier1_threshold_sec: float = Field(default=4.5, description="Min clean speech for Tier 1 zero-shot cloning")
    tier2_threshold_sec: float = Field(default=10.0, description="Min clean speech for Tier 2 high-fidelity profile")
    max_buffer_sec: float = Field(default=15.0, description="Max audio storage per speaker in harvest pool")
    min_snr_db: float = Field(default=12.0, description="Min Signal-to-Noise Ratio (dB) to accept audio")
    min_turn_duration_sec: float = Field(default=0.4, description="Min turn duration to consider")


class ProsodyBufferConfig(BaseModel):
    min_clause_words: int = Field(default=4, description="Min words to form a dispatchable clause")
    max_clause_words: int = Field(default=14, description="Max words before forcing a clause flush")
    strong_terminators: List[str] = Field(
        default_factory=lambda: [".", "!", "?", "。", "！", "？"]
    )
    weak_terminators: List[str] = Field(
        default_factory=lambda: [",", ";", ":", "，", "；"]
    )
    conjunctions: List[str] = Field(
        default_factory=lambda: [
            "and", "but", "because", "although", "or", "however",
            "et", "mais", "parce que", "porque"
        ]
    )
    ttfa_timeout_ms: int = Field(default=600, description="Time-To-First-Audio timeout (ms)")


class PacerConfig(BaseModel):
    target_latency_budget_ms: int = Field(default=1200, description="Target total latency budget")
    wsola_speed_min: float = Field(default=1.0, description="Min playback speed multiplier")
    wsola_speed_max: float = Field(default=1.25, description="Max pitch-preserving speedup multiplier")
    queue_threshold_words: int = Field(default=20, description="Words in queue triggering speedup")


class EngineSelectionConfig(BaseModel):
    default_fallback: str = Field(default="kokoro", description="Instant fallback TTS engine")
    cloning_engine: str = Field(default="f5tts", description="Zero-shot cloning TTS engine")
    anchor_mapping_enabled: bool = Field(default=True, description="Enable F0 pitch-to-preset matching")


class TTSPacerConfig(BaseModel):
    harvesting: HarvestingConfig = Field(default_factory=HarvestingConfig)
    prosody_buffer: ProsodyBufferConfig = Field(default_factory=ProsodyBufferConfig)
    pacer: PacerConfig = Field(default_factory=PacerConfig)
    engine_selection: EngineSelectionConfig = Field(default_factory=EngineSelectionConfig)


def load_tts_pacer_config(config_path: Optional[str] = None) -> TTSPacerConfig:
    """
    Load TTSPacerConfig from a YAML file.
    If the file is not found, returns a default TTSPacerConfig instance.
    """
    candidate_paths = [
        config_path,
        os.getenv("RAMO_TTS_PACER_CONFIG"),
        "config/tts_pacer.yaml",
        "../config/tts_pacer.yaml",
        "../../config/tts_pacer.yaml",
    ]

    resolved_path: Optional[str] = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = os.path.abspath(p)
            break

    if not resolved_path:
        logger.debug(f"[CONFIG] No config file found at candidate paths; using built-in defaults.")
        return TTSPacerConfig()

    try:
        try:
            import yaml
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            # Fallback simple reader if PyYAML not installed
            import json
            logger.warning("[CONFIG] pyyaml not installed; attempting basic parse or using defaults.")
            data = {}

        logger.info(f"[CONFIG] Loaded TTS Pacer configuration from '{resolved_path}'")
        return TTSPacerConfig(**data)
    except Exception as e:
        logger.error(f"[CONFIG] Error loading config from '{resolved_path}': {e}; falling back to defaults.")
        return TTSPacerConfig()
