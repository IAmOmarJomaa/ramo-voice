"""
services/common/tests/test_config.py
====================================
Tests loading of centralized config/tts_pacer.yaml with type safety and default fallbacks.
"""

import os
import pytest
from ramo_common.config import load_tts_pacer_config, TTSPacerConfig


def test_default_config_fallback():
    """When no file exists, load_tts_pacer_config should return robust defaults."""
    cfg = load_tts_pacer_config(config_path="/non_existent_path.yaml")
    assert isinstance(cfg, TTSPacerConfig)
    assert cfg.harvesting.tier1_threshold_sec == 4.5
    assert cfg.harvesting.tier2_threshold_sec == 10.0
    assert cfg.harvesting.min_snr_db == 12.0
    assert cfg.prosody_buffer.min_clause_words == 4
    assert cfg.prosody_buffer.max_clause_words == 14
    assert cfg.prosody_buffer.ttfa_timeout_ms == 600
    assert cfg.pacer.wsola_speed_min == 1.0
    assert cfg.pacer.wsola_speed_max == 1.25
    assert cfg.engine_selection.default_fallback == "kokoro"


def test_load_real_config_file(tmp_path):
    """When a YAML file is provided, it should override defaults cleanly."""
    yaml_content = """
harvesting:
  tier1_threshold_sec: 5.0
  tier2_threshold_sec: 12.0
  max_buffer_sec: 20.0
  min_snr_db: 15.0

prosody_buffer:
  min_clause_words: 5
  max_clause_words: 16
  ttfa_timeout_ms: 500

pacer:
  target_latency_budget_ms: 1000
  wsola_speed_min: 1.0
  wsola_speed_max: 1.30

engine_selection:
  default_fallback: "supertonic"
  cloning_engine: "cosyvoice"
"""
    yaml_file = tmp_path / "tts_pacer.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_tts_pacer_config(config_path=str(yaml_file))
    assert cfg.harvesting.tier1_threshold_sec == 5.0
    assert cfg.harvesting.tier2_threshold_sec == 12.0
    assert cfg.harvesting.min_snr_db == 15.0
    assert cfg.prosody_buffer.min_clause_words == 5
    assert cfg.prosody_buffer.ttfa_timeout_ms == 500
    assert cfg.pacer.wsola_speed_max == 1.30
    assert cfg.engine_selection.default_fallback == "supertonic"
