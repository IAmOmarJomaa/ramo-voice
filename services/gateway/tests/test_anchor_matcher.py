"""
services/gateway/tests/test_anchor_matcher.py
=============================================
TDD tests for AnchorVoiceMatcher: mapping F0 pitch and embeddings to Kokoro preset voices.
"""

import pytest
import numpy as np
from ramo_gateway.anchor_matcher import AnchorVoiceMatcher


def test_pitch_to_kokoro_preset_matching():
    matcher = AnchorVoiceMatcher()

    # 1. High-pitch female speaker (F0 ~ 240 Hz)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    female_audio = 0.5 * np.sin(2 * np.pi * 240 * t)
    voice_female = matcher.match_voice(speaker_id="SPEAKER_01", audio=female_audio)
    assert voice_female in ["af_heart", "bf_emma", "af_bella", "af_nicole"]

    # 2. Low-pitch male speaker (F0 ~ 110 Hz)
    male_audio = 0.5 * np.sin(2 * np.pi * 110 * t)
    voice_male = matcher.match_voice(speaker_id="SPEAKER_02", audio=male_audio)
    assert voice_male in ["am_adam", "bm_george", "am_michael", "bm_lewis"]

    # 3. Consistency: Same speaker ID gets cached preset
    voice_female_repeat = matcher.match_voice(speaker_id="SPEAKER_01", audio=female_audio)
    assert voice_female_repeat == voice_female
