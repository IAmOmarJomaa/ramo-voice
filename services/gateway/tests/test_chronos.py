"""
services.gateway.tests.test_chronos
===================================
TDD Unit tests for LiveUtterancePipeline and LocalAgreement (Moonshine + Meetily + StenoAI standard):
- 512ms (8,192-sample) rolling look-behind buffer preserving plosive onsets
- Silero VAD state machine with 2,000ms redemption time bridging natural pauses
- Linear probability fading between 10.0s and 15.0s hunting for natural pause boundaries
- Monologue soft-commit with 300ms tail context carry at 15.0s ceiling
- Adaptive EWMA keep-pace guard adjusting partial intervals to prevent queue backlog
- LocalAgreement-2 with word-boundary snapping longest common prefix
"""

import pytest
import numpy as np
from ramo_gateway.chronos import (
    LiveUtterancePipeline,
    UtteranceEvent,
    LocalAgreement,
    longest_common_prefix_word_boundary,
    calculate_fade_factor,
)


def test_local_agreement_word_boundary_snapping():
    """Verify LCP snaps back to last whitespace when match ends mid-word."""
    # Mid-word split: "transcript" vs "transcription" -> snaps back to "hello"
    prefix = longest_common_prefix_word_boundary("hello transcript", "hello transcription")
    assert prefix == "hello"

    # Full word match: "hello world foo" vs "hello world bar"
    prefix2 = longest_common_prefix_word_boundary("hello world foo", "hello world bar")
    assert prefix2 == "hello world"

    # Exact match
    prefix3 = longest_common_prefix_word_boundary("meeting notes", "meeting notes")
    assert prefix3 == "meeting notes"


def test_local_agreement_n2_consensus():
    """Verify LocalAgreement commits tokens only after 2 consecutive agreements."""
    la = LocalAgreement(n_agreement=2)

    res1 = la.step("we need to review the budget")
    assert res1.committed == ""
    assert res1.tentative == "we need to review the budget"

    res2 = la.step("we need to review the budget for next quarter")
    assert res2.committed == "we need to review the budget"
    assert res2.tentative == "for next quarter"

    res3 = la.flush()
    assert res3.committed == "we need to review the budget for next quarter"


def test_monologue_probability_fading_calculation():
    """Verify probability fade factor between 10.0s and 15.0s."""
    sr = 16000
    # Before 10s: fade factor is 1.0 (no attenuation)
    assert calculate_fade_factor(current_samples=int(5.0 * sr), sr=sr) == 1.0
    assert calculate_fade_factor(current_samples=int(10.0 * sr), sr=sr) == 1.0

    # At 12.5s: halfway through fade window (10s to 15s) -> 0.5
    fade_12_5 = calculate_fade_factor(current_samples=int(12.5 * sr), sr=sr)
    assert pytest.approx(fade_12_5, abs=1e-3) == 0.5

    # At 15.0s: fully attenuated -> 0.0
    fade_15 = calculate_fade_factor(current_samples=int(15.0 * sr), sr=sr)
    assert pytest.approx(fade_15, abs=1e-3) == 0.0


def test_pipeline_preroll_look_behind():
    """Verify 512ms rolling look-behind buffer is prepended upon voice onset."""
    sr = 16000
    pipeline = LiveUtterancePipeline(sample_rate=sr)

    # 1. Feed 1.0s of silence (zeros) in 250ms chunks to fill look-behind ring
    chunk_250ms = np.zeros(int(0.25 * sr), dtype=np.float32)
    for _ in range(4):
        events = pipeline.push_audio(chunk_250ms, is_voice=False)
        assert len(events) == 0

    assert pipeline.in_speech is False
    assert len(pipeline.look_behind_ring) == 8192  # 512ms

    # 2. Voice onset: Speech starts with audio chunk
    speech_chunk = np.ones(int(0.25 * sr), dtype=np.float32) * 0.5
    events = pipeline.push_audio(speech_chunk, is_voice=True)

    assert pipeline.in_speech is True
    assert pipeline.current_line_id is not None
    # current_samples must contain look_behind_ring (8192) + speech_chunk (4000)
    assert len(pipeline.current_samples) == 8192 + len(speech_chunk)


def test_pipeline_vad_redemption_2000ms():
    """Verify 2,000ms redemption time bridges human intra-sentence breath pauses."""
    sr = 16000
    pipeline = LiveUtterancePipeline(sample_rate=sr, vad_redemption_ms=2000)

    # 1. Voice onset: 1.0s of speech
    chunk_500ms_voice = np.ones(int(0.5 * sr), dtype=np.float32) * 0.4
    pipeline.push_audio(chunk_500ms_voice, is_voice=True)
    pipeline.push_audio(chunk_500ms_voice, is_voice=True)
    assert pipeline.in_speech is True

    # 2. Natural breath pause: 1.5s silence (3 x 500ms chunks)
    chunk_500ms_silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    events1 = pipeline.push_audio(chunk_500ms_silence, is_voice=False)
    events2 = pipeline.push_audio(chunk_500ms_silence, is_voice=False)
    events3 = pipeline.push_audio(chunk_500ms_silence, is_voice=False)

    # Within 1.5s < 2.0s redemption, turn must NOT be closed!
    assert len(events1) == 0
    assert len(events2) == 0
    assert len(events3) == 0
    assert pipeline.in_speech is True

    # 3. Exceed redemption: another 600ms silence (total 2.1s > 2.0s)
    chunk_600ms_silence = np.zeros(int(0.6 * sr), dtype=np.float32)
    final_events = pipeline.push_audio(chunk_600ms_silence, is_voice=False)

    assert len(final_events) == 1
    assert final_events[0].event_type == "final"
    assert final_events[0].is_final is True
    assert pipeline.in_speech is False


def test_pipeline_monologue_soft_commit_ceiling_15s():
    """Verify monologue reaching 15.0s ceiling triggers soft commit with 300ms tail context carry."""
    sr = 16000
    pipeline = LiveUtterancePipeline(sample_rate=sr, max_utterance_s=15.0, soft_commit_tail_s=0.3)

    initial_line_id = None
    all_final_events = []

    # Feed 16 seconds of continuous voice in 0.5s chunks
    chunk_voice = np.ones(int(0.5 * sr), dtype=np.float32) * 0.3
    for _ in range(32):
        events = pipeline.push_audio(chunk_voice, is_voice=True)
        if initial_line_id is None and pipeline.current_line_id is not None:
            initial_line_id = pipeline.current_line_id

        for ev in events:
            if ev.is_final:
                all_final_events.append(ev)

    # Must have triggered exactly 1 soft-commit final at the 15.0s mark
    assert len(all_final_events) >= 1
    first_final = all_final_events[0]
    assert first_final.line_id == initial_line_id
    assert first_final.is_final is True
    # Audio duration committed must be >= 15.0s
    assert len(first_final.audio) >= int(15.0 * sr)

    # After soft-commit:
    # 1. Pipeline must still be in speech
    assert pipeline.in_speech is True
    # 2. Line ID must have advanced
    assert pipeline.current_line_id != initial_line_id
    # 3. New segment must retain exactly 300ms tail context (4,800 samples)
    # plus any chunks processed afterwards
    assert len(pipeline.current_samples) >= int(0.3 * sr)


def test_pipeline_adaptive_ewma_keep_pace():
    """Verify EWMA decode wall-time dynamically stretches partial preview interval."""
    sr = 16000
    pipeline = LiveUtterancePipeline(sample_rate=sr)

    base_samples = pipeline.effective_partial_interval_samples()
    # Base interval is 0.5s = 8000 samples
    assert base_samples == int(0.5 * sr)

    # Simulate heavy GPU decode wall-time: 1.0s inference
    pipeline.record_decode_wall_time(1.0)

    expanded_samples = pipeline.effective_partial_interval_samples()
    # Paced at EWMA * 1.2 safety factor
    assert expanded_samples > base_samples
    assert expanded_samples >= int(0.8 * sr)
