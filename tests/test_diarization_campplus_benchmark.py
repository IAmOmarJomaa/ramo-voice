"""
tests/test_diarization_campplus_benchmark.py
============================================
Automated verification test for CampPlus ONNX neural diarization and voice harvesting
using the 3-minute ground-truth multi-speaker benchmark audio fixture.
"""

import json
import os
import numpy as np
import pytest
import soundfile as sf

from ramo_speaker.cluster import SpeakerClusterer
from ramo_speaker.segmenter import AudioSegmenter
from ramo_speaker.harvester import VoiceprintHarvester, SpeakerTurn


@pytest.fixture
def benchmark_data():
    gt_path = "tests/fixtures/benchmark_ground_truth.json"
    wav_path = "tests/fixtures/benchmark_turn_taking_3min.wav"

    if not os.path.exists(gt_path) or not os.path.exists(wav_path):
        pytest.skip("Benchmark fixture files not found.")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    audio, sr = sf.read(wav_path, dtype="float32")
    return gt, audio, sr


def test_campplus_model_loaded():
    """Verify CampPlus ONNX model is discovered and loaded."""
    segmenter = AudioSegmenter()
    if segmenter._session is None:
        pytest.skip("CampPlus ONNX model not present in models/campplus.onnx")

    assert segmenter._session is not None
    assert segmenter.embedding_dim in (192, 512)


def test_three_speaker_separation_on_benchmark(benchmark_data):
    """
    Verify that CampPlus ONNX and SpeakerClusterer accurately separate
    Dr. Kenisha Zimmerman, Dr. Danny Benjamin, and Dr. Mike Smith into 3 distinct speakers.
    """
    gt, audio, sr = benchmark_data
    segmenter = AudioSegmenter()
    if segmenter._session is None:
        pytest.skip("CampPlus ONNX model not present in models/campplus.onnx")

    clusterer = SpeakerClusterer(similarity_threshold=0.62, momentum=0.70)
    harvester = VoiceprintHarvester(tier1_threshold_sec=4.5, tier2_threshold_sec=10.0)

    assigned_speakers = []
    embeddings = []

    for turn in gt["turns"]:
        start_samp = int(turn["start_sec"] * sr)
        end_samp = int(turn["end_sec"] * sr)
        # Use first 4 seconds of speech turn for pure timbre extraction
        slice_samp = min(end_samp, start_samp + int(4.0 * sr))
        turn_audio = audio[start_samp:slice_samp]

        emb = segmenter.extract_embedding(turn_audio)
        embeddings.append((turn["speaker_name"], emb))

        spk_id = clusterer.assign_or_update(emb, is_overlap=False)
        assigned_speakers.append(spk_id)

        harvester.add_turn(
            SpeakerTurn(
                speaker_id=spk_id,
                start_sec=turn["start_sec"],
                end_sec=turn["end_sec"],
                audio=turn_audio,
                transcript=turn["text"][:50],
                is_overlap=False,
            )
        )

    # 1. Verify 3 distinct speakers assigned
    unique_speakers = list(set(assigned_speakers))
    assert len(unique_speakers) == 3, f"Expected 3 distinct speakers, got: {assigned_speakers}"
    assert assigned_speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    # 2. Verify cross-speaker cosine similarities are strictly under 0.62 threshold
    sim_zimmerman_benjamin = float(np.dot(embeddings[0][1], embeddings[1][1]))
    sim_zimmerman_smith = float(np.dot(embeddings[0][1], embeddings[2][1]))
    sim_benjamin_smith = float(np.dot(embeddings[1][1], embeddings[2][1]))

    assert sim_zimmerman_benjamin < 0.62, f"Expected < 0.62, got {sim_zimmerman_benjamin:.4f}"
    assert sim_zimmerman_smith < 0.62, f"Expected < 0.62, got {sim_zimmerman_smith:.4f}"
    assert sim_benjamin_smith < 0.62, f"Expected < 0.62, got {sim_benjamin_smith:.4f}"

    # 3. Verify harvester has candidate samples for all 3 speakers
    for spk in unique_speakers:
        assert harvester.has_candidate(spk)
        sample = harvester.get_best_sample(spk)
        assert sample is not None
        assert len(sample) >= int(3.5 * sr)
