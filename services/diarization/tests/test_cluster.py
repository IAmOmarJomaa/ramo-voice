"""
Tests for SpeakerClusterer in ramo_speaker.cluster.
Harvested from StenoAI and Pyannote architectures.
"""

import pytest
import numpy as np
from ramo_speaker.cluster import SpeakerClusterer


def test_cluster_distinct_speakers():
    clusterer = SpeakerClusterer(similarity_threshold=0.75)

    # Two orthogonal speaker embeddings
    v_speaker1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_speaker2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    spk1 = clusterer.assign_or_update(v_speaker1)
    spk2 = clusterer.assign_or_update(v_speaker2)

    assert spk1 != spk2
    assert spk1 == "SPEAKER_00"
    assert spk2 == "SPEAKER_01"


def test_cluster_rolling_centroid_update():
    clusterer = SpeakerClusterer(similarity_threshold=0.75, momentum=0.8)

    # Initial embedding
    v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk_id = clusterer.assign_or_update(v_base)

    # Slightly perturbed embedding of same speaker
    v_perturbed = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    v_perturbed /= np.linalg.norm(v_perturbed)

    spk_id_again = clusterer.assign_or_update(v_perturbed)
    assert spk_id_again == spk_id

    # Verify centroid moved towards perturbed vector
    centroid = clusterer.get_centroid(spk_id)
    assert centroid[1] > 0.0


def test_cluster_crosstalk_rejection():
    clusterer = SpeakerClusterer(similarity_threshold=0.75)

    v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk_id = clusterer.assign_or_update(v_base)

    # Overlapping crosstalk embedding
    v_crosstalk = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    v_crosstalk /= np.linalg.norm(v_crosstalk)

    # With is_overlap=True, it should NOT corrupt the centroid
    centroid_before = clusterer.get_centroid(spk_id).copy()
    assigned = clusterer.assign_or_update(v_crosstalk, is_overlap=True)

    centroid_after = clusterer.get_centroid(spk_id)
    np.testing.assert_allclose(centroid_before, centroid_after)


def test_cluster_crosstalk_never_mints_new_speaker():
    clusterer = SpeakerClusterer(similarity_threshold=0.75)
    v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk_id = clusterer.assign_or_update(v_base)
    assert spk_id == "SPEAKER_00"

    # Overlapping audio with low similarity (< 0.75)
    v_crosstalk_low_sim = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    assigned = clusterer.assign_or_update(v_crosstalk_low_sim, is_overlap=True)
    # Must assign to nearest known speaker without creating SPEAKER_01
    assert assigned == "SPEAKER_00"
    assert len(clusterer.get_speakers()) == 1


def test_cluster_duration_short_interjection_inherits_previous():
    """Moonshine standard: Utterance < 1.0s inherits previous speaker to avoid interjection fragmentation."""
    clusterer = SpeakerClusterer(similarity_threshold=0.75)
    v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spk_id = clusterer.assign_or_update(v_base, duration_s=3.0)
    assert spk_id == "SPEAKER_00"

    # Short 0.5s interjection ("Yeah") with noisy/low similarity
    v_short_noise = np.array([0.2, 0.9, 0.0, 0.0], dtype=np.float32)
    v_short_noise /= np.linalg.norm(v_short_noise)

    # With previous_speaker_id provided, must inherit without creating SPEAKER_01
    assigned = clusterer.assign_or_update(
        v_short_noise,
        duration_s=0.5,
        previous_speaker_id="SPEAKER_00"
    )
    assert assigned == "SPEAKER_00"
    assert len(clusterer.get_speakers()) == 1


def test_cluster_duration_scaled_threshold():
    """Moonshine standard: 1.0s-3.0s utterances use scaled threshold, preventing false clusters."""
    clusterer = SpeakerClusterer(similarity_threshold=0.80)
    v_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    clusterer.assign_or_update(v_base, duration_s=4.0)

    # Embedding with sim ~ 0.65 (below 0.80 strict, but matches under scaled threshold at 1.2s)
    # Cosine sim = dot([1, 0, 0, 0], [0.65, sqrt(1 - 0.65^2), 0, 0]) = 0.65
    angle_vec = np.array([0.65, np.sqrt(1.0 - 0.65**2), 0.0, 0.0], dtype=np.float32)

    # At duration_s=1.2s, scaled threshold allows match
    assigned_scaled = clusterer.assign_or_update(angle_vec, duration_s=1.2)
    assert assigned_scaled == "SPEAKER_00"

    # At duration_s=4.0s with another orthogonal vector, strict threshold rejects and creates new speaker
    v_new = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assigned_strict = clusterer.assign_or_update(v_new, duration_s=4.0)
    assert assigned_strict == "SPEAKER_01"

