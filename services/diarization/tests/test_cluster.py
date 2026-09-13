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
