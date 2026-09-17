import numpy as np
import pytest
from ramo_speaker.cluster import SpeakerClusterer


def test_clusterer_reset():
    clusterer = SpeakerClusterer(similarity_threshold=0.62)
    v1 = np.random.randn(192)
    spk1 = clusterer.assign_or_update(v1)
    assert spk1 == "SPEAKER_00"
    assert len(clusterer.get_speakers()) == 1

    # Call reset
    clusterer.reset()
    assert len(clusterer.get_speakers()) == 0

    # New audio should be assigned SPEAKER_00 again
    v2 = np.random.randn(192)
    spk2 = clusterer.assign_or_update(v2)
    assert spk2 == "SPEAKER_00"
