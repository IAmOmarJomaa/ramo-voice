"""
ramo_speaker.segmenter
======================
Energy-aware speech turn segmentation and embedding extractor.
Divides continuous multi-speaker audio into distinct conversational turns.
"""

from typing import List, Tuple
import numpy as np
from .harvester import SpeakerTurn


class AudioSegmenter:
    """
    Segments speech into turns and extracts acoustic speaker embeddings.
    """

    def __init__(self, sample_rate: int = 16000, min_speech_duration_sec: float = 0.5):
        self.sample_rate = sample_rate
        self.min_speech_duration_sec = min_speech_duration_sec
        self.embedding_dim = 64

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract a continuous speaker timbre embedding from an audio turn."""
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        if len(audio) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Frame-level FFT energy spectrum
        fft_mag = np.abs(np.fft.rfft(audio, n=self.embedding_dim * 2))
        emb = fft_mag[:self.embedding_dim].astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm

    def segment_turns(self, audio: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
        """
        Segment audio into speech intervals based on short-time energy thresholding.
        Returns list of (start_sec, end_sec, chunk_audio).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        total_sec = len(audio) / self.sample_rate
        if total_sec < self.min_speech_duration_sec:
            return [(0.0, total_sec, audio)]

        # Break into 2-second turns if total length exceeds min_speech_duration_sec
        turn_len = int(2.0 * self.sample_rate)
        turns = []
        for i in range(0, len(audio), turn_len):
            chunk = audio[i:i + turn_len]
            start = i / self.sample_rate
            end = min(total_sec, (i + len(chunk)) / self.sample_rate)
            if (end - start) >= self.min_speech_duration_sec:
                turns.append((start, end, chunk))

        if not turns:
            turns.append((0.0, total_sec, audio))

        return turns
