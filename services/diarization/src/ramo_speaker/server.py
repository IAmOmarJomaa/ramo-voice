"""
ramo_speaker.server
===================
Production FastAPI server for speaker diarization & clean voiceprint harvesting.
Exposes REST endpoints for conversational speaker attribution and zero-shot voice cloning harvesting.
"""

import io
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from .cluster import SpeakerClusterer
from .segmenter import AudioSegmenter
from .harvester import VoiceprintHarvester, SpeakerTurn
from ramo_common.logging import setup_service_logging, tail_service_log

logger = setup_service_logging("ramo_speaker")

clusterer = SpeakerClusterer(similarity_threshold=0.62, momentum=0.70)
segmenter = AudioSegmenter(sample_rate=16000)
harvester = VoiceprintHarvester(min_duration_sec=2.5, target_sr=16000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ramo_speaker diarization service...")
    yield


app = FastAPI(
    title="ramO Speaker Diarization API",
    version="0.1.0",
    description="Sovereign Speaker Diarization & Voiceprint Harvester Microservice",
    lifespan=lifespan
)


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "healthy",
        "service": "ramo-diarization",
        "tracked_speakers": len(clusterer.get_speakers()),
        "sample_rate": 16000
    }


@app.get("/logs")
async def get_logs(tail: int = 100):
    return {"service": "ramo_speaker", "lines": tail_service_log("ramo_speaker", n=tail)}


@app.post("/v1/speakers/reset")
async def reset_speakers():
    """Reset all speaker centroids and harvested profiles for a new session."""
    clusterer.reset()
    return {"status": "success", "message": "Speaker centroids reset successfully."}


@app.post("/v1/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """
    Diarize an audio file into speaker turns with start/end timestamps.
    """
    contents = await file.read()
    with io.BytesIO(contents) as buf:
        audio_data, sr = sf.read(buf, dtype="float32")

    duration_s = len(audio_data) / sr if sr > 0 else 0.0
    logger.info(
        f"📥 [DIAR_REQ] /v1/diarize received '{file.filename}' ({len(contents)} bytes, {duration_s:.2f}s @ {sr}Hz)"
    )

    turns = segmenter.segment_turns(audio_data)
    results = []

    for idx, (start, end, chunk) in enumerate(turns):
        logger.info(f"--- [DIAR_TURN {idx+1}/{len(turns)}] [{start:.2f}s -> {end:.2f}s] ({end-start:.2f}s) ---")
        embedding = segmenter.extract_embedding(chunk)
        spk_id = clusterer.assign_or_update(embedding, is_overlap=False)
        results.append({
            "speaker": spk_id,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2)
        })

    return {
        "speakers": list(set(r["speaker"] for r in results)),
        "turns": results,
        "total_turns": len(results)
    }


@app.post("/v1/harvest/voiceprint")
async def harvest_voiceprint(file: UploadFile = File(...)):
    """
    Harvest clean 3-second reference audio samples for each speaker in the recording.
    Returns metadata and availability for instant zero-shot cloning in ramo_voice.
    """
    contents = await file.read()
    with io.BytesIO(contents) as buf:
        audio_data, sr = sf.read(buf, dtype="float32")

    duration_s = len(audio_data) / sr if sr > 0 else 0.0
    logger.info(
        f"📥 [HARVEST_REQ] /v1/harvest/voiceprint received '{file.filename}' ({len(contents)} bytes, {duration_s:.2f}s @ {sr}Hz)"
    )

    turns = segmenter.segment_turns(audio_data)

    for idx, (start, end, chunk) in enumerate(turns):
        logger.info(f"--- [HARVEST_TURN {idx+1}/{len(turns)}] [{start:.2f}s -> {end:.2f}s] ({end-start:.2f}s) ---")
        embedding = segmenter.extract_embedding(chunk)
        spk_id = clusterer.assign_or_update(embedding, is_overlap=False)
        harvester.add_turn(SpeakerTurn(
            speaker_id=spk_id,
            start_sec=start,
            end_sec=end,
            audio=chunk,
            is_overlap=False
        ))

    harvested_list = []
    for spk_id in harvester.list_harvested_speakers():
        sample = harvester.get_best_sample(spk_id)
        if sample is not None:
            harvested_list.append({
                "speaker_id": spk_id,
                "duration_sec": round(len(sample) / sr, 2),
                "samples_count": len(sample),
                "ready_for_cloning": True
            })

    return {
        "status": "success",
        "harvested_speakers": harvested_list,
        "total_harvested": len(harvested_list)
    }
