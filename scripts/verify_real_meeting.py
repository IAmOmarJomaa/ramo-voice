"""
scripts/verify_real_meeting.py
==============================
Live Verification Runner for SOTA Multi-Speaker TTS Orchestration:
Streams real meeting audio fixture (benchmark_turn_taking_3min.wav / meeting_sample_en.wav)
through the complete sovereign pipeline:
- Acoustic Overlap & Crosstalk Rejection
- Faster-Whisper Transcription & Word Timestamps
- Online CampPlus Speaker Centroids
- Single-Speaker Pristine Voice Harvesting (<4.5s Anchor -> >=4.5s Zero-Shot Promotion)
- Prosodic Clause Buffering (breath pause preservation)
- WSOLA Dynamic Latency Pacer (pitch-preserving speedup)
- Listen-Ready Audio File Output in tests/outputs/
"""

import os
import sys
import time
import asyncio
import soundfile as sf
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure services are on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for svc in ["gateway", "stt", "diarization", "enhancement", "translation", "tts", "common"]:
    p = os.path.join(ROOT, "services", svc, "src")
    if p not in sys.path:
        sys.path.insert(0, p)

from ramo_gateway.pipeline_client import PipelineDispatcher


async def run_live_meeting_verification():
    print("=" * 80)
    print("🎙️  SOVEREIGN MULTI-SPEAKER TTS ORCHESTRATOR: REAL MEETING VERIFICATION")
    print("=" * 80)

    sr = 16000
    dispatcher = PipelineDispatcher(sample_rate=sr)
    print("⚙️  Warming up neural engines (Faster-Whisper, CampPlus, Qwen, Kokoro, F5-TTS)...")
    await dispatcher.initialize()
    print("✅ All neural engines initialized successfully.\n")

    # Locate audio fixture (Prefer 3-minute multi-speaker benchmark)
    fixture_path = os.path.join(ROOT, "tests", "fixtures", "benchmark_turn_taking_3min.wav")
    if not os.path.exists(fixture_path):
        fixture_path = os.path.join(ROOT, "tests", "fixtures", "meeting_sample_en.wav")

    print(f"📂 Loading audio fixture: {os.path.relpath(fixture_path, ROOT)}")
    audio_data, file_sr = sf.read(fixture_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    if file_sr != sr:
        from scipy.signal import resample
        num_samples = int(len(audio_data) * sr / file_sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)

    total_sec = len(audio_data) / sr
    print(f"📊 Fixture duration: {total_sec:.2f}s ({len(audio_data)} samples @ {sr}Hz)\n")

    # Output directory for listen-ready audio files
    output_dir = os.path.join(ROOT, "tests", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Segment audio into natural turns using turn segmenter
    turn_slices = dispatcher.segmenter.segment_turns(audio_data)
    print(f"✂️  Segmented recording into {len(turn_slices)} conversational turn(s).\n")

    print("-" * 80)
    print(f"{'Turn':<6} | {'Speaker':<12} | {'Dur(s)':<6} | {'Overlap':<8} | {'Profile Tier':<16} | {'WSOLA':<6} | {'Output WAV'}")
    print("-" * 80)

    last_known = "Unknown"
    saved_files = []

    for i, (t_start, t_end, chunk_pcm) in enumerate(turn_slices[:8], start=1):
        dur = t_end - t_start
        if dur < 0.3:
            continue

        # 0. Clean audio
        pcm16_in = (np.clip(chunk_pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        cleaned_f32, _ = dispatcher.clean_audio_pcm(pcm16_in)

        # 1. Overlap detection
        is_overlap, overlap_score = dispatcher.detect_overlap(cleaned_f32)
        overlap_label = "CROSSTALK" if is_overlap else "Clean"

        # 2. Speaker identification
        speaker_id = dispatcher.identify_speaker(cleaned_f32, last_known=last_known, is_overlap=is_overlap)
        last_known = speaker_id

        # 3. Faster-Whisper transcription
        stt_res = await dispatcher.process_stt(cleaned_f32)
        transcript = stt_res.get("raw_text", "").strip()
        words = stt_res.get("words", [])
        emotion = stt_res.get("emotion", "NEUTRAL")

        if not transcript:
            continue

        # 4. Voice harvesting (single-speaker purity filter)
        harvested_prof = dispatcher.harvest_speech_turn(
            speaker_id=speaker_id,
            audio_f32=cleaned_f32,
            transcript=transcript,
            is_overlap=is_overlap,
        )

        acc_sec = harvested_prof.duration_sec if harvested_prof else 0.0
        if harvested_prof and harvested_prof.ready_for_cloning:
            tier_label = f"TIER 1 (CLONED, {acc_sec:.1f}s)"
        else:
            tier_label = f"TIER 0 (ANCHOR, {acc_sec:.1f}s)"

        # 5. Translation to French
        translated_text, is_bypass, action = await dispatcher.translate_text(
            text=transcript,
            source_lang="en",
            target_lang="fr",
            session_id="verification_session",
            speaker_id=speaker_id,
        )

        # 6. Prosodic clause buffer
        prosody_buf = dispatcher.get_prosody_buffer("verification_session")
        clauses = prosody_buf.add_text(translated_text)
        clauses.extend(prosody_buf.flush())
        effective_clause = " ".join(clauses) if clauses else translated_text

        # 7. Dynamic Latency Pacer & Synthesis
        simulated_latency = 800 + (i * 120)  # simulate slight queue buildup
        tts_pcm, tts_sr, speed = await dispatcher.synthesize_speech(
            text=effective_clause,
            speaker_id=speaker_id,
            speaker_audio=cleaned_f32,
            current_latency_ms=simulated_latency,
            queue_words=len(effective_clause.split()),
        )

        # 8. Save output WAV for the user to listen
        out_filename = f"turn_{i:02d}_{speaker_id}_fr.wav"
        out_filepath = os.path.join(output_dir, out_filename)
        audio_f32_out = np.frombuffer(tts_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(out_filepath, audio_f32_out, tts_sr, format="WAV", subtype="PCM_16")
        saved_files.append((out_filepath, speaker_id, transcript, effective_clause, speed))

        print(
            f"#{i:<5} | {speaker_id:<12} | {dur:<6.2f} | {overlap_label:<8} | {tier_label:<16} | {speed:<6.2f}x | {out_filename}"
        )
        print(f"       🗣️  Source  : \"{transcript}\" [{len(words)} words, Emotion: {emotion}]")
        print(f"       🌐  French  : \"{effective_clause}\"")
        if action:
            print(f"       📋  Action  : {action}")
        print()

    print("=" * 80)
    print("🎧  LISTEN-READY AUDIO OUTPUTS SAVED:")
    print("=" * 80)
    for path, spk, src, fr, spd in saved_files:
        rel = os.path.relpath(path, ROOT)
        print(f"▶️  {rel}")
        print(f"    Speaker: {spk} | Speed: {spd}x")
        print(f"    Text   : {fr}\n")

    print(f"🎉 Verification complete! {len(saved_files)} turns processed, synthesized, and verified.")


if __name__ == "__main__":
    asyncio.run(run_live_meeting_verification())
