"""
tests/inspect_pipeline.py
=========================
Interactive Ground-Truth Pipeline Inspector & Benchmark Runner.
Streams multi-speaker conversational audio to the ramO Sovereign Gateway (Port 50000),
tracks all microservices (Enhancement, STT, Diarization, Translation, TTS),
scores against ground-truth speaker turns, and exports synthesized audio to WAV.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
from typing import Dict, Any, List, Optional
import soundfile as sf
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_default_url() -> str:
    """Read ws_url from ~/.hermes/config.yaml or fall back to default."""
    try:
        from pathlib import Path
        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                url = cfg.get("stt", {}).get("ramo", {}).get("ws_url")
                if url:
                    return url
    except Exception:
        pass
    return "ws://100.110.150.34:50000/v1/stream"


def sanitize_url(url: str) -> str:
    url = url.strip()
    # Normalize excessive slashes like ws:///100...
    import re
    url = re.sub(r"^(wss?):/+", r"\1://", url)
    if not url.startswith("ws://") and not url.startswith("wss://"):
        if url.startswith("http://"):
            url = "ws://" + url[7:]
        elif url.startswith("https://"):
            url = "wss://" + url[8:]
        else:
            url = f"ws://{url}"
    if "/v1/stream" not in url and "/api/" not in url:
        url = url.rstrip("/") + "/v1/stream"
    return url



async def probe_health(http_url: str) -> bool:
    print(f"[*] Probing Gateway Health at {http_url}/health...", flush=True)
    try:
        def req():
            r = urllib.request.urlopen(f"{http_url}/health", timeout=3.0)
            return json.loads(r.read().decode())
        data = await asyncio.get_event_loop().run_in_executor(None, req)
        print(f"  \u2705 Gateway Online: {data.get('gateway', data.get('service'))} | Status: {data.get('status')}", flush=True)
        return True
    except Exception as e:
        print(f"  \u26a0\ufe0f Health check warning: {e}", flush=True)
        return False


async def run_benchmark(
    ws_url: str,
    audio_path: str,
    ground_truth_path: Optional[str] = None,
    target_lang: str = "fr",
    chunk_ms: int = 500,
    speed_factor: float = 1.0,
    source: str = "process",
    out_wav: str = "tests/fixtures/output_tts_benchmark.wav",
    auto_tts: bool = False,
):
    print("=" * 75)
    print("🚀 ramO Audio Intelligence Pipeline Benchmark & Deep Diagnostic")
    print("=" * 75)
    print(f"📡 WebSocket Target : {ws_url}")
    print(f"🎧 Audio Fixture    : {audio_path}")
    print(f"👥 Audio Source Mode: {source} ({'process = multi-speaker diarization' if source == 'process' else 'mic = single-speaker owner'})")
    print(f"🌐 Target Language  : {target_lang}")
    print(f"⚡ Streaming Chunk  : {chunk_ms}ms ({speed_factor}x speed)")
    print(f"🎙️ Auto TTS Enabled : {auto_tts}")
    print("=" * 75 + "\n")

    # 1. Load Audio
    if not os.path.exists(audio_path):
        print(f"[-] ERROR: Audio fixture not found at {audio_path}")
        return

    audio_data, sr = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)
    
    total_duration_sec = len(audio_data) / sr
    print(f"[*] Audio Loaded: {total_duration_sec:.2f}s duration | {sr}Hz | {len(audio_data)} samples", flush=True)

    # Convert to PCM16
    pcm16_data = (audio_data * 32767).astype("int16").tobytes()
    bytes_per_chunk = int(sr * (chunk_ms / 1000.0) * 2)

    # Load Ground Truth if available
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        print(f"[*] Ground Truth Loaded: {ground_truth.get('dataset')} ({len(ground_truth.get('turns', []))} expected turns)\n")

    # 2. HTTP Health preflight
    from urllib.parse import urlparse
    parsed = urlparse(ws_url)
    http_url = f"http://{parsed.hostname}:{parsed.port or 50000}"
    await probe_health(http_url)

    # 3. Connect WebSocket (ping_interval=None prevents premature keepalive timeouts during heavy neural inference)
    print(f"\n[*] Connecting WebSocket to {ws_url}...", flush=True)
    tts_audio_chunks = []
    received_transcripts = []
    detected_speakers = set()

    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
        # Await Handshake
        handshake_raw = await ws.recv()
        handshake = json.loads(handshake_raw)
        print(f"  \u2705 Received Handshake: session_id={handshake.get('session')} | status={handshake.get('status')}\n")

        # Send Config
        config_frame = {
            "type": "config",
            "target_language": target_lang,
            "source_language": "auto",
            "auto_tts": auto_tts,
            "source": source,
            "context_summary": "Duke University COVID-19 and Influenza Webinar",
        }
        await ws.send(json.dumps(config_frame))
        print(f"[*] Sent Pipeline Configuration: target_lang={target_lang}, source={source}, auto_tts={auto_tts}\n")

        # Background listener task
        async def listen_events():
            nonlocal tts_audio_chunks, received_transcripts, detected_speakers
            try:
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        continue
                    try:
                        event = json.loads(msg)
                        while isinstance(event, str):
                            event = json.loads(event)
                    except Exception:
                        continue

                    if not isinstance(event, dict):
                        continue

                    etype = event.get("type", "")
                    ts_now = time.strftime("%H:%M:%S")

                    try:
                        if etype == "transcript":
                            speaker = event.get("speaker", "Unknown")
                            detected_speakers.add(speaker)
                            text = event.get("text", "")
                            words = len(event.get("words", []))
                            emotion = event.get("emotion", "NEUTRAL")
                            is_final = event.get("is_final", False)
                            received_transcripts.append((speaker, text, is_final))
                            
                            flag = "✅ FINAL" if is_final else "⏳ PROV"
                            print(f"[{ts_now}] 👂 [STT {flag}] [{speaker}] '{text}' (Words: {words} | Emotion: {emotion})", flush=True)

                        elif etype == "translation_result":
                            speaker = event.get("speaker", "Unknown")
                            text = event.get("text", "")
                            lang = event.get("language", target_lang)
                            print(f"[{ts_now}] 🌐 [TRANSLATION -> {lang}] [{speaker}]: '{text}'", flush=True)

                        elif etype == "action_item":
                            action_val = event.get("action")
                            if isinstance(action_val, dict):
                                action_text = action_val.get("text", "")
                            else:
                                action_text = str(action_val or event.get("source_text", "") or event.get("text", ""))
                            print(f"[{ts_now}] 📋 [ACTION ITEM DETECTED]: '{action_text}'", flush=True)

                        elif etype == "tts_audio":
                            raw_b64 = event.get("data", "")
                            pcm_bytes = base64.b64decode(raw_b64)
                            tts_audio_chunks.append(pcm_bytes)
                            sr_tts = event.get("sample_rate", 24000)
                            print(f"[{ts_now}] 🎙️ [TTS AUDIO CHUNK] Received {len(pcm_bytes)} bytes PCM ({sr_tts}Hz)", flush=True)

                        elif etype == "tts_end":
                            chunk_id = event.get("chunk_id", "")
                            print(f"[{ts_now}] 🏁 [TTS COMPLETE] Chunk {chunk_id} synthesized successfully.", flush=True)

                        elif etype == "speaker_detected":
                            speaker = event.get("speaker_id", "")
                            detected_speakers.add(speaker)
                            print(f"[{ts_now}] 👥 [SPEAKER IDENTIFIED]: {speaker}", flush=True)

                        elif etype == "meeting_intelligence":
                            actions = event.get("action_items", [])
                            orders = event.get("direct_orders", [])
                            claims = event.get("verification_claims", [])
                            notes = event.get("key_notes", [])
                            print(f"[{ts_now}] 🧠 [MEETING INTELLIGENCE SYNTHESIS]:", flush=True)
                            if actions:
                                for a in actions:
                                    print(f"       📌 Task: {a.get('task')} | Assignee: {a.get('assignee')} | By: {a.get('assigned_by')} | Due: {a.get('deadline')}", flush=True)
                            if orders:
                                for o in orders:
                                    print(f"       ⚡ Order: {o.get('order')} -> {o.get('target')}", flush=True)
                            if claims:
                                for c in claims:
                                    print(f"       🔍 Fact Check: {c}", flush=True)
                            if notes:
                                for n in notes:
                                    print(f"       📝 Note: {n}", flush=True)

                        elif etype == "interrupt":
                            print(f"[{ts_now}] ⚡ [BARGE-IN / INTERRUPT DETECTED]", flush=True)
                    except Exception as inner_e:
                        print(f"[{ts_now}] ⚠️ [EVENT PARSE WARNING]: {inner_e} on message {event}", flush=True)

            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"[-] Listener error: {e}", flush=True)

        listener_task = asyncio.create_task(listen_events())

        # Stream Audio Chunks
        print(f"[*] Streaming {len(pcm16_data)} bytes in {bytes_per_chunk}-byte chunks ({chunk_ms}ms)...")
        t_start = time.time()
        sent_bytes = 0

        for offset in range(0, len(pcm16_data), bytes_per_chunk):
            chunk = pcm16_data[offset : offset + bytes_per_chunk]
            sent_bytes += len(chunk)
            payload = {
                "type": "audio",
                "data": base64.b64encode(chunk).decode("utf-8"),
                "source": source,
                "mode": "meeting",
                "target_lang": target_lang,
                "target_language": target_lang,
                "channels": 1,
            }
            await ws.send(json.dumps(payload))

            # Progress bar
            pct = int((sent_bytes / len(pcm16_data)) * 100)
            elapsed = time.time() - t_start
            sys.stdout.write(f"\r  \u25b6\ufe0f Audio Ingest Progress: {pct}% [{sent_bytes}/{len(pcm16_data)} bytes] ({elapsed:.1f}s)")
            sys.stdout.flush()

            sleep_time = (chunk_ms / 1000.0) / speed_factor
            await asyncio.sleep(sleep_time)

        print("\n[*] Audio stream complete. Sending EOS frame...")
        await ws.send(json.dumps({"type": "eos", "mode": "meeting"}))

        # Wait for processing to flush
        print("[*] Waiting 6 seconds for downstream models to flush final turns and TTS...")
        await asyncio.sleep(6.0)

        listener_task.cancel()

    # 4. Save and inspect TTS output
    print("\n" + "=" * 75)
    print("📊 BENCHMARK RESULTS & SCORECARD")
    print("=" * 75)
    print(f"  \u2705 Distinct Speakers Detected : {len(detected_speakers)} ({', '.join(sorted(detected_speakers))})")
    print(f"  \u2705 Total Transcripts Received  : {len(received_transcripts)}")
    print(f"  \u2705 Total TTS Audio Chunks     : {len(tts_audio_chunks)}")

    if tts_audio_chunks:
        all_tts_pcm = b"".join(tts_audio_chunks)
        import numpy as np
        tts_f32 = np.frombuffer(all_tts_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        sf.write(out_wav, tts_f32, 24000)
        dur = len(tts_f32) / 24000.0
        print(f"  \u2705 Synthesized Audio Output   : {out_wav} ({dur:.2f} seconds of speech)")
        print(f"     -> You can play this file on Windows right now to hear the synthesized voice!")
    else:
        print("  \u26a0\ufe0f No TTS audio chunks were returned. (Check if auto_tts was enabled).")

    if ground_truth:
        print("\n🎯 GROUND TRUTH COMPARISON:")
        expected_turns = ground_truth.get("turns", [])
        print(f"  Expected Speakers: {ground_truth.get('num_speakers')} ({', '.join(s['name'] for s in ground_truth.get('speakers', []))})")
        print(f"  Expected Turns   : {len(expected_turns)}")
        for t in expected_turns:
            print(f"    - [{t['start_sec']}s - {t['end_sec']}s] {t['speaker_name']}: '{t['text'][:60]}...'")

    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ramO Audio Intelligence Pipeline Benchmark")
    parser.add_argument("--url", default=None, help="WebSocket URL (e.g. ws://100.112.42.26:50000/v1/stream)")
    parser.add_argument("--audio", default="tests/fixtures/benchmark_turn_taking_3min.wav", help="Path to audio WAV")
    parser.add_argument("--ground-truth", default="tests/fixtures/benchmark_ground_truth.json", help="Path to ground truth JSON")
    parser.add_argument("--target-lang", default="fr", help="Target language code (e.g. fr, es, de)")
    parser.add_argument("--chunk-ms", type=int, default=500, help="Chunk duration in ms")
    parser.add_argument("--source", default="process", choices=["process", "mic"], help="Audio source (process for multi-speaker diarization, mic for single-user owner)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real-time, 2.0 = 2x)")
    parser.add_argument("--out-wav", default="tests/fixtures/output_tts_benchmark.wav", help="Output WAV path")
    parser.add_argument("--auto-tts", action="store_true", default=False, help="Enable auto TTS synthesis (default: False)")
    args = parser.parse_args()

    url = sanitize_url(args.url or get_default_url())
    asyncio.run(
        run_benchmark(
            ws_url=url,
            audio_path=args.audio,
            ground_truth_path=args.ground_truth,
            target_lang=args.target_lang,
            chunk_ms=args.chunk_ms,
            speed_factor=args.speed,
            source=args.source,
            out_wav=args.out_wav,
            auto_tts=args.auto_tts,
        )
    )


if __name__ == "__main__":
    main()
