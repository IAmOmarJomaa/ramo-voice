#!/usr/bin/env python3
"""
deploy/test_stream_client.py
============================
Standalone real-time WebSocket audio streaming client for ramO Voice Engine.
Streams audio over WebSocket to verify STT, Diarization, Translation, and TTS
outside Bridge-Tauri.

Usage:
  python deploy/test_stream_client.py --url ws://ramo-gpu:50000/v1/stream
  python deploy/test_stream_client.py --url ws://100.102.134.65:50000/v1/stream
  python deploy/test_stream_client.py --url ws://127.0.0.1:50000/v1/stream --file tests/fixtures/meeting_sample_en.wav
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import websockets
except ImportError:
    print("[-] Error: websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("[-] Error: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)


async def run_client(url: str, wav_path: str, target_lang: str, auto_tts: bool):
    print("=" * 70)
    print("🎙️  ramO Voice Engine: Standalone Live Streaming Test Client")
    print("=" * 70)
    print(f"[*] Target WebSocket : {url}")
    print(f"[*] Audio Fixture    : {wav_path}")
    print(f"[*] Target Language  : {target_lang}")
    print(f"[*] Auto TTS Enabled : {auto_tts}")
    print("=" * 70 + "\n")

    if not os.path.exists(wav_path):
        print(f"[-] Error: WAV file not found: {wav_path}")
        return False

    audio_data, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        print(f"[*] Resampling audio from {sr}Hz to 16000Hz...")
        import scipy.signal
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = scipy.signal.resample(audio_data, num_samples)
        sr = 16000

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    pcm_bytes = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    duration_s = len(pcm_bytes) / 32000.0
    print(f"[*] Loaded {duration_s:.2f}s of 16kHz audio ({len(pcm_bytes)} bytes)\n")

    print(f"[*] Connecting to {url}...")
    try:
        async with websockets.connect(url, ping_interval=30, ping_timeout=30) as ws:
            # 1. Wait for connected handshake
            raw_msg = await ws.recv()
            handshake = json.loads(raw_msg)
            if handshake.get("type") == "connected":
                print(f"✅ [HANDSHAKE] Connected to {handshake.get('gateway', 'ramO Gateway')}")
                print(f"   Session ID: {handshake.get('session', 'unknown')}\n")
            else:
                print(f"[!] Unexpected handshake message: {handshake}")

            # 2. Send configuration frame
            config_msg = {
                "type": "config",
                "target_language": target_lang,
                "auto_tts": auto_tts,
                "source": "mic",
            }
            await ws.send(json.dumps(config_msg))
            print(f"[*] Sent configuration frame (target_lang={target_lang}, auto_tts={auto_tts})\n")

            done_event = asyncio.Event()
            stats = {
                "provisional_count": 0,
                "final_count": 0,
                "translations": [],
                "tts_received": 0,
                "tts_bytes": 0,
                "speakers": set(),
            }

            # Receiver worker
            async def receiver():
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        m_type = msg.get("type", "")

                        if m_type == "transcript":
                            is_final = msg.get("is_final", False)
                            spk = msg.get("speaker", "Unknown")
                            text = msg.get("text", "")
                            stats["speakers"].add(spk)
                            if is_final:
                                stats["final_count"] += 1
                                print(f"\n🟢 [FINAL STT] [{spk}] \"{text}\"")
                            else:
                                stats["provisional_count"] += 1
                                print(f"  ⚡ [PREVIEW] [{spk}] {text}")

                        elif m_type == "translation_result":
                            t_text = msg.get("text", "")
                            t_lang = msg.get("language", "")
                            stats["translations"].append(t_text)
                            print(f"🌐 [TRANSLATION] [{t_lang}] \"{t_text}\"")

                        elif m_type == "action_item":
                            act = msg.get("action", "")
                            print(f"📋 [ACTION ITEM] \"{act}\"")

                        elif m_type == "tts_audio":
                            stats["tts_received"] += 1
                            audio_b64 = msg.get("data", "")
                            raw_audio = base64.b64decode(audio_b64)
                            stats["tts_bytes"] += len(raw_audio)
                            dur = len(raw_audio) / (2.0 * msg.get("sample_rate", 24000))
                            print(f"🔊 [TTS AUDIO] Chunk: {dur:.2f}s ({len(raw_audio)} bytes) @ {msg.get('sample_rate')}Hz")

                        elif m_type == "tts_end":
                            print("🏁 [TTS COMPLETE] Speech synthesis completed for turn.")
                            done_event.set()

                        elif m_type == "interrupt":
                            print("⚡ [BARGE-IN] Interruption detected!")

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[-] Receiver error: {e}")

            receiver_task = asyncio.create_task(receiver())

            # 3. Stream audio in 0.5s chunks simulating live mic
            print("=" * 70)
            print("[*] Streaming audio at real-time pace (0.5s chunks)...")
            print("=" * 70 + "\n")
            chunk_size = 16000
            stream_start = time.time()

            for offset in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[offset : offset + chunk_size]
                frame = {
                    "type": "audio",
                    "data": base64.b64encode(chunk).decode("ascii"),
                    "target_language": target_lang,
                    "auto_tts": auto_tts,
                    "source": "mic",
                }
                await ws.send(json.dumps(frame))
                await asyncio.sleep(0.5)

            stream_dur = time.time() - stream_start
            print(f"\n[*] Finished streaming {duration_s:.2f}s of audio in {stream_dur:.2f}s")

            # 4. Send EOS flush
            print("[*] Sending EOS (End-Of-Speech) signal to trigger final commit...")
            await ws.send(json.dumps({"type": "eos"}))

            # 5. Wait for downstream models
            print("[*] Waiting for final translation and TTS completion (max 20s)...")
            try:
                await asyncio.wait_for(done_event.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                print("[-] Timeout waiting for final TTS")

            receiver_task.cancel()

            # Summary
            print("\n" + "=" * 70)
            print("📊 ramO Live Streaming Test Summary")
            print("=" * 70)
            print(f"[*] Provisional Preview Frames : {stats['provisional_count']}")
            print(f"[*] Final Committed Turns     : {stats['final_count']}")
            print(f"[*] Speakers Diarized         : {list(stats['speakers'])}")
            print(f"[*] Translations Received     : {len(stats['translations'])}")
            for i, t in enumerate(stats['translations'], 1):
                print(f"     {i}. \"{t}\"")
            print(f"[*] TTS Audio Chunks Received : {stats['tts_received']} ({stats['tts_bytes']} bytes)")
            print("=" * 70)

            success = stats["final_count"] > 0
            if success:
                print("🎉 TEST VERDICT: SUCCESS (Live pipeline verified!)\n")
            else:
                print("⚠️  TEST VERDICT: INCOMPLETE\n")
            return success

    except Exception as e:
        print(f"\n[-] Connection failed: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="ramO Live WebSocket Streaming Client")
    parser.add_argument("--url", default="ws://ramo-gpu:50000/v1/stream", help="WebSocket endpoint URL")
    parser.add_argument("--file", default="tests/fixtures/meeting_sample_en.wav", help="Path to 16kHz WAV audio")
    parser.add_argument("--lang", default="fr", help="Target translation language (default: fr)")
    parser.add_argument("--no-tts", action="store_true", help="Disable automatic TTS")

    args = parser.parse_args()
    success = asyncio.run(run_client(
        url=args.url,
        wav_path=args.file,
        target_lang=args.lang,
        auto_tts=not args.no_tts,
    ))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
