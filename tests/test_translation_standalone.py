"""
tests/test_translation_standalone.py
=====================================
Standalone verification harness for ramO Translation & Intelligence Engine.
Tests real translation fidelity and action item classification.
"""

import argparse
import asyncio
import json
import sys
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_UTTERANCES = [
    ("Can you hear me? Can you hear me? Yeah, we can hear you.", "fr", "SPEAKER_00"),
    ("Good Evening, I am Dr. Kenisha Zimmerman. I am really pleased to welcome you tonight.", "fr", "SPEAKER_00"),
    ("We will have our panelists present during the first few minutes of the discussion.", "fr", "SPEAKER_01"),
    ("I will send the executive report by 5pm.", "fr", "SPEAKER_02"),
]


async def run_direct_engine():
    print("\n=======================================================")
    print("\u1f9e0 Testing Translation Engine (Direct Python Runtime)")
    print("======================================================\n")
    from ramo_translate.router import TranslationRouter

    router = TranslationRouter()
    await router.engine.load()


    for text, target_lang, spk in TEST_UTTERANCES:
        print(f"[*] Input [{spk}] ({target_lang}): \"{text}\"")
        res = await router.translate(text, source_lang="en", target_lang=target_lang, speaker_id=spk)
        print(f"  \u2705 Translated : \"{res.translated_text}\"")
        print(f"  \u26a1 Fast-Bypass: {res.is_bypass}")
        if res.detected_action:
            print(f"  \u1f4cb Action Item: {res.detected_action}")
        print()


def run_http_probe(url: str):
    print("\n======================================================")
    print(f"\u1f4e1 Probing Translation Service via HTTP: {url}")
    print("=======================================================\n")

    for text, target_lang, spk in TEST_UTTERANCES:
        payload = json.dumps({
            "text": text,
            "source_language": "en",
            "target_language": target_lang,
            "speaker_id": spk,
            "session_id": "standalone_bench",
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"*] Input [{spk}]: \"{text}\"")
                print(f"  \u2705 Translated : \"{data.get('translated_text')}\"")
                print(f"  \u26a1 Fast-Bypass: {data.get('is_bypass')}")
                if data.get("detected_action"):
                    print(f"  \u1f4cb Action Item: {data.get('detected_action')}")
                print()
        except Exception as e:
            print(f"  \u274c Error contacting {url}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ramO Translation Standalone Test Harness")
    parser.add_argument("--url", type=str, default="", help="HTTO URL for /v1/translate endpoint")
    args = parser.parse_args()

    if args.url:
        run_http_probe(args.url)
    else:
        asyncio.run(run_direct_engine())
