import pytest
from ramo_translate.engines.local_engine import LocalLLMEngine


def test_xml_sandboxed_prompt_structure():
    engine = LocalLLMEngine()
    prompt = engine.format_translation_prompt(
        text="will be talking about COVID-19 and the flu.",
        source_lang="en",
        target_lang="fr",
        sliding_window="SPEAKER_00: Good evening, I am Dr. Kenisha Zimmerman.",
        meeting_context="Duke Webinar",
    )
    assert "<|im_start|>system\n" in prompt
    assert "<input_to_translate>\nwill be talking about COVID-19 and the flu.\n</input_to_translate>" in prompt
    assert "<context>" in prompt
    assert "</context>" in prompt
    assert "SPEAKER_00: Good evening, I am Dr. Kenisha Zimmerman." in prompt
    assert "French" in prompt
    assert '{"translated_text": "..."}' in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_xml_sandboxed_prompt_no_context():
    engine = LocalLLMEngine()
    prompt = engine.format_translation_prompt(
        text="Hello world.",
        source_lang="en",
        target_lang="es",
        sliding_window="",
        meeting_context="",
    )
    assert "<input_to_translate>\nHello world.\n</input_to_translate>" in prompt
    assert "<context>\n" not in prompt
    assert "Spanish" in prompt
