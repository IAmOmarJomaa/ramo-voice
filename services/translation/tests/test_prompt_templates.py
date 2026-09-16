import pytest
from ramo_translate.engines.local_engine import LocalLLMEngine


def test_prompt_template_french_loaded():
    engine = LocalLLMEngine()
    template = engine.get_language_template("fr")
    assert template is not None
    assert "Vous êtes un traducteur en temps réel" in template["system_rules"]
    assert "Traduisez le texte ci-dessus en français" in template["user_instruction"]


def test_apc_system_prompt_padding():
    engine = LocalLLMEngine()
    system_prompt = engine.get_aligned_system_prompt("fr")
    assert "strict, ultra-low-latency real-time speech translator" in system_prompt
    assert "French" in system_prompt
    assert system_prompt.startswith("<|im_start|>system\n")
    assert system_prompt.endswith("<|im_end|>\n")


def test_format_translation_prompt_with_sliding_window():
    engine = LocalLLMEngine()
    prompt = engine.format_translation_prompt(
        text="Can you hear me?",
        source_lang="en",
        target_lang="fr",
        sliding_window="Speaker 1: Hello everyone.",
        meeting_context="Project sync",
    )
    assert "<|im_start|>system" in prompt
    assert "<input_to_translate>\nCan you hear me?\n</input_to_translate>" in prompt
    assert "<context>" in prompt
    assert "Speaker 1: Hello everyone." in prompt
    assert "<|im_start|>assistant" in prompt


def test_parse_llm_json_or_raw():
    engine = LocalLLMEngine()
    json_out = '{"translated_text": "Pouvez-vous m\'entendre ?", "requires_retranslation": false, "detected_action_item": null}'
    res, action = engine.parse_llm_output(json_out, fallback_text="Can you hear me?")
    assert res == "Pouvez-vous m'entendre ?"
    assert action is None

    json_action = '{"translated_text": "J\'enverrai le rapport avant 17h.", "detected_action_item": "task_delegation"}'
    res2, action2 = engine.parse_llm_output(json_action, fallback_text="I will send the report by 5pm.")
    assert res2 == "J'enverrai le rapport avant 17h."
    assert action2 == "task_delegation"

    raw_out = "Pouvez-vous m'entendre ?"
    res3, action3 = engine.parse_llm_output(raw_out, fallback_text="Can you hear me?")
    assert res3 == "Pouvez-vous m'entendre ?"
