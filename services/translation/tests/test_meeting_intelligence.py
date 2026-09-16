import pytest
from ramo_translate.intelligence import MeetingIntelligenceSynthesizer


def test_format_intelligence_prompt():
    synth = MeetingIntelligenceSynthesizer()
    dialogue = [
        "SPEAKER_00: Dr. Smith, will you send the slides before 5pm?",
        "SPEAKER_01: Yes, I will send the executive report by 5pm."
    ]
    prompt = synth.format_prompt(dialogue)
    assert "<|im_start|>system" in prompt
    assert "<recent_dialogue>" in prompt
    assert "</recent_dialogue>" in prompt
    assert "SPEAKER_00: Dr. Smith, will you send the slides before 5pm?" in prompt
    assert "action_items" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_parse_intelligence_json():
    synth = MeetingIntelligenceSynthesizer()
    raw_json = '''{
        "action_items": [
            {
                "task": "Send the executive report",
                "assignee": "SPEAKER_01",
                "assigned_by": "SPEAKER_00",
                "deadline": "5pm"
            }
        ],
        "direct_orders": [],
        "verification_claims": [],
        "key_notes": ["Executive report distribution agreed for 5pm"]
    }'''
    res = synth.parse_output(raw_json)
    assert len(res["action_items"]) == 1
    assert res["action_items"][0]["task"] == "Send the executive report"
    assert res["action_items"][0]["assignee"] == "SPEAKER_01"
    assert res["action_items"][0]["deadline"] == "5pm"
    assert "Executive report distribution agreed for 5pm" in res["key_notes"]


def test_parse_intelligence_empty_fallback():
    synth = MeetingIntelligenceSynthesizer()
    res = synth.parse_output("")
    assert res["action_items"] == []
    assert res["direct_orders"] == []
    assert res["verification_claims"] == []
    assert res["key_notes"] == []
