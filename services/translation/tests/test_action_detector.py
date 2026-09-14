import pytest
from ramo_translate.action_detector import detect_action_item


def test_detect_task_delegation():
    text = "I will send the executive report by tomorrow morning."
    action = detect_action_item(text)
    assert action == "task_delegation"


def test_detect_schedule_change():
    text = "Let's reschedule the architectural sync to Friday at 2pm."
    action = detect_action_item(text)
    assert action == "schedule_change"


def test_detect_fact_check():
    text = "Our active user retention was 85 percent last quarter."
    action = detect_action_item(text)
    assert action == "fact_check"


def test_detect_none_for_casual_speech():
    text = "Good morning everyone, happy Monday."
    action = detect_action_item(text)
    assert action is None
