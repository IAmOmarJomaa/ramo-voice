import pytest
from fastapi.testclient import TestClient
from ramo_translate.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["service"] == "ramo-translation"


def test_translate_fastpath_endpoint(client):
    res = client.post(
        "/v1/translate",
        json={
            "text": "Thank you!",
            "source_language": "en",
            "target_language": "fr",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["translated_text"] == "Merci !"
    assert data["target_language"] == "fr"


def test_action_item_endpoint(client):
    res = client.post(
        "/v1/meeting/action_items",
        json={
            "text": "I will deliver the pull request by tomorrow morning.",
            "speaker_id": "SPEAKER_00",
        },
    )
    assert res.status_code == 200
    assert res.json()["detected_action"] == "task_delegation"
