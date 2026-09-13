"""
Integration tests for ramo_gateway.server FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from ramo_gateway.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ramo-gateway"
