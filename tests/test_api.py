from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import SAMPLE_PAYLOAD
from src.train import train_and_save


@pytest.fixture(scope="session", autouse=True)
def trained_artifacts():
    train_and_save(random_state=42)


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"]
    assert payload["feature_count"] == 24
    assert "UCI Machine Learning Repository" in payload["dataset_citation"]


def test_screen_valid_payload():
    response = client.post("/api/v1/screen", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    payload = response.json()
    assert set(["prediction", "probability", "top_features", "disclaimer"]).issubset(payload)
    assert payload["prediction"] in {"ckd", "notckd"}
    assert 0 <= payload["probability"] <= 1
    assert isinstance(payload["top_features"], list)


def test_screen_missing_required_field():
    invalid_payload = SAMPLE_PAYLOAD.copy()
    invalid_payload.pop("age")
    response = client.post("/api/v1/screen", json=invalid_payload)
    assert response.status_code == 422
