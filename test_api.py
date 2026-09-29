"""
Integration tests for model prediction and payload validation.
"""

from fastapi.testclient import TestClient
import pytest
from src.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint_valid_payload(client):
    payload = {
        "tenure_months": 3,
        "contract_type": "Month-to-Month",
        "internet_service": "Fiber Optic",
        "payment_method": "Electronic Check",
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "support_tickets": 4,
        "paperless_billing": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_risk_score" in data
    assert "risk_level" in data
    assert 0.0 <= data["churn_risk_score"] <= 1.0


def test_prediction_handles_missing_total_charges(client):
    payload = {
        "tenure_months": 1,
        "contract_type": "Month-to-Month",
        "internet_service": "DSL",
        "payment_method": "Mailed Check",
        "monthly_charges": 45.0,
        "total_charges": None,
        "support_tickets": 0,
        "paperless_billing": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_level"] in ["LOW", "MEDIUM", "HIGH"]