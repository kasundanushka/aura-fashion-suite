"""
Integration Tests for Flask API Endpoints.
"""

import pytest
import json
from backend.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"

def test_get_products_endpoint(client):
    response = client.get("/api/products")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "products" in data
    assert len(data["products"]) > 0

def test_get_forecast_endpoint(client):
    prod_resp = client.get("/api/products")
    prods = json.loads(prod_resp.data).get("products", [])
    sku = next((p["sku"] for p in prods if not p["sku"].startswith("TEST-")), prods[0]["sku"])
    response = client.get(f"/api/forecast/{sku}?weeks=4")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["sku"] == sku
    assert "forecast" in data
    assert "model_evaluation" in data
    assert len(data["forecast"]) == 28

def test_get_forecast_invalid_weeks(client):
    prod_resp = client.get("/api/products")
    prods = json.loads(prod_resp.data).get("products", [])
    sku = next((p["sku"] for p in prods if not p["sku"].startswith("TEST-")), prods[0]["sku"])
    response = client.get(f"/api/forecast/{sku}?weeks=999")
    assert response.status_code == 400

def test_get_forecast_nonexistent_sku(client):
    response = client.get("/api/forecast/NONEXISTENT-999")
    assert response.status_code == 404

def test_get_trend_endpoint(client):
    prod_resp = client.get("/api/products")
    prods = json.loads(prod_resp.data).get("products", [])
    sku = next((p["sku"] for p in prods if not p["sku"].startswith("TEST-")), prods[0]["sku"])
    response = client.get(f"/api/trend/{sku}")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["sku"] == sku
    assert data["trend_status"] in ["rising", "stable", "declining"]
    assert "confidence_score" in data

def test_get_reorder_alerts_endpoint(client):
    response = client.get("/api/reorder-alerts")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "kpis" in data
    assert "alerts" in data
    assert len(data["alerts"]) > 0
