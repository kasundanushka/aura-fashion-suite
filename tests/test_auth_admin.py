"""
Unit and Integration Tests for Authentication and Role-Based Admin Routes.
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

def test_admin_login_success(client):
    """Verifies that primary administrator can authenticate successfully."""
    response = client.post("/api/auth/login", json={
        "email": "admin@aura.ai",
        "password": "admin123"
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["user"]["role"] == "admin"
    assert "token" in data

def test_buyer_login_success(client):
    """Verifies that demo buyer account authenticates with user role."""
    response = client.post("/api/auth/login", json={
        "email": "demo@aura.ai",
        "password": "demo123"
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["user"]["role"] == "user"

def test_login_invalid_credentials(client):
    """Verifies that invalid password returns 401 Unauthorized."""
    response = client.post("/api/auth/login", json={
        "email": "admin@aura.ai",
        "password": "wrong_password_123"
    })
    assert response.status_code == 401
    data = json.loads(response.data)
    assert "error" in data

def test_google_auth_admin_account(client):
    """Verifies Google sign in for admin@aura.ai grants admin role."""
    response = client.post("/api/auth/google", json={
        "email": "admin@aura.ai",
        "name": "System Administrator"
    })
    assert response.status_code in [200, 201]
    data = json.loads(response.data)
    assert data["user"]["role"] == "admin"

def test_google_auth_regular_user(client):
    """Verifies Google sign in for external user assigns user role."""
    response = client.post("/api/auth/google", json={
        "email": "fashionista.test@gmail.com",
        "name": "Fashionista Test"
    })
    assert response.status_code in [200, 201]
    data = json.loads(response.data)
    assert data["user"]["role"] == "user"

def test_admin_panel_blocked_for_anonymous(client):
    """Verifies unauthenticated requests to /api/admin are rejected with 401."""
    response = client.get("/api/admin/stats")
    assert response.status_code == 401
    data = json.loads(response.data)
    assert data["code"] == "UNAUTHORIZED"

def test_admin_panel_blocked_for_buyer(client):
    """Verifies buyer account receives 403 Forbidden when accessing /api/admin."""
    response = client.get("/api/admin/stats", headers={
        "X-Admin-Email": "demo@aura.ai"
    })
    assert response.status_code == 403
    data = json.loads(response.data)
    assert data["code"] == "FORBIDDEN"

def test_admin_panel_allowed_for_admin(client):
    """Verifies admin user has full access to /api/admin/stats."""
    response = client.get("/api/admin/stats", headers={
        "X-Admin-Email": "admin@aura.ai"
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "total_products" in data["stats"]
    assert "admin_users" in data["stats"]

def test_admin_list_and_manage_users(client):
    """Verifies admin can list registered users."""
    response = client.get("/api/admin/users", headers={
        "X-Admin-Email": "admin@aura.ai"
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    emails = [u["email"] for u in data["users"]]
    assert "admin@aura.ai" in emails

def test_admin_add_product(client):
    """Verifies admin can add a product SKU to catalog."""
    test_sku = "TEST-SKU-999"
    response = client.post("/api/admin/products", headers={
        "X-Admin-Email": "admin@aura.ai"
    }, json={
        "sku": test_sku,
        "name": "Test Evening Gown",
        "category": "Dresses",
        "subcategory": "Gowns",
        "cost_price": 45.0,
        "retail_price": 120.0,
        "current_stock": 50
    })
    assert response.status_code in [201, 409]
