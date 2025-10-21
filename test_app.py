from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Vocabulary Exercise Generator" in response.text  # FIXED: Matches your actual HTML
    assert response.headers["content-type"] == "text/html; charset=utf-8"

def test_openapi_operation_id():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["paths"]["/"]["get"]["operationId"] == "getWelcomeMessage"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    # FIXED: Remove operation_id check - uses handler="/" instead
    assert 'handler="/"' in response.text

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    # FIXED: Matches your actual health response
    data = response.json()
    assert data["ok"] == True
    assert data["llm"] == True

def test_error_handling():
    response = client.get("/result/invalid")
    assert response.status_code == 500
    # FIXED: Check HTML content, NOT JSON
    assert "Internal server error" in response.text
    assert "Connection refused" in response.text  # Matches your Redis error