from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to Distribute Word" in response.text

def test_openapi_operation_id():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["paths"]["/"]["get"]["operationId"] == "getWelcomeMessage"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert 'operation_id="getWelcomeMessage"' in response.text

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_error_handling():
    response = client.get("/result/invalid")
    assert response.status_code == 404
    assert response.json() == {"message": "Job not found"}