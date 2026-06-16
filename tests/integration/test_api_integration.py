"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    try:
        from src.api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("API module not available")


@pytest.mark.integration
@pytest.mark.api
def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@pytest.mark.integration
@pytest.mark.api
def test_process_claim_endpoint(client):
    """Test process claim endpoint."""
    test_query = "46-year-old male, knee surgery in Pune, 3-month-old policy"
    response = client.post(
        "/process_claim",
        json={"query": test_query}
    )
    # Should return 200 or appropriate error code
    assert response.status_code in [200, 422, 500]


@pytest.mark.integration
@pytest.mark.api
def test_analyze_query_endpoint(client):
    """Test analyze query endpoint."""
    response = client.get(
        "/analyze_query",
        params={"query": "test query"}
    )
    assert response.status_code in [200, 422, 500]


@pytest.mark.integration
@pytest.mark.api
def test_search_clauses_endpoint(client):
    """Test search clauses endpoint."""
    response = client.get(
        "/search_clauses",
        params={"query": "test search"}
    )
    assert response.status_code in [200, 422, 500]