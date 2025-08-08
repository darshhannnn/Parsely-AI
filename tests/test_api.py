import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_process_claim():
    payload = {"query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy", "return_json": True}
    res = client.post("/process_claim", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "decision" in data
    assert "mapped_clauses" in data

def test_analyze_query():
    res = client.get("/analyze_query", params={"query": "knee surgery in Pune"})
    assert res.status_code == 200
    assert "extracted_entities" in res.json()

def test_search_clauses():
    res = client.get("/search_clauses", params={"query": "knee surgery in Pune", "top_k": 3})
    assert res.status_code == 200
    assert "clauses" in res.json()

def test_policy_summary():
    res = client.get("/policy_summary")
    assert res.status_code == 200
    assert "policy_breakdown" in res.json()
