import pytest
from datetime import datetime
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.pipeline.core.models import JSONResponse as PipelineResponse, ProcessingMetadata

client = TestClient(app)


def _make_response(success: bool = True, error_message: str = None) -> PipelineResponse:
    """Build a pipeline JSONResponse for mocking"""
    now = datetime.now()
    return PipelineResponse(
        success=success,
        processing_id="test-processing-id",
        timestamp=now,
        document_info={"url": "https://example.com/doc.pdf"},
        results=[{"query": "What is the grace period?", "answer": "Thirty days."}],
        metadata=ProcessingMetadata(correlation_id="test-processing-id", start_time=now),
        error_message=error_message,
    )


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "stages" in data
    assert "version" in data


def test_formats():
    res = client.get("/formats")
    assert res.status_code == 200
    formats = res.json()["supported_formats"]
    assert "pdf" in formats
    assert "docx" in formats


def test_process_document_success():
    payload = {
        "url": "https://example.com/doc.pdf",
        "questions": ["What is the grace period?"],
    }
    with patch("src.api.main.pipeline.process_document", return_value=_make_response()):
        res = client.post("/process", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["answer"] == "Thirty days."


def test_process_document_failure():
    payload = {
        "url": "https://example.com/doc.pdf",
        "questions": ["What is the grace period?"],
    }
    with patch(
        "src.api.main.pipeline.process_document",
        return_value=_make_response(success=False, error_message="Download failed"),
    ):
        res = client.post("/process", json=payload)
    assert res.status_code == 500
    assert "Download failed" in res.json()["detail"]


def test_process_document_validation_error():
    # Missing required fields must yield a validation error
    res = client.post("/process", json={"url": "https://example.com/doc.pdf"})
    assert res.status_code == 422
