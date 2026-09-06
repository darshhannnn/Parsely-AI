"""
Unit tests for the hackathon API endpoint
"""

import os
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.hackathon_main import (
    app,
    verify_bearer_token,
    download_document_from_blob_url,
    process_document_and_questions,
)

client = TestClient(app)

VALID_TOKEN = "8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63"


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    """Ensure the auth dependency has a token to compare against."""
    monkeypatch.setattr("src.api.hackathon_main.EXPECTED_TOKEN", VALID_TOKEN)


class TestHackathonAPI:
    """Test class for hackathon API endpoints"""

    def setup_method(self):
        """Setup for each test method"""
        self.test_request = {
            "documents": "https://example.com/test.pdf",
            "questions": [
                "What is the grace period?",
                "What are the coverage limits?"
            ]
        }

    def test_health_endpoint(self):
        """Test the health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data

    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "main_endpoint" in data
        assert data["main_endpoint"] == "/hackrx/run"
        assert data["web_interface"] == "/ui"

    def test_web_interface_served(self):
        """Test the browser testing interface is served at /ui"""
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "LLM Document Processing" in response.text

    def test_authentication_valid_token(self):
        """Test authentication with valid token"""
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}

        with patch('src.api.hackathon_main.download_document_from_blob_url') as mock_download, \
             patch('src.api.hackathon_main.process_document_and_questions') as mock_process:

            mock_download.return_value = ("/tmp/test.pdf", "pdf")
            mock_process.return_value = ["Answer 1", "Answer 2"]

            response = client.post("/hackrx/run", json=self.test_request, headers=headers)
            assert response.status_code == 200
            assert response.json()["answers"] == ["Answer 1", "Answer 2"]

    def test_authentication_invalid_token(self):
        """Test authentication with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}

        response = client.post("/hackrx/run", json=self.test_request, headers=headers)
        assert response.status_code == 401
        assert "Invalid bearer token" in response.json()["detail"]

    def test_authentication_missing_token(self):
        """Test authentication with missing token"""
        response = client.post("/hackrx/run", json=self.test_request)
        assert response.status_code == 403  # FastAPI returns 403 for missing auth

    def test_request_validation_missing_documents(self):
        """Test request validation with missing documents field"""
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        invalid_request = {"questions": ["Test question"]}

        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422  # Validation error

    def test_request_validation_missing_questions(self):
        """Test request validation with missing questions field"""
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        invalid_request = {"documents": "https://example.com/test.pdf"}

        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422  # Validation error

    def test_request_validation_empty_questions(self):
        """Test request validation with empty questions list (min 1 required)"""
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        invalid_request = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }

        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422  # min_length=1 on questions

    def test_request_validation_whitespace_question(self):
        """Test request validation rejects whitespace-only questions"""
        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        invalid_request = {
            "documents": "https://example.com/test.pdf",
            "questions": ["   "]
        }

        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422


class TestDownloadDocumentFromBlobURL:
    """Test class for document download functionality"""

    @patch('src.api.hackathon_main.requests.get')
    def test_successful_download(self, mock_get):
        """Test successful document download"""
        mock_response = Mock()
        mock_response.content = b"%PDF-1.4 mock pdf content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        blob_url = "https://example.com/test.pdf"
        temp_file_path, document_format = download_document_from_blob_url(blob_url)

        assert temp_file_path is not None
        assert temp_file_path.endswith('.pdf')
        assert document_format == 'pdf'
        assert os.path.exists(temp_file_path)

        # Clean up
        os.unlink(temp_file_path)

    @patch('src.api.hackathon_main.requests.get')
    def test_download_request_exception(self, mock_get):
        """Test download with request exception"""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        blob_url = "https://example.com/test.pdf"

        with pytest.raises(HTTPException) as exc_info:
            download_document_from_blob_url(blob_url)

        assert exc_info.value.status_code == 400
        assert "Failed to download document from blob URL" in str(exc_info.value.detail)

    @patch('src.api.hackathon_main.requests.get')
    def test_download_http_error(self, mock_get):
        """Test download with HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP 404")
        mock_get.return_value = mock_response

        blob_url = "https://example.com/test.pdf"

        with pytest.raises(HTTPException) as exc_info:
            download_document_from_blob_url(blob_url)

        assert exc_info.value.status_code == 400
        assert "Failed to download document from blob URL" in str(exc_info.value.detail)

    @patch('src.api.hackathon_main.requests.get')
    def test_download_unexpected_error(self, mock_get):
        """Test download with unexpected error"""
        mock_get.side_effect = Exception("Unexpected failure")

        blob_url = "https://example.com/test.pdf"

        with pytest.raises(HTTPException) as exc_info:
            download_document_from_blob_url(blob_url)

        assert exc_info.value.status_code == 500
        assert "Unexpected error downloading document" in str(exc_info.value.detail)


class TestProcessDocumentAndQuestions:
    """Test class for document processing functionality"""

    @patch('google.generativeai.GenerativeModel')
    @patch('src.api.hackathon_main.extract_pdf_content_enhanced')
    def test_successful_processing(self, mock_extract, mock_genai_model, monkeypatch):
        """Test successful document and question processing"""
        monkeypatch.setattr("src.api.hackathon_main.GOOGLE_API_KEY", "test-key")

        mock_extract.return_value = ("Sample policy document text.", {"sections": {}})

        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Approved based on coverage."
        mock_genai_model.return_value = mock_model

        pdf_path = "/tmp/test.pdf"
        questions = ["Test question 1", "Test question 2"]

        result = process_document_and_questions(pdf_path, "pdf", questions)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(answer, str) for answer in result)
        assert all(a == "Approved based on coverage." for a in result)
        assert mock_model.generate_content.call_count == 2

    @patch('google.generativeai.GenerativeModel')
    @patch('src.api.hackathon_main.extract_pdf_content_enhanced')
    def test_long_answer_truncation(self, mock_extract, mock_genai_model, monkeypatch):
        """Test answers longer than 500 characters are truncated"""
        monkeypatch.setattr("src.api.hackathon_main.GOOGLE_API_KEY", "test-key")

        mock_extract.return_value = ("Sample policy document text.", {"sections": {}})

        mock_model = Mock()
        mock_model.generate_content.return_value.text = "x" * 600
        mock_genai_model.return_value = mock_model

        result = process_document_and_questions("/tmp/test.pdf", "pdf", ["Question?"])

        assert len(result) == 1
        assert len(result[0]) == 500
        assert result[0].endswith("...")

    @patch('src.api.hackathon_main.extract_pdf_content_enhanced')
    def test_missing_api_key(self, mock_extract, monkeypatch):
        """Test processing without a configured API key"""
        monkeypatch.setattr("src.api.hackathon_main.GOOGLE_API_KEY", None)

        with pytest.raises(HTTPException) as exc_info:
            process_document_and_questions("/tmp/test.pdf", "pdf", ["Question?"])

        assert exc_info.value.status_code == 500
        assert "GOOGLE_API_KEY not configured" in str(exc_info.value.detail)

    @patch('src.api.hackathon_main.extract_pdf_content_enhanced')
    def test_extraction_failure_returns_error_answers(self, mock_extract, monkeypatch):
        """Test extraction failure yields per-question error answers"""
        monkeypatch.setattr("src.api.hackathon_main.GOOGLE_API_KEY", "test-key")
        mock_extract.side_effect = Exception("bad pdf")

        result = process_document_and_questions("/tmp/test.pdf", "pdf", ["Q1", "Q2"])

        assert len(result) == 2
        assert all("Unable to process pdf document" in answer for answer in result)


class TestIntegration:
    """Integration tests for the complete API workflow"""

    @patch('src.api.hackathon_main.download_document_from_blob_url')
    @patch('src.api.hackathon_main.process_document_and_questions')
    @patch('os.unlink')
    def test_complete_workflow(self, mock_unlink, mock_process, mock_download):
        """Test the complete API workflow"""
        mock_download.return_value = ("/tmp/test.pdf", "pdf")
        mock_process.return_value = ["Answer 1", "Answer 2"]
        mock_unlink.return_value = None

        headers = {"Authorization": f"Bearer {VALID_TOKEN}"}
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["Question 1", "Question 2"]
        }

        response = client.post("/hackrx/run", json=request_data, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == 2
        assert data["answers"] == ["Answer 1", "Answer 2"]

        # Verify mocks were called
        mock_download.assert_called_once_with("https://example.com/test.pdf")
        mock_process.assert_called_once_with("/tmp/test.pdf", "pdf", ["Question 1", "Question 2"])
        mock_unlink.assert_called_once_with("/tmp/test.pdf")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
