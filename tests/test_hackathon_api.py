"""
Unit tests for the hackathon API endpoint
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the hackathon API
from src.api.hackathon_main import app, verify_bearer_token, download_pdf_from_blob_url, process_document_and_questions

client = TestClient(app)

class TestHackathonAPI:
    """Test class for hackathon API endpoints"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.valid_token = "8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63"
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
    
    def test_authentication_valid_token(self):
        """Test authentication with valid token"""
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        
        with patch('src.api.hackathon_main.download_pdf_from_blob_url') as mock_download, \
             patch('src.api.hackathon_main.process_document_and_questions') as mock_process:
            
            mock_download.return_value = "/tmp/test.pdf"
            mock_process.return_value = ["Answer 1", "Answer 2"]
            
            response = client.post("/hackrx/run", json=self.test_request, headers=headers)
            assert response.status_code == 200
    
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
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        invalid_request = {"questions": ["Test question"]}
        
        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_request_validation_missing_questions(self):
        """Test request validation with missing questions field"""
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        invalid_request = {"documents": "https://example.com/test.pdf"}
        
        response = client.post("/hackrx/run", json=invalid_request, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_request_validation_empty_questions(self):
        """Test request validation with empty questions list"""
        headers = {"Authorization": f"Bearer {self.valid_token}"}
        invalid_request = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }
        
        with patch('src.api.hackathon_main.download_pdf_from_blob_url') as mock_download, \
             patch('src.api.hackathon_main.process_document_and_questions') as mock_process:
            
            mock_download.return_value = "/tmp/test.pdf"
            mock_process.return_value = []
            
            response = client.post("/hackrx/run", json=invalid_request, headers=headers)
            assert response.status_code == 200
            assert response.json()["answers"] == []

class TestDownloadPDFFromBlobURL:
    """Test class for PDF download functionality"""
    
    @patch('src.api.hackathon_main.requests.get')
    def test_successful_download(self, mock_get):
        """Test successful PDF download"""
        # Mock successful response
        mock_response = Mock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        blob_url = "https://example.com/test.pdf"
        result = download_pdf_from_blob_url(blob_url)
        
        assert result is not None
        assert result.endswith('.pdf')
        assert os.path.exists(result)
        
        # Clean up
        os.unlink(result)
    
    @patch('src.api.hackathon_main.requests.get')
    def test_download_request_exception(self, mock_get):
        """Test download with request exception"""
        mock_get.side_effect = Exception("Network error")
        
        blob_url = "https://example.com/test.pdf"
        
        with pytest.raises(HTTPException) as exc_info:
            download_pdf_from_blob_url(blob_url)
        
        assert exc_info.value.status_code == 500
        assert "Unexpected error downloading PDF" in str(exc_info.value.detail)
    
    @patch('src.api.hackathon_main.requests.get')
    def test_download_http_error(self, mock_get):
        """Test download with HTTP error"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response
        
        blob_url = "https://example.com/test.pdf"
        
        with pytest.raises(HTTPException) as exc_info:
            download_pdf_from_blob_url(blob_url)
        
        assert exc_info.value.status_code == 400
        assert "Failed to download PDF from blob URL" in str(exc_info.value.detail)

class TestProcessDocumentAndQuestions:
    """Test class for document processing functionality"""
    
    @patch('src.api.hackathon_main.InsuranceClaimProcessor')
    @patch('src.api.hackathon_main.DocumentProcessor')
    def test_successful_processing(self, mock_doc_processor_class, mock_processor_class):
        """Test successful document and question processing"""
        # Mock the processors
        mock_processor = Mock()
        mock_doc_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_doc_processor_class.return_value = mock_doc_processor
        
        # Mock the processing results
        mock_doc_processor.process_document.return_value = {"content": "test content"}
        mock_processor.process_claim.return_value = {
            "decision": "approved",
            "justification": "Test justification"
        }
        
        pdf_path = "/tmp/test.pdf"
        questions = ["Test question 1", "Test question 2"]
        
        result = process_document_and_questions(pdf_path, questions)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(answer, str) for answer in result)
    
    @patch('src.api.hackathon_main.InsuranceClaimProcessor')
    def test_processing_with_exception(self, mock_processor_class):
        """Test document processing with exception"""
        mock_processor_class.side_effect = Exception("Processing error")
        
        pdf_path = "/tmp/test.pdf"
        questions = ["Test question"]
        
        with pytest.raises(HTTPException) as exc_info:
            process_document_and_questions(pdf_path, questions)
        
        assert exc_info.value.status_code == 500
        assert "Error processing document and questions" in str(exc_info.value.detail)

class TestIntegration:
    """Integration tests for the complete API workflow"""
    
    @patch('src.api.hackathon_main.download_pdf_from_blob_url')
    @patch('src.api.hackathon_main.process_document_and_questions')
    @patch('os.unlink')
    def test_complete_workflow(self, mock_unlink, mock_process, mock_download):
        """Test the complete API workflow"""
        # Setup mocks
        mock_download.return_value = "/tmp/test.pdf"
        mock_process.return_value = ["Answer 1", "Answer 2"]
        mock_unlink.return_value = None
        
        headers = {"Authorization": f"Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63"}
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
        mock_process.assert_called_once_with("/tmp/test.pdf", ["Question 1", "Question 2"])
        mock_unlink.assert_called_once_with("/tmp/test.pdf")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])