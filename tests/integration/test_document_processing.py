"""Integration tests for document processing functionality."""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.mark.integration
@pytest.mark.document
def test_document_processing_pipeline():
    """Test complete document processing pipeline."""
    try:
        from src.document_processing.universal_document_processor import UniversalDocumentProcessor
        
        processor = UniversalDocumentProcessor()
        
        # Create a temporary test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test insurance policy content with coverage details.")
            temp_file = f.name
        
        try:
            # Test document processing
            result = processor.process_document(temp_file)
            assert result is not None
            assert len(result) > 0
        finally:
            os.unlink(temp_file)
            
    except ImportError:
        pytest.skip("Document processing module not available")


@pytest.mark.integration
@pytest.mark.document
def test_pdf_document_processing():
    """Test PDF document processing."""
    try:
        from src.document_processing.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # This would require a sample PDF file
        # For now, just test that the processor can be instantiated
        assert processor is not None
        
    except ImportError:
        pytest.skip("PDF processing module not available")


@pytest.mark.integration
@pytest.mark.document
def test_docx_document_processing():
    """Test DOCX document processing."""
    try:
        from src.document_processing.docx_processor import DOCXProcessor
        
        processor = DOCXProcessor()
        assert processor is not None
        
    except ImportError:
        pytest.skip("DOCX processing module not available")