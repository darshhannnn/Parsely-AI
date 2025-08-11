"""
Integration tests for multi-format content extraction
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline.stages.stage1_input_documents import ContentExtractor, Stage1DocumentProcessor
from src.pipeline.core.models import DocumentContent, ExtractedContent
from src.pipeline.core.interfaces import DocumentType
from src.pipeline.core.exceptions import ContentExtractionError, UnsupportedFormatError


class TestMultiFormatExtraction:
    """Test suite for multi-format content extraction"""
    
    @pytest.fixture
    def extractor(self):
        return ContentExtractor()
    
    @pytest.fixture
    def pdf_document(self):
        return DocumentContent(
            url='test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 sample PDF content'
        )
    
    @pytest.fixture
    def docx_document(self):
        return DocumentContent(
            url='test.docx',
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            raw_content=b'PK\x03\x04 sample DOCX content'
        )
    
    @pytest.fixture
    def email_document(self):
        email_content = b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000

This is a test email body.
"""
        return DocumentContent(
            url='test.eml',
            content_type='message/rfc822',
            raw_content=email_content
        )
    
    def test_extractor_initialization_with_enhanced(self, extractor):
        """Test extractor initialization with enhanced extractors"""
        # The extractor should try to load enhanced extractors
        assert hasattr(extractor, 'use_enhanced')
        
        # If enhanced extractors are available, they should be loaded
        if extractor.use_enhanced:
            assert hasattr(extractor, 'pdf_extractor')
            assert hasattr(extractor, 'docx_extractor')
            assert hasattr(extractor, 'email_extractor')
    
    def test_extractor_initialization_fallback(self):
        """Test extractor initialization with fallback to basic extraction"""
        with patch('src.pipeline.stages.stage1_input_documents.EnhancedPDFExtractor', side_effect=ImportError):
            extractor = ContentExtractor()
            assert extractor.use_enhanced == False
    
    @patch('src.pipeline.stages.extractors.pdf_extractor.EnhancedPDFExtractor.extract_content')
    def test_pdf_extraction_enhanced(self, mock_extract, extractor, pdf_document):
        """Test PDF extraction using enhanced extractor"""
        if not extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock enhanced extraction result
        mock_result = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='Enhanced PDF content extraction',
            pages=['Page 1 content'],
            sections={'page_1': 'Page 1 content'},
            metadata={
                'extraction_method': 'Enhanced PyPDF2',
                'page_count': 1,
                'structure_analysis': {'sections': [], 'headings': []}
            }
        )
        mock_extract.return_value = mock_result
        
        result = extractor.extract_content(pdf_document, DocumentType.PDF)
        
        assert result == mock_result
        assert result.metadata['extraction_method'] == 'Enhanced PyPDF2'
        mock_extract.assert_called_once_with(pdf_document)
    
    @patch('src.pipeline.stages.extractors.docx_extractor.EnhancedDOCXExtractor.extract_content')
    def test_docx_extraction_enhanced(self, mock_extract, extractor, docx_document):
        """Test DOCX extraction using enhanced extractor"""
        if not extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock enhanced extraction result
        mock_result = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.DOCX.value,
            text_content='Enhanced DOCX content extraction',
            sections={'Introduction': 'Introduction content'},
            metadata={
                'extraction_method': 'Enhanced python-docx',
                'paragraph_count': 5,
                'structure_analysis': {'sections': [], 'headings': []}
            }
        )
        mock_extract.return_value = mock_result
        
        result = extractor.extract_content(docx_document, DocumentType.DOCX)
        
        assert result == mock_result
        assert result.metadata['extraction_method'] == 'Enhanced python-docx'
        mock_extract.assert_called_once_with(docx_document)
    
    @patch('src.pipeline.stages.extractors.email_extractor.EnhancedEmailExtractor.extract_content')
    def test_email_extraction_enhanced(self, mock_extract, extractor, email_document):
        """Test email extraction using enhanced extractor"""
        if not extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock enhanced extraction result
        mock_result = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.EMAIL.value,
            text_content='Enhanced email content extraction',
            sections={'headers': 'Subject: Test', 'body': 'Email body'},
            metadata={
                'extraction_method': 'Enhanced email.parser',
                'headers': {'subject': 'Test Email'},
                'structure': {'is_multipart': False},
                'content_analysis': {'word_count': 10}
            }
        )
        mock_extract.return_value = mock_result
        
        result = extractor.extract_content(email_document, DocumentType.EMAIL)
        
        assert result == mock_result
        assert result.metadata['extraction_method'] == 'Enhanced email.parser'
        mock_extract.assert_called_once_with(email_document)
    
    @patch('PyPDF2.PdfReader')
    @patch('builtins.open', create=True)
    def test_pdf_extraction_fallback(self, mock_open, mock_pdf_reader, pdf_document):
        """Test PDF extraction using fallback basic extractor"""
        # Create extractor without enhanced extractors
        with patch('src.pipeline.stages.stage1_input_documents.EnhancedPDFExtractor', side_effect=ImportError):
            extractor = ContentExtractor()
        
        # Mock PDF reader for fallback
        mock_page = Mock()
        mock_page.extract_text.return_value = "Basic PDF content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {}
        mock_pdf_reader.return_value = mock_reader
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.pdf'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor.extract_content(pdf_document, DocumentType.PDF)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.PDF.value
        assert "Basic PDF content" in result.text_content
        assert result.metadata['extraction_method'] == 'PyPDF2'
    
    @patch('docx.Document')
    def test_docx_extraction_fallback(self, mock_document, docx_document):
        """Test DOCX extraction using fallback basic extractor"""
        # Create extractor without enhanced extractors
        with patch('src.pipeline.stages.stage1_input_documents.EnhancedDOCXExtractor', side_effect=ImportError):
            extractor = ContentExtractor()
        
        # Mock document for fallback
        mock_para = Mock()
        mock_para.text = "Basic DOCX content"
        mock_para.style.name = "Normal"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        mock_document.return_value = mock_doc
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.docx'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor.extract_content(docx_document, DocumentType.DOCX)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.DOCX.value
        assert "Basic DOCX content" in result.text_content
        assert result.metadata['extraction_method'] == 'python-docx'
    
    @patch('email.message_from_bytes')
    def test_email_extraction_fallback(self, mock_email_parser, email_document):
        """Test email extraction using fallback basic extractor"""
        # Create extractor without enhanced extractors
        with patch('src.pipeline.stages.stage1_input_documents.EnhancedEmailExtractor', side_effect=ImportError):
            extractor = ContentExtractor()
        
        # Mock email message for fallback
        mock_msg = Mock()
        mock_msg.get.side_effect = lambda key, default='': {
            'Subject': 'Basic Email',
            'From': 'sender@example.com',
            'To': 'recipient@example.com',
            'Date': '2024-01-01'
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content.return_value = "Basic email content"
        mock_msg.get_content_type.return_value = "text/plain"
        
        mock_email_parser.return_value = mock_msg
        
        result = extractor.extract_content(email_document, DocumentType.EMAIL)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.EMAIL.value
        assert "Basic Email" in result.text_content
        assert result.metadata['extraction_method'] == 'email.parser'
    
    def test_unsupported_format_error(self, extractor):
        """Test error handling for unsupported formats"""
        unsupported_document = DocumentContent(
            url='test.txt',
            content_type='text/plain',
            raw_content=b'plain text content'
        )
        
        with pytest.raises(UnsupportedFormatError):
            extractor.extract_content(unsupported_document, DocumentType.PDF)  # Wrong type
    
    def test_extraction_error_handling(self, extractor, pdf_document):
        """Test error handling during extraction"""
        with patch('PyPDF2.PdfReader', side_effect=Exception("Extraction failed")):
            with pytest.raises(ContentExtractionError):
                extractor.extract_content(pdf_document, DocumentType.PDF)


class TestStage1MultiFormatIntegration:
    """Integration tests for Stage 1 multi-format processing"""
    
    @pytest.fixture
    def processor(self):
        return Stage1DocumentProcessor()
    
    def test_processor_uses_enhanced_extractor(self, processor):
        """Test that processor uses enhanced content extractor"""
        assert isinstance(processor.content_extractor, ContentExtractor)
        # The extractor should attempt to use enhanced extractors
        assert hasattr(processor.content_extractor, 'use_enhanced')
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    @patch('src.pipeline.stages.extractors.pdf_extractor.EnhancedPDFExtractor.extract_content')
    def test_end_to_end_pdf_processing_enhanced(self, mock_extract, mock_get, mock_head, processor):
        """Test end-to-end PDF processing with enhanced extractor"""
        if not processor.content_extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-type': 'application/pdf', 'content-length': '1024'}
        mock_head.return_value = mock_head_response
        
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'%PDF-1.4 content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock enhanced extraction
        mock_extracted = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='Enhanced PDF processing result',
            pages=['Page 1'],
            sections={'page_1': 'Page 1'},
            metadata={
                'extraction_method': 'Enhanced PyPDF2',
                'page_count': 1,
                'structure_analysis': {'sections': [], 'headings': []}
            }
        )
        mock_extract.return_value = mock_extracted
        
        result = processor.process_document_url('https://example.com/test.pdf')
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.PDF.value
        assert result.metadata['extraction_method'] == 'Enhanced PyPDF2'
        assert 'structure_analysis' in result.metadata
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    @patch('src.pipeline.stages.extractors.docx_extractor.EnhancedDOCXExtractor.extract_content')
    def test_end_to_end_docx_processing_enhanced(self, mock_extract, mock_get, mock_head, processor):
        """Test end-to-end DOCX processing with enhanced extractor"""
        if not processor.content_extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {
            'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'content-length': '2048'
        }
        mock_head.return_value = mock_head_response
        
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'PK\x03\x04 docx content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock enhanced extraction
        mock_extracted = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.DOCX.value,
            text_content='Enhanced DOCX processing result',
            sections={'Introduction': 'Introduction content'},
            metadata={
                'extraction_method': 'Enhanced python-docx',
                'paragraph_count': 5,
                'structure_analysis': {'sections': [], 'headings': []}
            }
        )
        mock_extract.return_value = mock_extracted
        
        result = processor.process_document_url('https://example.com/test.docx')
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.DOCX.value
        assert result.metadata['extraction_method'] == 'Enhanced python-docx'
        assert 'structure_analysis' in result.metadata
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    @patch('src.pipeline.stages.extractors.email_extractor.EnhancedEmailExtractor.extract_content')
    def test_end_to_end_email_processing_enhanced(self, mock_extract, mock_get, mock_head, processor):
        """Test end-to-end email processing with enhanced extractor"""
        if not processor.content_extractor.use_enhanced:
            pytest.skip("Enhanced extractors not available")
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-type': 'message/rfc822', 'content-length': '512'}
        mock_head.return_value = mock_head_response
        
        email_content = b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email

Email body content."""
        
        mock_response = Mock()
        mock_response.headers = {'content-type': 'message/rfc822'}
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [email_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock enhanced extraction
        mock_extracted = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.EMAIL.value,
            text_content='Enhanced email processing result',
            sections={'headers': 'Subject: Test', 'body': 'Email body'},
            metadata={
                'extraction_method': 'Enhanced email.parser',
                'headers': {'subject': 'Test Email'},
                'structure': {'is_multipart': False},
                'content_analysis': {'word_count': 10}
            }
        )
        mock_extract.return_value = mock_extracted
        
        result = processor.process_document_url('https://example.com/test.eml')
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.EMAIL.value
        assert result.metadata['extraction_method'] == 'Enhanced email.parser'
        assert 'content_analysis' in result.metadata
    
    def test_format_detection_accuracy(self, processor):
        """Test format detection accuracy for different content types"""
        # Test PDF detection
        pdf_content = b'%PDF-1.4 content'
        pdf_format = processor.detect_format(pdf_content)
        assert pdf_format == DocumentType.PDF
        
        # Test DOCX detection
        docx_content = b'PK\x03\x04' + b'word/' + b'x' * 100
        docx_format = processor.detect_format(docx_content)
        assert docx_format == DocumentType.DOCX
        
        # Test email detection
        email_content = b'From: test@example.com\nSubject: Test\n\nBody'
        email_format = processor.detect_format(email_content)
        assert email_format == DocumentType.EMAIL


class TestMultiFormatPerformance:
    """Performance tests for multi-format extraction"""
    
    @pytest.fixture
    def extractor(self):
        return ContentExtractor()
    
    def test_extraction_performance_tracking(self, extractor):
        """Test that extraction performance is tracked"""
        # The timing_decorator should be applied to extract_content method
        assert hasattr(extractor.extract_content, '__wrapped__')
    
    def test_concurrent_extraction_support(self, extractor):
        """Test that extractor supports concurrent operations"""
        # Each extractor instance should be independent
        extractor1 = ContentExtractor()
        extractor2 = ContentExtractor()
        
        assert extractor1 is not extractor2
        assert extractor1.logger is not extractor2.logger


if __name__ == '__main__':
    pytest.main([__file__])