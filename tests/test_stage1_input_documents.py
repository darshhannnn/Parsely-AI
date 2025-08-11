"""
Unit tests for Stage 1: Input Documents processing
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import requests

from src.pipeline.stages.stage1_input_documents import (
    DocumentDownloader, DocumentFormatDetector, ContentExtractor,
    DocumentValidator, Stage1DocumentProcessor
)
from src.pipeline.core.models import DocumentContent, ExtractedContent
from src.pipeline.core.interfaces import DocumentType
from src.pipeline.core.exceptions import (
    DocumentDownloadError, UnsupportedFormatError, DocumentSizeError,
    ContentExtractionError, ValidationError, TimeoutError
)


class TestDocumentDownloader:
    """Test suite for DocumentDownloader"""
    
    @pytest.fixture
    def downloader(self):
        return DocumentDownloader(timeout_seconds=10, max_size_mb=5)
    
    def test_init(self, downloader):
        """Test downloader initialization"""
        assert downloader.timeout_seconds == 10
        assert downloader.max_size_mb == 5
        assert downloader.session is not None
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    def test_download_document_success(self, mock_get, mock_head, downloader):
        """Test successful document download"""
        # Mock HEAD response
        mock_head_response = Mock()
        mock_head_response.headers = {
            'content-length': '1024',
            'content-type': 'application/pdf'
        }
        mock_head.return_value = mock_head_response
        
        # Mock GET response
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'%PDF-1.4 test content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test download
        result = downloader.download_document('https://example.com/test.pdf')
        
        assert isinstance(result, DocumentContent)
        assert result.url == 'https://example.com/test.pdf'
        assert result.content_type == 'application/pdf'
        assert result.raw_content == b'%PDF-1.4 test content'
        assert result.size_bytes == len(b'%PDF-1.4 test content')
    
    def test_download_invalid_url(self, downloader):
        """Test download with invalid URL"""
        with pytest.raises(ValidationError):
            downloader.download_document('not-a-url')
    
    @patch('requests.Session.head')
    def test_download_size_limit_exceeded(self, mock_head, downloader):
        """Test download with size limit exceeded"""
        mock_head_response = Mock()
        mock_head_response.headers = {
            'content-length': str(10 * 1024 * 1024)  # 10MB > 5MB limit
        }
        mock_head.return_value = mock_head_response
        
        with pytest.raises(DocumentSizeError):
            downloader.download_document('https://example.com/large.pdf')
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    def test_download_timeout(self, mock_get, mock_head, downloader):
        """Test download timeout"""
        mock_head.return_value = Mock(headers={})
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(TimeoutError):
            downloader.download_document('https://example.com/test.pdf')
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    def test_download_http_error(self, mock_get, mock_head, downloader):
        """Test download HTTP error"""
        mock_head.return_value = Mock(headers={})
        
        mock_response = Mock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_get.side_effect = http_error
        
        with pytest.raises(DocumentDownloadError):
            downloader.download_document('https://example.com/notfound.pdf')
    
    def test_detect_content_type_from_content(self, downloader):
        """Test content type detection from file content"""
        # Test PDF detection
        pdf_content = b'%PDF-1.4 some content'
        assert downloader._detect_content_type_from_content(pdf_content, 'test.pdf') == 'application/pdf'
        
        # Test DOCX detection
        docx_content = b'PK\x03\x04' + b'word/' + b'x' * 100
        assert downloader._detect_content_type_from_content(docx_content, 'test.docx') == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        
        # Test email detection
        email_content = b'From: test@example.com\nTo: user@example.com\nSubject: Test'
        assert downloader._detect_content_type_from_content(email_content, 'test.eml') == 'message/rfc822'
    
    def test_extract_filename_from_url(self, downloader):
        """Test filename extraction from URL"""
        assert downloader._extract_filename_from_url('https://example.com/test.pdf') == 'test.pdf'
        assert downloader._extract_filename_from_url('https://example.com/path/document.docx') == 'document.docx'
        assert downloader._extract_filename_from_url('https://example.com/') == 'document'


class TestDocumentFormatDetector:
    """Test suite for DocumentFormatDetector"""
    
    @pytest.fixture
    def detector(self):
        return DocumentFormatDetector()
    
    def test_detect_format_pdf(self, detector):
        """Test PDF format detection"""
        document = DocumentContent(
            url='test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 content'
        )
        assert detector.detect_format(document) == DocumentType.PDF
    
    def test_detect_format_docx(self, detector):
        """Test DOCX format detection"""
        document = DocumentContent(
            url='test.docx',
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            raw_content=b'PK\x03\x04' + b'word/' + b'x' * 100
        )
        assert detector.detect_format(document) == DocumentType.DOCX
    
    def test_detect_format_email(self, detector):
        """Test email format detection"""
        document = DocumentContent(
            url='test.eml',
            content_type='message/rfc822',
            raw_content=b'From: test@example.com\nSubject: Test'
        )
        assert detector.detect_format(document) == DocumentType.EMAIL
    
    def test_detect_format_unsupported(self, detector):
        """Test unsupported format detection"""
        document = DocumentContent(
            url='test.txt',
            content_type='text/plain',
            raw_content=b'plain text content'
        )
        with pytest.raises(UnsupportedFormatError):
            detector.detect_format(document)
    
    def test_is_email_content(self, detector):
        """Test email content detection"""
        email_content = b'From: test@example.com\nTo: user@example.com\nSubject: Test'
        assert detector._is_email_content(email_content) == True
        
        non_email_content = b'This is just regular text content'
        assert detector._is_email_content(non_email_content) == False


class TestContentExtractor:
    """Test suite for ContentExtractor"""
    
    @pytest.fixture
    def extractor(self):
        return ContentExtractor()
    
    @patch('PyPDF2.PdfReader')
    @patch('builtins.open', create=True)
    def test_extract_pdf_content(self, mock_open, mock_pdf_reader, extractor):
        """Test PDF content extraction"""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page 1 content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {'Title': 'Test PDF'}
        mock_pdf_reader.return_value = mock_reader
        
        document = DocumentContent(
            url='test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 content'
        )
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.pdf'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor._extract_pdf_content(document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.PDF.value
        assert result.text_content == "Page 1 content"
        assert len(result.pages) == 1
        assert result.pages[0] == "Page 1 content"
        assert result.metadata['page_count'] == 1
    
    @patch('docx.Document')
    def test_extract_docx_content(self, mock_document, extractor):
        """Test DOCX content extraction"""
        # Mock paragraph
        mock_para = Mock()
        mock_para.text = "Test paragraph content"
        mock_para.style.name = "Normal"
        
        # Mock document
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        mock_document.return_value = mock_doc
        
        document = DocumentContent(
            url='test.docx',
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            raw_content=b'PK\x03\x04 docx content'
        )
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.docx'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor._extract_docx_content(document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.DOCX.value
        assert result.text_content == "Test paragraph content"
        assert result.metadata['paragraph_count'] == 1
    
    @patch('email.message_from_bytes')
    def test_extract_email_content(self, mock_email_parser, extractor):
        """Test email content extraction"""
        # Mock email message
        mock_msg = Mock()
        mock_msg.get.side_effect = lambda key, default='': {
            'Subject': 'Test Subject',
            'From': 'sender@example.com',
            'To': 'recipient@example.com',
            'Date': '2024-01-01'
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content.return_value = "Email body content"
        mock_msg.get_content_type.return_value = "text/plain"
        
        mock_email_parser.return_value = mock_msg
        
        document = DocumentContent(
            url='test.eml',
            content_type='message/rfc822',
            raw_content=b'From: sender@example.com\nSubject: Test Subject\n\nEmail body content'
        )
        
        result = extractor._extract_email_content(document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.EMAIL.value
        assert "Test Subject" in result.text_content
        assert "Email body content" in result.text_content
        assert result.metadata['headers']['subject'] == 'Test Subject'
    
    def test_extract_content_unsupported_format(self, extractor):
        """Test extraction with unsupported format"""
        document = DocumentContent(
            url='test.txt',
            content_type='text/plain',
            raw_content=b'plain text'
        )
        
        with pytest.raises(UnsupportedFormatError):
            extractor.extract_content(document, DocumentType.PDF)  # Wrong type


class TestDocumentValidator:
    """Test suite for DocumentValidator"""
    
    @pytest.fixture
    def validator(self):
        return DocumentValidator(min_content_length=10)
    
    def test_validate_document_success(self, validator):
        """Test successful document validation"""
        content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This is a valid document with sufficient content length.',
            pages=['Page 1 content'],
            metadata={'page_count': 1}
        )
        
        assert validator.validate_document(content) == True
    
    def test_validate_document_too_short(self, validator):
        """Test validation failure for short content"""
        content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='Short',  # Less than 10 characters
            pages=['Short'],
            metadata={'page_count': 1}
        )
        
        with pytest.raises(ValidationError):
            validator.validate_document(content)
    
    def test_validate_document_empty_content(self, validator):
        """Test validation failure for empty content"""
        content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='',
            pages=[],
            metadata={'page_count': 0}
        )
        
        with pytest.raises(ValidationError):
            validator.validate_document(content)
    
    def test_validate_document_unsupported_type(self, validator):
        """Test validation failure for unsupported document type"""
        content = ExtractedContent(
            document_id='test-id',
            document_type='unsupported',
            text_content='This is valid content with sufficient length.',
            metadata={}
        )
        
        with pytest.raises(ValidationError):
            validator.validate_document(content)
    
    def test_validate_pdf_no_pages(self, validator):
        """Test validation failure for PDF with no pages"""
        content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This is valid content with sufficient length.',
            pages=[],  # No pages
            metadata={'page_count': 0}
        )
        
        with pytest.raises(ValidationError):
            validator.validate_document(content)


class TestStage1DocumentProcessor:
    """Test suite for complete Stage 1 processor"""
    
    @pytest.fixture
    def processor(self):
        return Stage1DocumentProcessor(timeout_seconds=10, max_size_mb=5, min_content_length=10)
    
    def test_init(self, processor):
        """Test processor initialization"""
        assert processor.downloader is not None
        assert processor.format_detector is not None
        assert processor.content_extractor is not None
        assert processor.validator is not None
    
    @patch.object(Stage1DocumentProcessor, 'download_document')
    @patch.object(Stage1DocumentProcessor, 'extract_content')
    @patch.object(Stage1DocumentProcessor, 'validate_document')
    def test_process_document_url_success(self, mock_validate, mock_extract, mock_download, processor):
        """Test complete document processing success"""
        # Mock responses
        mock_document = DocumentContent(
            url='https://example.com/test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 content'
        )
        mock_download.return_value = mock_document
        
        mock_extracted = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This is extracted content from the PDF document.',
            pages=['Page 1 content'],
            metadata={'page_count': 1}
        )
        mock_extract.return_value = mock_extracted
        mock_validate.return_value = True
        
        # Test processing
        result = processor.process_document_url('https://example.com/test.pdf')
        
        assert result == mock_extracted
        mock_download.assert_called_once_with('https://example.com/test.pdf')
        mock_extract.assert_called_once_with(mock_document)
        mock_validate.assert_called_once_with(mock_extracted)
    
    @patch.object(Stage1DocumentProcessor, 'download_document')
    def test_process_document_url_download_failure(self, mock_download, processor):
        """Test processing failure during download"""
        mock_download.side_effect = DocumentDownloadError("Download failed", "https://example.com/test.pdf")
        
        with pytest.raises(DocumentDownloadError):
            processor.process_document_url('https://example.com/test.pdf')
    
    def test_detect_format(self, processor):
        """Test format detection"""
        pdf_content = b'%PDF-1.4 content'
        result = processor.detect_format(pdf_content)
        assert result == DocumentType.PDF


# Integration tests
class TestStage1Integration:
    """Integration tests for Stage 1 components"""
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PDF content for testing"""
        return b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n299\n%%EOF'
    
    @patch('requests.Session.head')
    @patch('requests.Session.get')
    def test_end_to_end_pdf_processing(self, mock_get, mock_head, sample_pdf_content):
        """Test end-to-end PDF processing"""
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {
            'content-length': str(len(sample_pdf_content)),
            'content-type': 'application/pdf'
        }
        mock_head.return_value = mock_head_response
        
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [sample_pdf_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Create processor and test
        processor = Stage1DocumentProcessor()
        
        # This would require actual PyPDF2 to work, so we'll mock it
        with patch('PyPDF2.PdfReader') as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Hello World"
            
            mock_reader = Mock()
            mock_reader.pages = [mock_page]
            mock_reader.metadata = {}
            mock_pdf_reader.return_value = mock_reader
            
            with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.pdf'):
                with patch('src.pipeline.core.utils.cleanup_temp_file'):
                    result = processor.process_document_url('https://example.com/test.pdf')
            
            assert isinstance(result, ExtractedContent)
            assert result.document_type == DocumentType.PDF.value
            assert "Hello World" in result.text_content
            assert len(result.pages) == 1


if __name__ == '__main__':
    pytest.main([__file__])