"""
Unit tests for enhanced document extractors
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline.stages.extractors.pdf_extractor import EnhancedPDFExtractor, PDFPageInfo, PDFStructure
from src.pipeline.stages.extractors.docx_extractor import EnhancedDOCXExtractor, DOCXParagraph, DOCXStructure
from src.pipeline.stages.extractors.email_extractor import EnhancedEmailExtractor, EmailHeader, EmailStructure
from src.pipeline.core.models import DocumentContent, ExtractedContent
from src.pipeline.core.interfaces import DocumentType
from src.pipeline.core.exceptions import ContentExtractionError


class TestEnhancedPDFExtractor:
    """Test suite for Enhanced PDF Extractor"""
    
    @pytest.fixture
    def extractor(self):
        return EnhancedPDFExtractor()
    
    @pytest.fixture
    def sample_document(self):
        return DocumentContent(
            url='test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 sample content'
        )
    
    @patch('PyPDF2.PdfReader')
    @patch('builtins.open', create=True)
    def test_extract_content_success(self, mock_open, mock_pdf_reader, extractor, sample_document):
        """Test successful PDF content extraction"""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content with headings and sections."
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {'Title': 'Test PDF', 'Author': 'Test Author'}
        mock_pdf_reader.return_value = mock_reader
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.pdf'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor.extract_content(sample_document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.PDF.value
        assert "Sample PDF content" in result.text_content
        assert len(result.pages) == 1
        assert result.metadata['page_count'] == 1
        assert result.metadata['extraction_method'] == 'Enhanced PyPDF2'
        assert 'structure_analysis' in result.metadata
        assert 'pages_info' in result.metadata
    
    def test_extract_pages_with_analysis(self, extractor):
        """Test page analysis functionality"""
        mock_page = Mock()
        mock_page.extract_text.return_value = "1. Introduction\nThis is a test document with tables and content.\n\nTable data: Value1    Value2    Value3"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        
        pages_info = extractor._extract_pages_with_analysis(mock_reader)
        
        assert len(pages_info) == 1
        assert isinstance(pages_info[0], PDFPageInfo)
        assert pages_info[0].page_number == 1
        assert pages_info[0].word_count > 0
        assert pages_info[0].char_count > 0
    
    def test_detect_tables_in_text(self, extractor):
        """Test table detection in text"""
        # Text with table-like structure
        table_text = "Name     Age     City\nJohn     25      NYC\nJane     30      LA"
        assert extractor._detect_tables_in_text(table_text) == True
        
        # Regular text
        regular_text = "This is just regular paragraph text without any table structure."
        assert extractor._detect_tables_in_text(regular_text) == False
    
    def test_identify_sections(self, extractor):
        """Test section identification"""
        pages_info = [
            PDFPageInfo(1, "1. Introduction\nThis is the introduction section.\n\n2. Methods\nThis describes the methods.", 10, 50),
            PDFPageInfo(2, "3. Results\nThese are the results.\n\nConclusion\nThis is the conclusion.", 8, 40)
        ]
        
        sections = extractor._identify_sections(pages_info)
        
        assert len(sections) >= 2
        assert any('Introduction' in s['title'] for s in sections)
        assert any('Methods' in s['title'] for s in sections)
    
    def test_identify_headings(self, extractor):
        """Test heading identification"""
        pages_info = [
            PDFPageInfo(1, "1. INTRODUCTION\nContent here\n\n2. METHODOLOGY\nMore content", 10, 50)
        ]
        
        headings = extractor._identify_headings(pages_info)
        
        assert len(headings) >= 2
        assert any('INTRODUCTION' in h['text'] for h in headings)
        assert any('METHODOLOGY' in h['text'] for h in headings)


class TestEnhancedDOCXExtractor:
    """Test suite for Enhanced DOCX Extractor"""
    
    @pytest.fixture
    def extractor(self):
        return EnhancedDOCXExtractor()
    
    @pytest.fixture
    def sample_document(self):
        return DocumentContent(
            url='test.docx',
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            raw_content=b'PK\x03\x04 docx content'
        )
    
    @patch('docx.Document')
    def test_extract_content_success(self, mock_document, extractor, sample_document):
        """Test successful DOCX content extraction"""
        # Mock paragraphs
        mock_para1 = Mock()
        mock_para1.text = "Introduction"
        mock_para1.style.name = "Heading 1"
        
        mock_para2 = Mock()
        mock_para2.text = "This is the introduction content."
        mock_para2.style.name = "Normal"
        
        # Mock document
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        
        # Mock core properties
        mock_props = Mock()
        mock_props.title = "Test Document"
        mock_props.author = "Test Author"
        mock_props.subject = "Test Subject"
        mock_props.keywords = "test, document"
        mock_props.category = "Test"
        mock_props.comments = "Test comments"
        mock_props.created = datetime.now()
        mock_props.modified = datetime.now()
        mock_props.last_modified_by = "Test User"
        mock_props.revision = 1
        mock_props.version = "1.0"
        
        mock_doc.core_properties = mock_props
        mock_document.return_value = mock_doc
        
        with patch('src.pipeline.core.utils.create_temp_file', return_value='/tmp/test.docx'):
            with patch('src.pipeline.core.utils.cleanup_temp_file'):
                result = extractor.extract_content(sample_document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.DOCX.value
        assert "Introduction" in result.text_content
        assert result.metadata['extraction_method'] == 'Enhanced python-docx'
        assert 'structure_analysis' in result.metadata
        assert 'document_properties' in result.metadata
    
    def test_extract_paragraphs_with_analysis(self, extractor):
        """Test paragraph analysis"""
        # Mock paragraph
        mock_para = Mock()
        mock_para.text = "This is a test paragraph"
        mock_para.style.name = "Normal"
        mock_para.runs = []
        
        # Mock document
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para]
        
        paragraphs_info = extractor._extract_paragraphs_with_analysis(mock_doc)
        
        assert len(paragraphs_info) == 1
        assert isinstance(paragraphs_info[0], DOCXParagraph)
        assert paragraphs_info[0].text == "This is a test paragraph"
        assert paragraphs_info[0].style_name == "Normal"
    
    def test_extract_heading_level(self, extractor):
        """Test heading level extraction"""
        assert extractor._extract_heading_level("Heading 1") == 1
        assert extractor._extract_heading_level("Heading 2") == 2
        assert extractor._extract_heading_level("Heading 3") == 3
        assert extractor._extract_heading_level("Normal") == 0
    
    def test_build_sections(self, extractor):
        """Test section building from paragraphs"""
        paragraphs = [
            DOCXParagraph("Introduction", "Heading 1", True, 1, False, False),
            DOCXParagraph("This is the introduction.", "Normal", False, 0, False, False),
            DOCXParagraph("Methods", "Heading 1", True, 1, False, False),
            DOCXParagraph("This describes the methods.", "Normal", False, 0, False, False)
        ]
        
        sections = extractor._build_sections(paragraphs)
        
        assert len(sections) >= 2
        assert any(s.title == "Introduction" for s in sections)
        assert any(s.title == "Methods" for s in sections)


class TestEnhancedEmailExtractor:
    """Test suite for Enhanced Email Extractor"""
    
    @pytest.fixture
    def extractor(self):
        return EnhancedEmailExtractor()
    
    @pytest.fixture
    def sample_document(self):
        email_content = b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <test@example.com>

This is a test email body with some content.
It contains multiple lines and paragraphs.

Best regards,
Test Sender
"""
        return DocumentContent(
            url='test.eml',
            content_type='message/rfc822',
            raw_content=email_content
        )
    
    @patch('email.message_from_bytes')
    def test_extract_content_success(self, mock_email_parser, extractor, sample_document):
        """Test successful email content extraction"""
        # Mock email message
        mock_msg = Mock()
        mock_msg.get.side_effect = lambda key, default='': {
            'Subject': 'Test Email',
            'From': 'Test Sender <sender@example.com>',
            'To': 'recipient@example.com',
            'Date': 'Mon, 1 Jan 2024 12:00:00 +0000',
            'Message-ID': '<test@example.com>',
            'In-Reply-To': '',
            'References': '',
            'Reply-To': '',
            'CC': '',
            'BCC': '',
            'X-Priority': '',
            'Priority': ''
        }.get(key, default)
        
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content.return_value = "This is a test email body."
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_charset.return_value = "utf-8"
        
        mock_email_parser.return_value = mock_msg
        
        result = extractor.extract_content(sample_document)
        
        assert isinstance(result, ExtractedContent)
        assert result.document_type == DocumentType.EMAIL.value
        assert "Test Email" in result.text_content
        assert result.metadata['extraction_method'] == 'Enhanced email.parser'
        assert 'headers' in result.metadata
        assert 'structure' in result.metadata
        assert 'content_analysis' in result.metadata
        assert 'security_analysis' in result.metadata
    
    def test_extract_enhanced_headers(self, extractor):
        """Test enhanced header extraction"""
        mock_msg = Mock()
        mock_msg.get.side_effect = lambda key, default='': {
            'Subject': 'Test Subject',
            'From': 'Test User <test@example.com>',
            'To': 'recipient1@example.com, recipient2@example.com',
            'CC': 'cc@example.com',
            'Date': 'Mon, 1 Jan 2024 12:00:00 +0000',
            'Message-ID': '<test123@example.com>',
            'References': '<ref1@example.com> <ref2@example.com>'
        }.get(key, default)
        
        headers = extractor._extract_enhanced_headers(mock_msg)
        
        assert isinstance(headers, EmailHeader)
        assert headers.subject == 'Test Subject'
        assert headers.from_addr == 'test@example.com'
        assert headers.from_name == 'Test User'
        assert len(headers.to_addrs) == 2
        assert len(headers.references) == 2
    
    def test_analyze_email_structure(self, extractor):
        """Test email structure analysis"""
        mock_msg = Mock()
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = 'text/plain'
        mock_msg.get_content.return_value = "Test email content"
        
        structure = extractor._analyze_email_structure(mock_msg)
        
        assert isinstance(structure, EmailStructure)
        assert structure.is_multipart == False
        assert structure.has_plain_text == True
        assert structure.has_html == False
    
    def test_html_to_text(self, extractor):
        """Test HTML to text conversion"""
        html_content = "<p>This is <b>bold</b> text with <a href='#'>link</a>.</p>"
        text = extractor._html_to_text(html_content)
        
        assert "This is bold text with link." in text
        assert "<p>" not in text
        assert "<b>" not in text
    
    def test_has_urls(self, extractor):
        """Test URL detection"""
        text_with_url = "Visit our website at https://example.com for more info."
        text_without_url = "This is just regular text without any links."
        
        assert extractor._has_urls(text_with_url) == True
        assert extractor._has_urls(text_without_url) == False
    
    def test_has_phone_numbers(self, extractor):
        """Test phone number detection"""
        text_with_phone = "Call us at 123-456-7890 or (555) 123-4567."
        text_without_phone = "This text has no phone numbers."
        
        assert extractor._has_phone_numbers(text_with_phone) == True
        assert extractor._has_phone_numbers(text_without_phone) == False
    
    def test_has_email_addresses(self, extractor):
        """Test email address detection"""
        text_with_email = "Contact us at support@example.com for help."
        text_without_email = "This text has no email addresses."
        
        assert extractor._has_email_addresses(text_with_email) == True
        assert extractor._has_email_addresses(text_without_email) == False
    
    def test_detect_language(self, extractor):
        """Test basic language detection"""
        english_text = "The quick brown fox jumps over the lazy dog."
        spanish_text = "El gato está en la casa con el perro."
        
        assert extractor._detect_language(english_text) == 'english'
        assert extractor._detect_language(spanish_text) == 'spanish'
    
    def test_extract_sentiment_indicators(self, extractor):
        """Test sentiment indicator extraction"""
        positive_text = "This is great and wonderful news!"
        negative_text = "This is terrible and awful."
        urgent_text = "URGENT: Please respond immediately!"
        
        positive_result = extractor._extract_sentiment_indicators(positive_text)
        negative_result = extractor._extract_sentiment_indicators(negative_text)
        urgent_result = extractor._extract_sentiment_indicators(urgent_text)
        
        assert positive_result['positive_indicators'] > 0
        assert negative_result['negative_indicators'] > 0
        assert urgent_result['urgent_indicators'] > 0
    
    def test_detect_suspicious_links(self, extractor):
        """Test suspicious link detection"""
        text_with_suspicious = "Click here: http://bit.ly/suspicious or visit http://192.168.1.1"
        text_normal = "Visit our website at https://example.com"
        
        suspicious_links = extractor._detect_suspicious_links(text_with_suspicious)
        normal_links = extractor._detect_suspicious_links(text_normal)
        
        assert len(suspicious_links) > 0
        assert len(normal_links) == 0
    
    def test_detect_phishing_indicators(self, extractor):
        """Test phishing indicator detection"""
        # Mock suspicious headers
        suspicious_headers = EmailHeader(
            subject="URGENT ACTION REQUIRED: Verify your account",
            from_addr="noreply@fake-paypal.com",
            from_name="PayPal Security",
            to_addrs=["user@example.com"],
            cc_addrs=[],
            bcc_addrs=[],
            reply_to="",
            date=None,
            message_id="",
            in_reply_to="",
            references=[],
            priority="",
            content_type="text/plain",
            encoding="utf-8"
        )
        
        suspicious_text = "Act now! Your account expires today! Click here immediately!"
        
        indicators = extractor._detect_phishing_indicators(suspicious_headers, suspicious_text)
        
        assert len(indicators) > 0
        assert any("Suspicious subject" in indicator for indicator in indicators)
        assert any("Sender name/address mismatch" in indicator for indicator in indicators)


# Integration tests
class TestEnhancedExtractorsIntegration:
    """Integration tests for enhanced extractors"""
    
    def test_pdf_extractor_import(self):
        """Test that PDF extractor can be imported and initialized"""
        from src.pipeline.stages.extractors.pdf_extractor import EnhancedPDFExtractor
        extractor = EnhancedPDFExtractor()
        assert extractor is not None
    
    def test_docx_extractor_import(self):
        """Test that DOCX extractor can be imported and initialized"""
        from src.pipeline.stages.extractors.docx_extractor import EnhancedDOCXExtractor
        extractor = EnhancedDOCXExtractor()
        assert extractor is not None
    
    def test_email_extractor_import(self):
        """Test that email extractor can be imported and initialized"""
        from src.pipeline.stages.extractors.email_extractor import EnhancedEmailExtractor
        extractor = EnhancedEmailExtractor()
        assert extractor is not None
    
    def test_all_extractors_import(self):
        """Test that all extractors can be imported together"""
        from src.pipeline.stages.extractors import (
            EnhancedPDFExtractor, EnhancedDOCXExtractor, EnhancedEmailExtractor
        )
        
        pdf_extractor = EnhancedPDFExtractor()
        docx_extractor = EnhancedDOCXExtractor()
        email_extractor = EnhancedEmailExtractor()
        
        assert pdf_extractor is not None
        assert docx_extractor is not None
        assert email_extractor is not None


if __name__ == '__main__':
    pytest.main([__file__])