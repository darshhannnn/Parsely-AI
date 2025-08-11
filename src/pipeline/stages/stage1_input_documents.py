"""
Stage 1: Input Documents - Document download, validation, and preprocessing
"""

import os
import requests
import tempfile
import time
import mimetypes
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path

from ..core.interfaces import IDocumentProcessor, DocumentType
from ..core.models import DocumentContent, ExtractedContent
from ..core.exceptions import (
    DocumentDownloadError, UnsupportedFormatError, DocumentSizeError,
    ContentExtractionError, ValidationError, TimeoutError
)
from ..core.logging_utils import get_pipeline_logger
from ..core.utils import (
    timing_decorator, retry_decorator, validate_url, get_file_size_mb,
    safe_filename, create_temp_file, cleanup_temp_file, calculate_content_hash
)


class DocumentDownloader:
    """Handles secure document downloading with validation"""
    
    def __init__(self, timeout_seconds: int = 30, max_size_mb: float = 50):
        self.timeout_seconds = timeout_seconds
        self.max_size_mb = max_size_mb
        self.logger = get_pipeline_logger()
        
        # Configure session with security headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLM-Document-Processor/2.0',
            'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,message/rfc822,*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
    
    @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    @timing_decorator
    def download_document(self, url: str) -> DocumentContent:
        """Download document from URL with validation and security checks"""
        
        # Validate URL format
        if not validate_url(url):
            raise ValidationError(f"Invalid URL format: {url}", "url", url)
        
        self.logger.info("Starting document download", url=url)
        
        try:
            # Make HEAD request first to check size and content type
            head_response = self.session.head(url, timeout=self.timeout_seconds, allow_redirects=True)
            
            # Check content length if available
            content_length = head_response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    raise DocumentSizeError(size_mb, self.max_size_mb)
            
            # Get content type
            content_type = head_response.headers.get('content-type', '').lower()
            
            # Download the actual content
            response = self.session.get(url, timeout=self.timeout_seconds, stream=True)
            response.raise_for_status()
            
            # Download with size checking
            content_chunks = []
            total_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content_chunks.append(chunk)
                    total_size += len(chunk)
                    
                    # Check size limit during download
                    if total_size > self.max_size_mb * 1024 * 1024:
                        raise DocumentSizeError(total_size / (1024 * 1024), self.max_size_mb)
            
            raw_content = b''.join(content_chunks)
            
            # Final content type detection if not available from headers
            if not content_type or content_type == 'application/octet-stream':
                content_type = self._detect_content_type_from_content(raw_content, url)
            
            # Create document content object
            document_content = DocumentContent(
                url=url,
                content_type=content_type,
                raw_content=raw_content,
                metadata={
                    'download_timestamp': time.time(),
                    'original_filename': self._extract_filename_from_url(url),
                    'response_headers': dict(response.headers),
                    'status_code': response.status_code
                },
                size_bytes=len(raw_content)
            )
            
            self.logger.info(
                "Document downloaded successfully",
                url=url,
                size_mb=len(raw_content) / (1024 * 1024),
                content_type=content_type
            )
            
            return document_content
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Download timeout after {self.timeout_seconds} seconds", self.timeout_seconds, "download")
        
        except requests.exceptions.HTTPError as e:
            raise DocumentDownloadError(f"HTTP error downloading document: {e}", url, e.response.status_code if e.response else None)
        
        except requests.exceptions.RequestException as e:
            raise DocumentDownloadError(f"Network error downloading document: {e}", url)
        
        except Exception as e:
            raise DocumentDownloadError(f"Unexpected error downloading document: {e}", url)
    
    def _detect_content_type_from_content(self, content: bytes, url: str) -> str:
        """Detect content type from file content and URL"""
        
        # Check file signature (magic numbers)
        if content.startswith(b'%PDF'):
            return 'application/pdf'
        elif content.startswith(b'PK\x03\x04') and b'word/' in content[:1000]:
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif content.startswith((b'Return-Path:', b'Received:', b'From:', b'To:', b'Subject:')):
            return 'message/rfc822'
        
        # Fallback to URL extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        if path.endswith('.pdf'):
            return 'application/pdf'
        elif path.endswith(('.docx', '.doc')):
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif path.endswith(('.eml', '.msg')):
            return 'message/rfc822'
        
        # Use mimetypes as final fallback
        mime_type, _ = mimetypes.guess_type(url)
        return mime_type or 'application/octet-stream'
    
    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        return safe_filename(filename) if filename else "document"


class DocumentFormatDetector:
    """Detects and validates document formats"""
    
    SUPPORTED_FORMATS = {
        'application/pdf': DocumentType.PDF,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
        'application/msword': DocumentType.DOCX,  # Legacy .doc files
        'message/rfc822': DocumentType.EMAIL,
        'text/plain': DocumentType.EMAIL  # Sometimes emails are served as text/plain
    }
    
    def __init__(self):
        self.logger = get_pipeline_logger()
    
    def detect_format(self, document: DocumentContent) -> DocumentType:
        """Detect document format from content type and content"""
        
        content_type = document.content_type.lower().split(';')[0]  # Remove charset info
        
        # Direct content type mapping
        if content_type in self.SUPPORTED_FORMATS:
            detected_format = self.SUPPORTED_FORMATS[content_type]
            self.logger.debug(f"Format detected from content type: {detected_format.value}")
            return detected_format
        
        # Content-based detection as fallback
        content = document.raw_content
        
        if content.startswith(b'%PDF'):
            return DocumentType.PDF
        elif content.startswith(b'PK\x03\x04') and b'word/' in content[:1000]:
            return DocumentType.DOCX
        elif self._is_email_content(content):
            return DocumentType.EMAIL
        
        # If we can't detect, raise an error
        supported_list = [fmt.value for fmt in DocumentType]
        raise UnsupportedFormatError(content_type, supported_list)
    
    def _is_email_content(self, content: bytes) -> bool:
        """Check if content appears to be an email"""
        try:
            text_content = content[:1000].decode('utf-8', errors='ignore').lower()
            email_headers = ['return-path:', 'received:', 'from:', 'to:', 'subject:', 'date:', 'message-id:']
            return any(header in text_content for header in email_headers)
        except:
            return False


class ContentExtractor:
    """Extracts content from different document formats using enhanced extractors"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        
        # Initialize enhanced extractors
        try:
            from .extractors import EnhancedPDFExtractor, EnhancedDOCXExtractor, EnhancedEmailExtractor
            self.pdf_extractor = EnhancedPDFExtractor()
            self.docx_extractor = EnhancedDOCXExtractor()
            self.email_extractor = EnhancedEmailExtractor()
            self.use_enhanced = True
            self.logger.info("Enhanced extractors loaded successfully")
        except ImportError as e:
            self.logger.warning(f"Enhanced extractors not available, falling back to basic extraction: {e}")
            self.use_enhanced = False
    
    @timing_decorator
    def extract_content(self, document: DocumentContent, document_type: DocumentType) -> ExtractedContent:
        """Extract content based on document type"""
        
        self.logger.info(f"Extracting content from {document_type.value} document")
        
        try:
            if self.use_enhanced:
                # Use enhanced extractors
                if document_type == DocumentType.PDF:
                    return self.pdf_extractor.extract_content(document)
                elif document_type == DocumentType.DOCX:
                    return self.docx_extractor.extract_content(document)
                elif document_type == DocumentType.EMAIL:
                    return self.email_extractor.extract_content(document)
                else:
                    raise UnsupportedFormatError(document_type.value, [fmt.value for fmt in DocumentType])
            else:
                # Fall back to basic extraction
                if document_type == DocumentType.PDF:
                    return self._extract_pdf_content(document)
                elif document_type == DocumentType.DOCX:
                    return self._extract_docx_content(document)
                elif document_type == DocumentType.EMAIL:
                    return self._extract_email_content(document)
                else:
                    raise UnsupportedFormatError(document_type.value, [fmt.value for fmt in DocumentType])
        
        except Exception as e:
            raise ContentExtractionError(f"Failed to extract content: {e}", document_type.value)
    
    def _extract_pdf_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content from PDF document"""
        try:
            import PyPDF2
            
            # Save content to temporary file
            temp_file = create_temp_file(suffix='.pdf')
            
            try:
                with open(temp_file, 'wb') as f:
                    f.write(document.raw_content)
                
                # Extract text using PyPDF2
                with open(temp_file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    pages = []
                    sections = {}
                    full_text = []
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        pages.append(page_text)
                        full_text.append(page_text)
                        sections[f"page_{page_num}"] = page_text
                    
                    text_content = '\n\n'.join(full_text)
                    
                    # Extract metadata
                    metadata = {
                        'page_count': len(pdf_reader.pages),
                        'pdf_metadata': dict(pdf_reader.metadata) if pdf_reader.metadata else {},
                        'extraction_method': 'PyPDF2'
                    }
                    
                    return ExtractedContent(
                        document_id=calculate_content_hash(document.raw_content),
                        document_type=DocumentType.PDF.value,
                        text_content=text_content,
                        pages=pages,
                        sections=sections,
                        metadata=metadata
                    )
            
            finally:
                cleanup_temp_file(temp_file)
        
        except ImportError:
            raise ContentExtractionError("PyPDF2 library not available for PDF processing", DocumentType.PDF.value)
        except Exception as e:
            raise ContentExtractionError(f"PDF extraction failed: {e}", DocumentType.PDF.value)
    
    def _extract_docx_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content from DOCX document"""
        try:
            from docx import Document
            
            # Save content to temporary file
            temp_file = create_temp_file(suffix='.docx')
            
            try:
                with open(temp_file, 'wb') as f:
                    f.write(document.raw_content)
                
                # Extract text using python-docx
                doc = Document(temp_file)
                
                paragraphs = []
                sections = {}
                current_section = "Introduction"
                section_content = []
                
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text.strip())
                        
                        # Detect headings (basic heuristic)
                        if para.style.name.startswith('Heading'):
                            # Save previous section
                            if section_content:
                                sections[current_section] = '\n'.join(section_content)
                            
                            # Start new section
                            current_section = para.text.strip()
                            section_content = []
                        else:
                            section_content.append(para.text.strip())
                
                # Save final section
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                text_content = '\n\n'.join(paragraphs)
                
                # Extract metadata
                metadata = {
                    'paragraph_count': len(paragraphs),
                    'section_count': len(sections),
                    'extraction_method': 'python-docx'
                }
                
                # Add document properties if available
                if hasattr(doc, 'core_properties'):
                    props = doc.core_properties
                    metadata['document_properties'] = {
                        'title': props.title,
                        'author': props.author,
                        'subject': props.subject,
                        'created': props.created.isoformat() if props.created else None,
                        'modified': props.modified.isoformat() if props.modified else None
                    }
                
                return ExtractedContent(
                    document_id=calculate_content_hash(document.raw_content),
                    document_type=DocumentType.DOCX.value,
                    text_content=text_content,
                    pages=None,  # DOCX doesn't have fixed pages
                    sections=sections,
                    metadata=metadata
                )
            
            finally:
                cleanup_temp_file(temp_file)
        
        except ImportError:
            raise ContentExtractionError("python-docx library not available for DOCX processing", DocumentType.DOCX.value)
        except Exception as e:
            raise ContentExtractionError(f"DOCX extraction failed: {e}", DocumentType.DOCX.value)
    
    def _extract_email_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content from email document"""
        try:
            import email
            from email import policy
            
            # Parse email content
            msg = email.message_from_bytes(document.raw_content, policy=policy.default)
            
            # Extract headers
            headers = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'date': msg.get('Date', ''),
                'message_id': msg.get('Message-ID', ''),
                'reply_to': msg.get('Reply-To', ''),
                'cc': msg.get('CC', ''),
                'bcc': msg.get('BCC', '')
            }
            
            # Extract body content
            body_parts = []
            sections = {}
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        content = part.get_content()
                        if content and content.strip():
                            body_parts.append(content.strip())
                    elif content_type == 'text/html':
                        # Store HTML content separately
                        html_content = part.get_content()
                        if html_content:
                            sections['html_body'] = html_content
            else:
                content = msg.get_content()
                if content and content.strip():
                    body_parts.append(content.strip())
            
            # Combine all text content
            header_text = f"Subject: {headers['subject']}\nFrom: {headers['from']}\nTo: {headers['to']}\nDate: {headers['date']}\n"
            body_text = '\n\n'.join(body_parts)
            text_content = header_text + '\n\n' + body_text
            
            # Store sections
            sections['headers'] = header_text
            sections['body'] = body_text
            
            # Extract metadata
            metadata = {
                'headers': headers,
                'is_multipart': msg.is_multipart(),
                'content_type': msg.get_content_type(),
                'extraction_method': 'email.parser'
            }
            
            return ExtractedContent(
                document_id=calculate_content_hash(document.raw_content),
                document_type=DocumentType.EMAIL.value,
                text_content=text_content,
                pages=None,  # Emails don't have pages
                sections=sections,
                metadata=metadata
            )
        
        except Exception as e:
            raise ContentExtractionError(f"Email extraction failed: {e}", DocumentType.EMAIL.value)


class DocumentValidator:
    """Validates extracted document content"""
    
    def __init__(self, min_content_length: int = 10):
        self.min_content_length = min_content_length
        self.logger = get_pipeline_logger()
    
    def validate_document(self, content: ExtractedContent) -> bool:
        """Validate extracted content meets minimum requirements"""
        
        self.logger.debug(f"Validating extracted content for document {content.document_id}")
        
        # Check if content exists and has minimum length
        if not content.text_content or len(content.text_content.strip()) < self.min_content_length:
            raise ValidationError(
                f"Document content too short (minimum {self.min_content_length} characters required)",
                "content_length",
                len(content.text_content) if content.text_content else 0
            )
        
        # Check document type is supported
        supported_types = [fmt.value for fmt in DocumentType]
        if content.document_type not in supported_types:
            raise ValidationError(
                f"Unsupported document type: {content.document_type}",
                "document_type",
                content.document_type
            )
        
        # Additional format-specific validation
        if content.document_type == DocumentType.PDF.value:
            if not content.pages or len(content.pages) == 0:
                raise ValidationError("PDF document has no extractable pages", "pages", 0)
        
        elif content.document_type == DocumentType.EMAIL.value:
            if not content.metadata.get('headers', {}).get('subject'):
                self.logger.warning("Email document missing subject header")
        
        self.logger.info(f"Document validation successful for {content.document_id}")
        return True


class Stage1DocumentProcessor(IDocumentProcessor):
    """Complete Stage 1: Input Documents processor with preprocessing pipeline"""
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_size_mb: float = 50,
        min_content_length: int = 10,
        enable_preprocessing: bool = True
    ):
        self.downloader = DocumentDownloader(timeout_seconds, max_size_mb)
        self.format_detector = DocumentFormatDetector()
        self.content_extractor = ContentExtractor()
        self.validator = DocumentValidator(min_content_length)
        self.logger = get_pipeline_logger()
        
        # Initialize preprocessing pipeline
        self.enable_preprocessing = enable_preprocessing
        if enable_preprocessing:
            try:
                from .preprocessing import PreprocessingPipeline, PreprocessingOptions, TempFileManager
                
                # Initialize temp file manager
                self.temp_file_manager = TempFileManager()
                
                # Initialize preprocessing pipeline
                preprocessing_options = PreprocessingOptions(
                    extract_comprehensive_metadata=True,
                    normalize_content=True,
                    preserve_original_content=True,
                    enable_temp_file_management=True
                )
                
                self.preprocessing_pipeline = PreprocessingPipeline(
                    options=preprocessing_options,
                    temp_file_manager=self.temp_file_manager
                )
                
                self.logger.info("Preprocessing pipeline initialized successfully")
                
            except ImportError as e:
                self.logger.warning(f"Preprocessing pipeline not available, using basic processing: {e}")
                self.enable_preprocessing = False
                self.preprocessing_pipeline = None
                self.temp_file_manager = None
        else:
            self.preprocessing_pipeline = None
            self.temp_file_manager = None
    
    @timing_decorator
    def download_document(self, url: str) -> DocumentContent:
        """Download document from URL with validation"""
        return self.downloader.download_document(url)
    
    def detect_format(self, content: bytes) -> DocumentType:
        """Detect document format from content"""
        # Create temporary DocumentContent for format detection
        temp_doc = DocumentContent(
            url="",
            content_type="application/octet-stream",
            raw_content=content
        )
        return self.format_detector.detect_format(temp_doc)
    
    @timing_decorator
    def extract_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content based on document type"""
        document_type = self.format_detector.detect_format(document)
        return self.content_extractor.extract_content(document, document_type)
    
    def validate_document(self, content: ExtractedContent) -> bool:
        """Validate extracted content"""
        return self.validator.validate_document(content)
    
    @timing_decorator
    def process_document_url(self, url: str) -> ExtractedContent:
        """Complete Stage 1 processing: download, extract, validate, and preprocess"""
        
        self.logger.log_stage_start("stage1_input_documents", url=url)
        
        try:
            # Step 1: Download document
            document = self.download_document(url)
            
            # Step 2: Extract content
            extracted_content = self.extract_content(document)
            
            # Step 3: Validate content
            self.validate_document(extracted_content)
            
            # Step 4: Apply preprocessing pipeline (if enabled)
            final_content = extracted_content
            preprocessing_info = {}
            
            if self.enable_preprocessing and self.preprocessing_pipeline:
                try:
                    preprocessing_result = self.preprocessing_pipeline.process(document, extracted_content)
                    
                    if preprocessing_result.success:
                        final_content = preprocessing_result.processed_content
                        preprocessing_info = {
                            'preprocessing_applied': True,
                            'preprocessing_summary': self.preprocessing_pipeline.get_pipeline_summary(preprocessing_result),
                            'validation_report': self.preprocessing_pipeline.validate_preprocessing_result(preprocessing_result)
                        }
                        
                        self.logger.info(
                            f"Preprocessing completed successfully for {extracted_content.document_id}",
                            processing_time_ms=preprocessing_result.processing_time_ms,
                            changes_made=len(preprocessing_result.normalization_result.changes_made) if preprocessing_result.normalization_result else 0
                        )
                    else:
                        self.logger.warning(
                            f"Preprocessing failed for {extracted_content.document_id}, using original content",
                            errors=preprocessing_result.errors
                        )
                        preprocessing_info = {
                            'preprocessing_applied': False,
                            'preprocessing_errors': preprocessing_result.errors,
                            'preprocessing_warnings': preprocessing_result.warnings
                        }
                
                except Exception as e:
                    self.logger.error(f"Preprocessing pipeline error: {e}")
                    preprocessing_info = {
                        'preprocessing_applied': False,
                        'preprocessing_error': str(e)
                    }
            else:
                preprocessing_info = {'preprocessing_applied': False, 'reason': 'disabled'}
            
            # Add preprocessing info to metadata
            if final_content.metadata is None:
                final_content.metadata = {}
            final_content.metadata['stage1_processing'] = preprocessing_info
            
            self.logger.log_stage_complete(
                "stage1_input_documents",
                0,  # Duration will be calculated by timing decorator
                document_id=final_content.document_id,
                document_type=final_content.document_type,
                content_length=len(final_content.text_content or ""),
                preprocessing_applied=preprocessing_info.get('preprocessing_applied', False)
            )
            
            return final_content
        
        except Exception as e:
            self.logger.log_stage_error("stage1_input_documents", e, url=url)
            raise
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        if self.temp_file_manager:
            return self.temp_file_manager.get_statistics()
        return {}
    
    def cleanup_temp_files(self) -> int:
        """Clean up temporary files"""
        if self.temp_file_manager:
            return self.temp_file_manager.cleanup_all_files()
        return 0
    
    def shutdown(self):
        """Shutdown the processor and cleanup resources"""
        if self.temp_file_manager:
            self.temp_file_manager.shutdown()
        self.logger.info("Stage1DocumentProcessor shutdown completed")