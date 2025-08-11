"""
Comprehensive metadata extraction for all document formats
"""

import os
import hashlib
import mimetypes
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from ...core.models import DocumentContent, ExtractedContent
from ...core.interfaces import DocumentType
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator, calculate_content_hash


class MetadataExtractor:
    """Comprehensive metadata extraction and preservation"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def extract_comprehensive_metadata(
        self, 
        document: DocumentContent, 
        extracted_content: ExtractedContent
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata from document and content"""
        
        self.logger.info(f"Extracting comprehensive metadata for document {extracted_content.document_id}")
        
        metadata = {
            # Document source metadata
            'source': self._extract_source_metadata(document),
            
            # Content analysis metadata
            'content_analysis': self._analyze_content(extracted_content),
            
            # Technical metadata
            'technical': self._extract_technical_metadata(document, extracted_content),
            
            # Processing metadata
            'processing': self._extract_processing_metadata(extracted_content),
            
            # Security metadata
            'security': self._extract_security_metadata(document),
            
            # Quality metrics
            'quality': self._assess_content_quality(extracted_content),
            
            # Accessibility metadata
            'accessibility': self._assess_accessibility(extracted_content),
            
            # Compliance metadata
            'compliance': self._assess_compliance(document, extracted_content)
        }
        
        # Add format-specific metadata
        format_metadata = self._extract_format_specific_metadata(document, extracted_content)
        if format_metadata:
            metadata['format_specific'] = format_metadata
        
        self.logger.info(f"Metadata extraction completed for {extracted_content.document_id}")
        return metadata
    
    def _extract_source_metadata(self, document: DocumentContent) -> Dict[str, Any]:
        """Extract metadata about document source"""
        parsed_url = urlparse(document.url)
        
        return {
            'url': document.url,
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'filename': os.path.basename(parsed_url.path) or 'unknown',
            'file_extension': Path(parsed_url.path).suffix.lower(),
            'content_type': document.content_type,
            'size_bytes': document.size_bytes,
            'size_mb': round(document.size_bytes / (1024 * 1024), 2),
            'download_timestamp': document.metadata.get('download_timestamp'),
            'original_filename': document.metadata.get('original_filename'),
            'response_headers': document.metadata.get('response_headers', {}),
            'status_code': document.metadata.get('status_code'),
            'content_hash': {
                'md5': hashlib.md5(document.raw_content).hexdigest(),
                'sha256': calculate_content_hash(document.raw_content),
                'sha1': hashlib.sha1(document.raw_content).hexdigest()
            }
        }
    
    def _analyze_content(self, content: ExtractedContent) -> Dict[str, Any]:
        """Analyze extracted content characteristics"""
        text = content.text_content or ""
        
        # Basic text statistics
        words = text.split()
        sentences = text.split('.')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Character analysis
        char_counts = {
            'total': len(text),
            'alphabetic': sum(1 for c in text if c.isalpha()),
            'numeric': sum(1 for c in text if c.isdigit()),
            'whitespace': sum(1 for c in text if c.isspace()),
            'punctuation': sum(1 for c in text if c in '.,;:!?'),
            'special': sum(1 for c in text if not c.isalnum() and not c.isspace())
        }
        
        # Language characteristics
        language_info = self._analyze_language_characteristics(text)
        
        # Content structure
        structure_info = self._analyze_content_structure(content)
        
        return {
            'statistics': {
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'paragraph_count': len(paragraphs),
                'line_count': len(lines),
                'average_words_per_sentence': len(words) / max(len(sentences), 1),
                'average_words_per_paragraph': len(words) / max(len(paragraphs), 1),
                'character_counts': char_counts
            },
            'language': language_info,
            'structure': structure_info,
            'readability': self._calculate_readability_metrics(text),
            'content_patterns': self._identify_content_patterns(text)
        }
    
    def _extract_technical_metadata(
        self, 
        document: DocumentContent, 
        content: ExtractedContent
    ) -> Dict[str, Any]:
        """Extract technical metadata"""
        
        return {
            'document_id': content.document_id,
            'document_type': content.document_type,
            'extraction_timestamp': content.extraction_timestamp.isoformat(),
            'extraction_method': content.metadata.get('extraction_method', 'unknown'),
            'mime_type': document.content_type,
            'encoding': self._detect_encoding(document.raw_content),
            'format_version': self._detect_format_version(document),
            'compression_info': self._analyze_compression(document),
            'embedded_objects': self._count_embedded_objects(content),
            'hyperlinks': self._extract_hyperlinks(content.text_content or ""),
            'references': self._extract_references(content.text_content or "")
        }
    
    def _extract_processing_metadata(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract processing-related metadata"""
        
        processing_info = {
            'extraction_timestamp': content.extraction_timestamp.isoformat(),
            'processing_duration_ms': 0,  # Will be updated by timing decorator
            'extraction_success': True,
            'warnings': [],
            'errors': [],
            'processing_stages': ['download', 'format_detection', 'content_extraction', 'validation']
        }
        
        # Add format-specific processing info
        if content.metadata:
            if 'page_count' in content.metadata:
                processing_info['pages_processed'] = content.metadata['page_count']
            if 'paragraph_count' in content.metadata:
                processing_info['paragraphs_processed'] = content.metadata['paragraph_count']
            if 'sections_detected' in content.metadata:
                processing_info['sections_detected'] = content.metadata['sections_detected']
        
        return processing_info
    
    def _extract_security_metadata(self, document: DocumentContent) -> Dict[str, Any]:
        """Extract security-related metadata"""
        
        security_info = {
            'download_security': {
                'https_used': document.url.startswith('https://'),
                'domain_verified': True,  # Placeholder - would implement actual verification
                'certificate_valid': True  # Placeholder - would implement actual verification
            },
            'content_security': {
                'suspicious_patterns': self._detect_suspicious_patterns(document.raw_content),
                'embedded_scripts': self._detect_embedded_scripts(document.raw_content),
                'external_references': self._detect_external_references(document.raw_content),
                'potential_malware_indicators': []  # Placeholder for malware detection
            },
            'privacy_indicators': {
                'contains_pii': self._detect_pii_patterns(document.raw_content),
                'contains_financial_data': self._detect_financial_patterns(document.raw_content),
                'contains_health_data': self._detect_health_patterns(document.raw_content)
            }
        }
        
        return security_info
    
    def _assess_content_quality(self, content: ExtractedContent) -> Dict[str, Any]:
        """Assess content quality metrics"""
        
        text = content.text_content or ""
        
        quality_metrics = {
            'completeness': {
                'has_content': bool(text.strip()),
                'content_length_adequate': len(text) > 100,
                'has_structure': bool(content.sections and len(content.sections) > 1),
                'extraction_complete': not self._has_extraction_errors(content)
            },
            'clarity': {
                'readability_score': self._calculate_readability_score(text),
                'language_consistency': self._check_language_consistency(text),
                'formatting_preserved': self._check_formatting_preservation(content)
            },
            'accuracy': {
                'ocr_confidence': self._estimate_ocr_confidence(content),
                'extraction_confidence': self._estimate_extraction_confidence(content),
                'text_coherence': self._assess_text_coherence(text)
            },
            'usability': {
                'searchable': bool(text.strip()),
                'machine_readable': True,
                'structured_data_available': bool(content.sections)
            }
        }
        
        # Calculate overall quality score
        quality_metrics['overall_score'] = self._calculate_overall_quality_score(quality_metrics)
        
        return quality_metrics
    
    def _assess_accessibility(self, content: ExtractedContent) -> Dict[str, Any]:
        """Assess content accessibility"""
        
        return {
            'text_available': bool(content.text_content and content.text_content.strip()),
            'structured_content': bool(content.sections),
            'headings_present': self._has_headings(content),
            'alt_text_available': False,  # Would need image analysis
            'reading_level': self._assess_reading_level(content.text_content or ""),
            'language_identified': self._identify_primary_language(content.text_content or ""),
            'accessibility_score': 0.0  # Would calculate based on above factors
        }
    
    def _assess_compliance(self, document: DocumentContent, content: ExtractedContent) -> Dict[str, Any]:
        """Assess compliance with various standards"""
        
        return {
            'gdpr': {
                'contains_personal_data': self._detect_personal_data(content.text_content or ""),
                'data_processing_lawful': True,  # Placeholder
                'consent_required': False  # Placeholder
            },
            'accessibility_standards': {
                'wcag_compliant': self._check_wcag_compliance(content),
                'section_508_compliant': self._check_section_508_compliance(content)
            },
            'industry_standards': {
                'iso_27001_considerations': self._check_iso_27001(document),
                'hipaa_considerations': self._check_hipaa_compliance(content.text_content or "")
            }
        }
    
    def _extract_format_specific_metadata(
        self, 
        document: DocumentContent, 
        content: ExtractedContent
    ) -> Optional[Dict[str, Any]]:
        """Extract format-specific metadata"""
        
        if content.document_type == DocumentType.PDF.value:
            return self._extract_pdf_metadata(content)
        elif content.document_type == DocumentType.DOCX.value:
            return self._extract_docx_metadata(content)
        elif content.document_type == DocumentType.EMAIL.value:
            return self._extract_email_metadata(content)
        
        return None
    
    def _extract_pdf_metadata(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract PDF-specific metadata"""
        metadata = content.metadata or {}
        
        return {
            'pdf_version': metadata.get('pdf_metadata', {}).get('Producer', 'Unknown'),
            'page_count': metadata.get('page_count', 0),
            'has_images': metadata.get('images_detected', 0) > 0,
            'has_tables': metadata.get('tables_detected', 0) > 0,
            'creation_date': metadata.get('pdf_metadata', {}).get('CreationDate'),
            'modification_date': metadata.get('pdf_metadata', {}).get('ModDate'),
            'author': metadata.get('pdf_metadata', {}).get('Author'),
            'title': metadata.get('pdf_metadata', {}).get('Title'),
            'subject': metadata.get('pdf_metadata', {}).get('Subject'),
            'keywords': metadata.get('pdf_metadata', {}).get('Keywords'),
            'creator': metadata.get('pdf_metadata', {}).get('Creator'),
            'producer': metadata.get('pdf_metadata', {}).get('Producer')
        }
    
    def _extract_docx_metadata(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract DOCX-specific metadata"""
        metadata = content.metadata or {}
        doc_props = metadata.get('document_properties', {})
        
        return {
            'title': doc_props.get('title'),
            'author': doc_props.get('author'),
            'subject': doc_props.get('subject'),
            'keywords': doc_props.get('keywords'),
            'category': doc_props.get('category'),
            'comments': doc_props.get('comments'),
            'created': doc_props.get('created'),
            'modified': doc_props.get('modified'),
            'last_modified_by': doc_props.get('last_modified_by'),
            'revision': doc_props.get('revision'),
            'version': doc_props.get('version'),
            'paragraph_count': metadata.get('paragraph_count', 0),
            'section_count': metadata.get('section_count', 0),
            'tables_count': metadata.get('tables_count', 0),
            'images_count': metadata.get('images_count', 0),
            'styles_used': metadata.get('styles_used', [])
        }
    
    def _extract_email_metadata(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract email-specific metadata"""
        metadata = content.metadata or {}
        headers = metadata.get('headers', {})
        structure = metadata.get('structure', {})
        
        return {
            'subject': headers.get('subject'),
            'from_address': headers.get('from'),
            'to_addresses': headers.get('to', []),
            'cc_addresses': headers.get('cc', []),
            'date_sent': headers.get('date'),
            'message_id': headers.get('message_id'),
            'is_multipart': structure.get('is_multipart', False),
            'has_attachments': structure.get('has_attachments', False),
            'attachment_count': len(metadata.get('attachments', [])),
            'has_html_content': structure.get('has_html', False),
            'thread_info': structure.get('thread_info', {}),
            'security_analysis': metadata.get('security_analysis', {})
        }
    
    # Helper methods for content analysis
    
    def _analyze_language_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze language characteristics of text"""
        if not text:
            return {'primary_language': 'unknown', 'confidence': 0.0}
        
        # Simple language detection based on character patterns
        # In production, would use proper language detection library
        
        # Count common English words
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su']
        french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son']
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        spanish_count = sum(1 for word in spanish_words if f' {word} ' in f' {text_lower} ')
        french_count = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ')
        
        total_words = len(text.split())
        
        if english_count >= spanish_count and english_count >= french_count:
            return {
                'primary_language': 'english',
                'confidence': min(english_count / max(total_words * 0.1, 1), 1.0),
                'word_indicators': english_count
            }
        elif spanish_count >= french_count:
            return {
                'primary_language': 'spanish',
                'confidence': min(spanish_count / max(total_words * 0.1, 1), 1.0),
                'word_indicators': spanish_count
            }
        elif french_count > 0:
            return {
                'primary_language': 'french',
                'confidence': min(french_count / max(total_words * 0.1, 1), 1.0),
                'word_indicators': french_count
            }
        else:
            return {'primary_language': 'unknown', 'confidence': 0.0, 'word_indicators': 0}
    
    def _analyze_content_structure(self, content: ExtractedContent) -> Dict[str, Any]:
        """Analyze content structure"""
        return {
            'has_sections': bool(content.sections and len(content.sections) > 1),
            'section_count': len(content.sections) if content.sections else 0,
            'has_pages': bool(content.pages),
            'page_count': len(content.pages) if content.pages else 0,
            'hierarchical_structure': self._detect_hierarchical_structure(content.text_content or ""),
            'list_structures': self._detect_list_structures(content.text_content or ""),
            'table_structures': self._detect_table_structures(content.text_content or "")
        }
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        if not text:
            return {'flesch_score': 0.0, 'reading_level': 'unknown'}
        
        # Simple readability calculation (Flesch Reading Ease approximation)
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return {'flesch_score': 0.0, 'reading_level': 'unknown'}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Determine reading level
        if flesch_score >= 90:
            reading_level = 'very_easy'
        elif flesch_score >= 80:
            reading_level = 'easy'
        elif flesch_score >= 70:
            reading_level = 'fairly_easy'
        elif flesch_score >= 60:
            reading_level = 'standard'
        elif flesch_score >= 50:
            reading_level = 'fairly_difficult'
        elif flesch_score >= 30:
            reading_level = 'difficult'
        else:
            reading_level = 'very_difficult'
        
        return {
            'flesch_score': round(flesch_score, 2),
            'reading_level': reading_level,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    def _identify_content_patterns(self, text: str) -> Dict[str, Any]:
        """Identify patterns in content"""
        import re
        
        patterns = {
            'email_addresses': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'phone_numbers': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
            'urls': len(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text, re.IGNORECASE)),
            'dates': len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)),
            'numbers': len(re.findall(r'\b\d+\b', text)),
            'currency': len(re.findall(r'\$\d+(?:\.\d{2})?', text)),
            'percentages': len(re.findall(r'\d+%', text)),
            'social_security': len(re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)),
            'credit_cards': len(re.findall(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', text))
        }
        
        return patterns
    
    # Additional helper methods (simplified implementations)
    
    def _detect_encoding(self, content: bytes) -> str:
        """Detect content encoding"""
        try:
            import chardet
            result = chardet.detect(content)
            return result.get('encoding', 'utf-8')
        except ImportError:
            return 'utf-8'
    
    def _detect_format_version(self, document: DocumentContent) -> str:
        """Detect format version"""
        if document.content_type == 'application/pdf':
            if document.raw_content.startswith(b'%PDF-1.4'):
                return 'PDF 1.4'
            elif document.raw_content.startswith(b'%PDF-1.5'):
                return 'PDF 1.5'
            elif document.raw_content.startswith(b'%PDF-1.6'):
                return 'PDF 1.6'
            elif document.raw_content.startswith(b'%PDF-1.7'):
                return 'PDF 1.7'
        return 'unknown'
    
    def _analyze_compression(self, document: DocumentContent) -> Dict[str, Any]:
        """Analyze compression information"""
        return {
            'is_compressed': document.content_type in [
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ],
            'compression_ratio': 0.0,  # Would calculate actual ratio
            'compression_type': 'zip' if document.raw_content.startswith(b'PK') else 'none'
        }
    
    def _count_embedded_objects(self, content: ExtractedContent) -> Dict[str, int]:
        """Count embedded objects"""
        metadata = content.metadata or {}
        return {
            'images': metadata.get('images_count', metadata.get('images_detected', 0)),
            'tables': metadata.get('tables_count', metadata.get('tables_detected', 0)),
            'hyperlinks': len(self._extract_hyperlinks(content.text_content or "")),
            'attachments': len(metadata.get('attachments', []))
        }
    
    def _extract_hyperlinks(self, text: str) -> List[str]:
        """Extract hyperlinks from text"""
        import re
        return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text, re.IGNORECASE)
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from text"""
        import re
        # Simple reference pattern matching
        references = re.findall(r'\[(\d+)\]|\((\d+)\)', text)
        return [ref[0] or ref[1] for ref in references]
    
    def _detect_suspicious_patterns(self, content: bytes) -> List[str]:
        """Detect suspicious patterns in content"""
        suspicious = []
        
        # Check for embedded scripts
        if b'<script' in content.lower():
            suspicious.append('embedded_javascript')
        
        # Check for suspicious URLs
        if b'bit.ly' in content or b'tinyurl' in content:
            suspicious.append('shortened_urls')
        
        return suspicious
    
    def _detect_embedded_scripts(self, content: bytes) -> bool:
        """Detect embedded scripts"""
        script_patterns = [b'<script', b'javascript:', b'vbscript:']
        return any(pattern in content.lower() for pattern in script_patterns)
    
    def _detect_external_references(self, content: bytes) -> List[str]:
        """Detect external references"""
        import re
        text = content.decode('utf-8', errors='ignore')
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text, re.IGNORECASE)
        return urls
    
    def _detect_pii_patterns(self, content: bytes) -> bool:
        """Detect PII patterns"""
        text = content.decode('utf-8', errors='ignore')
        import re
        
        # Check for SSN, credit cards, etc.
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        return any(re.search(pattern, text) for pattern in pii_patterns)
    
    def _detect_financial_patterns(self, content: bytes) -> bool:
        """Detect financial data patterns"""
        text = content.decode('utf-8', errors='ignore')
        import re
        
        financial_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{9}\b',  # Routing number
            r'account\s+number',  # Account references
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in financial_patterns)
    
    def _detect_health_patterns(self, content: bytes) -> bool:
        """Detect health data patterns"""
        text = content.decode('utf-8', errors='ignore')
        
        health_keywords = [
            'medical', 'patient', 'diagnosis', 'treatment', 'prescription',
            'doctor', 'physician', 'hospital', 'clinic', 'health'
        ]
        
        return any(keyword in text.lower() for keyword in health_keywords)
    
    # Quality assessment methods (simplified)
    
    def _has_extraction_errors(self, content: ExtractedContent) -> bool:
        """Check if extraction had errors"""
        return not content.text_content or len(content.text_content.strip()) < 10
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score"""
        return self._calculate_readability_metrics(text).get('flesch_score', 0.0)
    
    def _check_language_consistency(self, text: str) -> bool:
        """Check language consistency"""
        return True  # Simplified - would implement actual consistency check
    
    def _check_formatting_preservation(self, content: ExtractedContent) -> bool:
        """Check if formatting was preserved"""
        return bool(content.sections and len(content.sections) > 1)
    
    def _estimate_ocr_confidence(self, content: ExtractedContent) -> float:
        """Estimate OCR confidence"""
        # For non-OCR documents, return 1.0
        return 1.0
    
    def _estimate_extraction_confidence(self, content: ExtractedContent) -> float:
        """Estimate extraction confidence"""
        if not content.text_content:
            return 0.0
        
        # Simple heuristic based on content length and structure
        base_score = 0.5
        if len(content.text_content) > 100:
            base_score += 0.2
        if content.sections and len(content.sections) > 1:
            base_score += 0.2
        if content.metadata:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _assess_text_coherence(self, text: str) -> float:
        """Assess text coherence"""
        if not text:
            return 0.0
        
        # Simple coherence assessment based on sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Check for reasonable sentence lengths
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_length <= 30:
            return 0.8
        else:
            return 0.5
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Completeness score
        completeness = quality_metrics.get('completeness', {})
        completeness_score = sum([
            completeness.get('has_content', False),
            completeness.get('content_length_adequate', False),
            completeness.get('has_structure', False),
            completeness.get('extraction_complete', False)
        ]) / 4
        scores.append(completeness_score)
        
        # Clarity score
        clarity = quality_metrics.get('clarity', {})
        readability = clarity.get('readability_score', 0) / 100  # Normalize to 0-1
        clarity_score = (readability + 
                        clarity.get('language_consistency', False) + 
                        clarity.get('formatting_preserved', False)) / 3
        scores.append(clarity_score)
        
        # Accuracy score
        accuracy = quality_metrics.get('accuracy', {})
        accuracy_score = (accuracy.get('ocr_confidence', 0) + 
                         accuracy.get('extraction_confidence', 0) + 
                         accuracy.get('text_coherence', 0)) / 3
        scores.append(accuracy_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    # Accessibility and compliance methods (simplified)
    
    def _has_headings(self, content: ExtractedContent) -> bool:
        """Check if content has headings"""
        text = content.text_content or ""
        import re
        # Simple heading detection
        return bool(re.search(r'^[A-Z][A-Z\s]+$', text, re.MULTILINE))
    
    def _assess_reading_level(self, text: str) -> str:
        """Assess reading level"""
        return self._calculate_readability_metrics(text).get('reading_level', 'unknown')
    
    def _identify_primary_language(self, text: str) -> str:
        """Identify primary language"""
        return self._analyze_language_characteristics(text).get('primary_language', 'unknown')
    
    def _detect_personal_data(self, text: str) -> bool:
        """Detect personal data"""
        import re
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        return any(re.search(pattern, text) for pattern in pii_patterns)
    
    def _check_wcag_compliance(self, content: ExtractedContent) -> bool:
        """Check WCAG compliance"""
        return bool(content.text_content and content.text_content.strip())
    
    def _check_section_508_compliance(self, content: ExtractedContent) -> bool:
        """Check Section 508 compliance"""
        return bool(content.text_content and content.text_content.strip())
    
    def _check_iso_27001(self, document: DocumentContent) -> Dict[str, Any]:
        """Check ISO 27001 considerations"""
        return {
            'secure_transmission': document.url.startswith('https://'),
            'data_classification': 'public'  # Would implement actual classification
        }
    
    def _check_hipaa_compliance(self, text: str) -> Dict[str, Any]:
        """Check HIPAA compliance considerations"""
        return {
            'contains_phi': self._detect_health_patterns(text.encode('utf-8')),
            'requires_encryption': False  # Would implement actual assessment
        }
    
    # Structure detection methods
    
    def _detect_hierarchical_structure(self, text: str) -> bool:
        """Detect hierarchical structure in text"""
        import re
        # Look for numbered sections, headings, etc.
        patterns = [
            r'^\d+\.\s',  # 1. Section
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS HEADINGS
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*$'  # Title Case Headings
        ]
        
        lines = text.split('\n')
        heading_count = 0
        
        for line in lines:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in patterns):
                heading_count += 1
        
        return heading_count >= 2
    
    def _detect_list_structures(self, text: str) -> int:
        """Detect list structures in text"""
        import re
        list_patterns = [
            r'^\s*[-•*]\s',  # Bullet lists
            r'^\s*\d+\.\s',  # Numbered lists
            r'^\s*[a-z]\)\s',  # Lettered lists
        ]
        
        lines = text.split('\n')
        list_items = 0
        
        for line in lines:
            if any(re.match(pattern, line) for pattern in list_patterns):
                list_items += 1
        
        return list_items
    
    def _detect_table_structures(self, text: str) -> int:
        """Detect table structures in text"""
        lines = text.split('\n')
        table_indicators = 0
        
        for line in lines:
            # Look for multiple spaces (column separation)
            if len(line.split()) >= 3 and '   ' in line:
                table_indicators += 1
            # Look for tab characters
            elif '\t' in line and len(line.split('\t')) >= 3:
                table_indicators += 1
            # Look for pipe characters (markdown tables)
            elif line.count('|') >= 2:
                table_indicators += 1
        
        return table_indicators