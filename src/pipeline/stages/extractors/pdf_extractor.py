"""
Enhanced PDF content extractor with advanced features
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ...core.models import DocumentContent, ExtractedContent
from ...core.interfaces import DocumentType
from ...core.exceptions import ContentExtractionError
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import create_temp_file, cleanup_temp_file, calculate_content_hash, timing_decorator


@dataclass
class PDFPageInfo:
    """Information about a PDF page"""
    page_number: int
    text_content: str
    word_count: int
    char_count: int
    has_images: bool = False
    has_tables: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class PDFStructure:
    """PDF document structure information"""
    total_pages: int
    total_words: int
    total_chars: int
    sections: List[Dict[str, Any]]
    headings: List[Dict[str, Any]]
    tables_detected: int
    images_detected: int


class EnhancedPDFExtractor:
    """Enhanced PDF content extractor with structure preservation"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def extract_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content from PDF with enhanced structure analysis"""
        
        self.logger.info("Starting enhanced PDF content extraction")
        
        try:
            import PyPDF2
            
            # Save content to temporary file
            temp_file = create_temp_file(suffix='.pdf')
            
            try:
                with open(temp_file, 'wb') as f:
                    f.write(document.raw_content)
                
                # Extract content using PyPDF2
                with open(temp_file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Extract pages with detailed information
                    pages_info = self._extract_pages_with_analysis(pdf_reader)
                    
                    # Analyze document structure
                    structure = self._analyze_document_structure(pages_info)
                    
                    # Extract sections and headings
                    sections = self._extract_sections(pages_info)
                    
                    # Combine all text content
                    full_text = []
                    pages = []
                    page_sections = {}
                    
                    for page_info in pages_info:
                        pages.append(page_info.text_content)
                        full_text.append(page_info.text_content)
                        page_sections[f"page_{page_info.page_number}"] = page_info.text_content
                    
                    text_content = '\n\n'.join(full_text)
                    
                    # Enhanced metadata
                    metadata = {
                        'page_count': len(pdf_reader.pages),
                        'total_words': structure.total_words,
                        'total_chars': structure.total_chars,
                        'sections_detected': len(structure.sections),
                        'headings_detected': len(structure.headings),
                        'tables_detected': structure.tables_detected,
                        'images_detected': structure.images_detected,
                        'pdf_metadata': dict(pdf_reader.metadata) if pdf_reader.metadata else {},
                        'extraction_method': 'Enhanced PyPDF2',
                        'structure_analysis': {
                            'sections': structure.sections,
                            'headings': structure.headings
                        },
                        'pages_info': [
                            {
                                'page_number': p.page_number,
                                'word_count': p.word_count,
                                'char_count': p.char_count,
                                'has_images': p.has_images,
                                'has_tables': p.has_tables
                            }
                            for p in pages_info
                        ]
                    }
                    
                    # Add sections from structure analysis
                    for section in structure.sections:
                        page_sections[section['title']] = section['content']
                    
                    return ExtractedContent(
                        document_id=calculate_content_hash(document.raw_content),
                        document_type=DocumentType.PDF.value,
                        text_content=text_content,
                        pages=pages,
                        sections=page_sections,
                        metadata=metadata
                    )
            
            finally:
                cleanup_temp_file(temp_file)
        
        except ImportError:
            raise ContentExtractionError("PyPDF2 library not available for PDF processing", DocumentType.PDF.value)
        except Exception as e:
            raise ContentExtractionError(f"Enhanced PDF extraction failed: {e}", DocumentType.PDF.value)
    
    def _extract_pages_with_analysis(self, pdf_reader) -> List[PDFPageInfo]:
        """Extract pages with detailed analysis"""
        pages_info = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                text_content = page.extract_text()
                
                # Basic text analysis
                word_count = len(text_content.split()) if text_content else 0
                char_count = len(text_content) if text_content else 0
                
                # Detect potential tables (simple heuristic)
                has_tables = self._detect_tables_in_text(text_content)
                
                # Detect potential images (check if page has images)
                has_images = self._detect_images_in_page(page)
                
                page_info = PDFPageInfo(
                    page_number=page_num,
                    text_content=text_content,
                    word_count=word_count,
                    char_count=char_count,
                    has_images=has_images,
                    has_tables=has_tables
                )
                
                pages_info.append(page_info)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze page {page_num}: {e}")
                # Create minimal page info
                pages_info.append(PDFPageInfo(
                    page_number=page_num,
                    text_content="",
                    word_count=0,
                    char_count=0
                ))
        
        return pages_info
    
    def _analyze_document_structure(self, pages_info: List[PDFPageInfo]) -> PDFStructure:
        """Analyze overall document structure"""
        
        total_pages = len(pages_info)
        total_words = sum(p.word_count for p in pages_info)
        total_chars = sum(p.char_count for p in pages_info)
        tables_detected = sum(1 for p in pages_info if p.has_tables)
        images_detected = sum(1 for p in pages_info if p.has_images)
        
        # Extract sections and headings
        sections = self._identify_sections(pages_info)
        headings = self._identify_headings(pages_info)
        
        return PDFStructure(
            total_pages=total_pages,
            total_words=total_words,
            total_chars=total_chars,
            sections=sections,
            headings=headings,
            tables_detected=tables_detected,
            images_detected=images_detected
        )
    
    def _identify_sections(self, pages_info: List[PDFPageInfo]) -> List[Dict[str, Any]]:
        """Identify document sections based on text patterns"""
        sections = []
        current_section = None
        section_content = []
        
        # Common section patterns
        section_patterns = [
            r'^(\d+\.?\s+[A-Z][^.]*?)$',  # Numbered sections
            r'^([A-Z][A-Z\s]{2,}[A-Z])$',  # ALL CAPS headings
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$',  # Title Case headings
        ]
        
        for page_info in pages_info:
            lines = page_info.text_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line matches section pattern
                is_section_header = False
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        # Save previous section
                        if current_section and section_content:
                            sections.append({
                                'title': current_section,
                                'content': '\n'.join(section_content),
                                'page_start': page_info.page_number,
                                'word_count': len(' '.join(section_content).split())
                            })
                        
                        # Start new section
                        current_section = line
                        section_content = []
                        is_section_header = True
                        break
                
                if not is_section_header and current_section:
                    section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(section_content),
                'page_start': pages_info[-1].page_number if pages_info else 1,
                'word_count': len(' '.join(section_content).split())
            })
        
        return sections
    
    def _identify_headings(self, pages_info: List[PDFPageInfo]) -> List[Dict[str, Any]]:
        """Identify document headings"""
        headings = []
        
        heading_patterns = [
            (r'^(\d+\.?\s+[A-Z][^.]*?)$', 'numbered'),
            (r'^([A-Z][A-Z\s]{2,}[A-Z])$', 'all_caps'),
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$', 'title_case'),
        ]
        
        for page_info in pages_info:
            lines = page_info.text_content.split('\n')
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                for pattern, heading_type in heading_patterns:
                    match = re.match(pattern, line)
                    if match:
                        headings.append({
                            'text': line,
                            'type': heading_type,
                            'page': page_info.page_number,
                            'line': line_num + 1,
                            'level': self._determine_heading_level(line, heading_type)
                        })
                        break
        
        return headings
    
    def _determine_heading_level(self, text: str, heading_type: str) -> int:
        """Determine heading level based on text and type"""
        if heading_type == 'numbered':
            # Count dots to determine level (1. vs 1.1. vs 1.1.1.)
            dots = text.count('.')
            return min(dots, 3)  # Max level 3
        elif heading_type == 'all_caps':
            return 1  # Top level
        else:
            return 2  # Default level
    
    def _detect_tables_in_text(self, text: str) -> bool:
        """Detect potential tables in text using heuristics"""
        if not text:
            return False
        
        lines = text.split('\n')
        
        # Look for patterns that suggest tables
        table_indicators = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for multiple spaces (column separation)
            if re.search(r'\s{3,}', line):
                table_indicators += 1
            
            # Check for tab characters
            if '\t' in line:
                table_indicators += 1
            
            # Check for pipe characters (markdown-style tables)
            if '|' in line and line.count('|') >= 2:
                table_indicators += 1
            
            # Check for numeric patterns (common in tables)
            if re.search(r'\d+\.\d+|\d+%|\$\d+', line):
                table_indicators += 1
        
        # If more than 20% of lines have table indicators, likely contains tables
        return table_indicators > len(lines) * 0.2
    
    def _detect_images_in_page(self, page) -> bool:
        """Detect if page contains images"""
        try:
            # Check if page has XObject resources (which may include images)
            if '/Resources' in page and '/XObject' in page['/Resources']:
                xobjects = page['/Resources']['/XObject']
                for obj in xobjects.values():
                    if hasattr(obj, 'get') and obj.get('/Subtype') == '/Image':
                        return True
            return False
        except:
            return False
    
    def _extract_sections(self, pages_info: List[PDFPageInfo]) -> Dict[str, str]:
        """Extract sections from pages"""
        sections = {}
        
        # Add page-based sections
        for page_info in pages_info:
            sections[f"page_{page_info.page_number}"] = page_info.text_content
        
        return sections