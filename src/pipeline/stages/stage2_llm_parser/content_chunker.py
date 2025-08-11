"""
Intelligent Content Chunking for LLM Document Processing
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from ...core.logging_utils import get_pipeline_logger
from ...core.exceptions import ChunkingError
from ...core.utils import timing_decorator, generate_correlation_id
from .llm_integration import LLMManager, LLMRequest


class ChunkType(Enum):
    """Types of content chunks"""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CLAUSE = "clause"
    SENTENCE = "sentence"
    MIXED = "mixed"


class ChunkingStrategy(Enum):
    """Chunking strategies"""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    STRUCTURE_AWARE = "structure_aware"
    HYBRID = "hybrid"


@dataclass
class ChunkMetadata:
    """Metadata for content chunks"""
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_level: int = 0
    paragraph_index: int = 0
    sentence_count: int = 0
    word_count: int = 0
    char_count: int = 0
    language: str = "en"
    confidence_score: float = 1.0
    relationships: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContentChunk:
    """Represents a chunk of document content"""
    id: str
    content: str
    document_id: str
    chunk_type: ChunkType
    start_position: int
    end_position: int
    metadata: ChunkMetadata
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_chunk_id()
        
        # Update metadata with content statistics
        self.metadata.char_count = len(self.content)
        self.metadata.word_count = len(self.content.split())
        self.metadata.sentence_count = len(self._split_sentences(self.content))
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_{self.start_position}_{content_hash}"
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be enhanced with NLTK or spaCy
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_context_window(self, window_size: int = 100) -> str:
        """Get content with context window"""
        start = max(0, self.start_position - window_size)
        end = self.end_position + window_size
        return f"[Context: {start}-{end}] {self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type.value,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "overlap_with_previous": self.overlap_with_previous,
            "overlap_with_next": self.overlap_with_next,
            "metadata": {
                "page_number": self.metadata.page_number,
                "section_title": self.metadata.section_title,
                "section_level": self.metadata.section_level,
                "paragraph_index": self.metadata.paragraph_index,
                "sentence_count": self.metadata.sentence_count,
                "word_count": self.metadata.word_count,
                "char_count": self.metadata.char_count,
                "language": self.metadata.language,
                "confidence_score": self.metadata.confidence_score,
                "relationships": self.metadata.relationships,
                "tags": self.metadata.tags,
                "created_at": self.metadata.created_at.isoformat()
            }
        }


@dataclass
class ChunkingConfig:
    """Configuration for content chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 50
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    use_llm_for_semantic_boundaries: bool = True
    semantic_similarity_threshold: float = 0.7
    max_chunks_per_document: int = 1000
    language: str = "en"
    document_type: Optional[str] = None
    
    def __post_init__(self):
        # Validate configuration
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        if self.overlap_size >= self.max_chunk_size:
            raise ValueError("overlap_size must be less than max_chunk_size")


class DocumentStructureAnalyzer:
    """Analyzes document structure for intelligent chunking"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
    
    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        structure = {
            "headings": self._extract_headings(content),
            "paragraphs": self._extract_paragraphs(content),
            "lists": self._extract_lists(content),
            "tables": self._extract_tables(content),
            "sections": self._identify_sections(content)
        }
        
        self.logger.debug(f"Document structure analysis complete", 
                         headings=len(structure["headings"]),
                         paragraphs=len(structure["paragraphs"]),
                         lists=len(structure["lists"]))
        
        return structure
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """Extract headings from content"""
        headings = []
        
        # Pattern for markdown-style headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        for i, line in enumerate(content.split('\n')):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append({
                    "level": level,
                    "title": title,
                    "line_number": i,
                    "position": content.find(line)
                })
        
        # Pattern for numbered headings (1., 1.1, etc.)
        numbered_heading_pattern = r'^(\d+(?:\.\d+)*\.?)\s+(.+)$'
        
        for i, line in enumerate(content.split('\n')):
            if not re.match(heading_pattern, line.strip()):  # Skip markdown headings
                match = re.match(numbered_heading_pattern, line.strip())
                if match and len(line.strip()) < 100:  # Likely a heading if short
                    number = match.group(1)
                    title = match.group(2).strip()
                    level = number.count('.') + 1
                    headings.append({
                        "level": level,
                        "title": title,
                        "line_number": i,
                        "position": content.find(line),
                        "number": number
                    })
        
        return sorted(headings, key=lambda x: x["position"])
    
    def _extract_paragraphs(self, content: str) -> List[Dict[str, Any]]:
        """Extract paragraphs from content"""
        paragraphs = []
        
        # Split by double newlines (common paragraph separator)
        paragraph_texts = re.split(r'\n\s*\n', content)
        
        position = 0
        for i, para_text in enumerate(paragraph_texts):
            para_text = para_text.strip()
            if para_text and len(para_text) > 20:  # Filter out very short paragraphs
                paragraphs.append({
                    "index": i,
                    "content": para_text,
                    "start_position": position,
                    "end_position": position + len(para_text),
                    "word_count": len(para_text.split()),
                    "sentence_count": len(re.split(r'[.!?]+', para_text))
                })
            position += len(para_text) + 2  # Account for newlines
        
        return paragraphs
    
    def _extract_lists(self, content: str) -> List[Dict[str, Any]]:
        """Extract lists from content"""
        lists = []
        
        # Pattern for bullet points and numbered lists
        list_patterns = [
            r'^\s*[-*+]\s+(.+)$',  # Bullet points
            r'^\s*\d+\.\s+(.+)$',  # Numbered lists
            r'^\s*[a-zA-Z]\.\s+(.+)$',  # Lettered lists
        ]
        
        current_list = None
        
        for i, line in enumerate(content.split('\n')):
            line = line.rstrip()
            is_list_item = False
            
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    item_text = match.group(1).strip()
                    
                    if current_list is None:
                        current_list = {
                            "start_line": i,
                            "items": [],
                            "type": "bullet" if pattern.startswith(r'^\s*[-*+]') else "numbered"
                        }
                    
                    current_list["items"].append({
                        "text": item_text,
                        "line_number": i
                    })
                    is_list_item = True
                    break
            
            if not is_list_item and current_list is not None:
                # End of current list
                current_list["end_line"] = i - 1
                lists.append(current_list)
                current_list = None
        
        # Handle list that ends at document end
        if current_list is not None:
            current_list["end_line"] = len(content.split('\n')) - 1
            lists.append(current_list)
        
        return lists
    
    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract tables from content (basic implementation)"""
        tables = []
        
        # Simple table detection - lines with multiple pipe characters
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.count('|') >= 2:  # Likely a table row
                # Look for table boundaries
                table_start = i
                table_end = i
                
                # Find table extent
                for j in range(i + 1, len(lines)):
                    if lines[j].count('|') >= 2:
                        table_end = j
                    else:
                        break
                
                if table_end > table_start:  # Multi-row table
                    table_content = '\n'.join(lines[table_start:table_end + 1])
                    tables.append({
                        "start_line": table_start,
                        "end_line": table_end,
                        "content": table_content,
                        "row_count": table_end - table_start + 1
                    })
        
        return tables
    
    def _identify_sections(self, content: str) -> List[Dict[str, Any]]:
        """Identify document sections based on headings"""
        headings = self._extract_headings(content)
        sections = []
        
        for i, heading in enumerate(headings):
            section_start = heading["position"]
            
            # Find section end (next heading of same or higher level)
            section_end = len(content)
            for j in range(i + 1, len(headings)):
                next_heading = headings[j]
                if next_heading["level"] <= heading["level"]:
                    section_end = next_heading["position"]
                    break
            
            section_content = content[section_start:section_end].strip()
            
            sections.append({
                "title": heading["title"],
                "level": heading["level"],
                "start_position": section_start,
                "end_position": section_end,
                "content": section_content,
                "word_count": len(section_content.split())
            })
        
        return sections


class SemanticChunker:
    """Semantic-based content chunking using LLM"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def identify_semantic_boundaries(self, content: str, max_chunk_size: int = 1000) -> List[int]:
        """Identify semantic boundaries in content using LLM"""
        
        # For very long content, process in segments
        if len(content) > 5000:
            return self._process_long_content(content, max_chunk_size)
        
        prompt = f"""
        Analyze the following text and identify the best places to split it into semantic chunks.
        Each chunk should be coherent and self-contained while maintaining context.
        
        Text to analyze:
        {content}
        
        Please identify character positions where natural semantic breaks occur.
        Consider:
        - Paragraph boundaries
        - Topic changes
        - Logical flow breaks
        - Sentence completions
        
        Return only the character positions as a comma-separated list of numbers.
        Example: 150, 320, 480, 650
        """
        
        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt="You are an expert at analyzing text structure and identifying semantic boundaries.",
                max_tokens=200,
                temperature=0.1
            )
            
            response = self.llm_manager.generate(request)
            
            # Parse the response to extract positions
            positions = self._parse_boundary_positions(response.content, len(content))
            
            self.logger.debug(f"Identified {len(positions)} semantic boundaries")
            return positions
            
        except Exception as e:
            self.logger.warning(f"LLM semantic boundary detection failed: {e}")
            # Fallback to simple paragraph-based boundaries
            return self._fallback_boundaries(content, max_chunk_size)
    
    def _process_long_content(self, content: str, max_chunk_size: int) -> List[int]:
        """Process very long content in segments"""
        boundaries = []
        segment_size = 3000  # Process in 3KB segments with overlap
        overlap = 500
        
        for start in range(0, len(content), segment_size - overlap):
            end = min(start + segment_size, len(content))
            segment = content[start:end]
            
            segment_boundaries = self.identify_semantic_boundaries(segment, max_chunk_size)
            
            # Adjust boundaries to global positions
            adjusted_boundaries = [pos + start for pos in segment_boundaries]
            boundaries.extend(adjusted_boundaries)
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        return boundaries
    
    def _parse_boundary_positions(self, response: str, content_length: int) -> List[int]:
        """Parse boundary positions from LLM response"""
        positions = []
        
        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        
        for num_str in numbers:
            try:
                pos = int(num_str)
                if 0 < pos < content_length:
                    positions.append(pos)
            except ValueError:
                continue
        
        return sorted(positions)
    
    def _fallback_boundaries(self, content: str, max_chunk_size: int) -> List[int]:
        """Fallback boundary detection based on paragraphs"""
        boundaries = []
        
        # Find paragraph breaks
        paragraphs = content.split('\n\n')
        position = 0
        
        for para in paragraphs:
            position += len(para) + 2  # Account for double newline
            if position < len(content):
                boundaries.append(position)
        
        return boundaries


class IntelligentContentChunker:
    """Main content chunking class with multiple strategies"""
    
    def __init__(self, config: ChunkingConfig, llm_manager: Optional[LLMManager] = None):
        self.config = config
        self.llm_manager = llm_manager
        self.logger = get_pipeline_logger()
        self.structure_analyzer = DocumentStructureAnalyzer()
        
        if llm_manager and config.use_llm_for_semantic_boundaries:
            self.semantic_chunker = SemanticChunker(llm_manager)
        else:
            self.semantic_chunker = None
    
    @timing_decorator
    def chunk_content(self, content: str, document_id: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[ContentChunk]:
        """Main method to chunk content based on configuration"""
        
        if not content or not content.strip():
            raise ChunkingError("Empty content provided for chunking", document_id)
        
        self.logger.info(f"Starting content chunking", 
                        document_id=document_id,
                        content_length=len(content),
                        strategy=self.config.strategy.value)
        
        try:
            # Analyze document structure
            structure = self.structure_analyzer.analyze_structure(content)
            
            # Choose chunking strategy
            if self.config.strategy == ChunkingStrategy.SEMANTIC:
                chunks = self._semantic_chunking(content, document_id, structure)
            elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
                chunks = self._fixed_size_chunking(content, document_id)
            elif self.config.strategy == ChunkingStrategy.SENTENCE_BASED:
                chunks = self._sentence_based_chunking(content, document_id)
            elif self.config.strategy == ChunkingStrategy.PARAGRAPH_BASED:
                chunks = self._paragraph_based_chunking(content, document_id, structure)
            elif self.config.strategy == ChunkingStrategy.STRUCTURE_AWARE:
                chunks = self._structure_aware_chunking(content, document_id, structure)
            elif self.config.strategy == ChunkingStrategy.HYBRID:
                chunks = self._hybrid_chunking(content, document_id, structure)
            else:
                raise ChunkingError(f"Unsupported chunking strategy: {self.config.strategy}", document_id)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks, content, document_metadata)
            
            # Validate chunks
            self._validate_chunks(chunks, content)
            
            self.logger.info(f"Content chunking completed", 
                           document_id=document_id,
                           chunk_count=len(chunks),
                           avg_chunk_size=sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Content chunking failed", document_id=document_id, error=str(e))
            raise ChunkingError(f"Failed to chunk content: {e}", document_id)
    
    def _semantic_chunking(self, content: str, document_id: str, structure: Dict[str, Any]) -> List[ContentChunk]:
        """Semantic-based chunking using LLM"""
        
        if not self.semantic_chunker:
            self.logger.warning("Semantic chunker not available, falling back to paragraph-based chunking")
            return self._paragraph_based_chunking(content, document_id, structure)
        
        boundaries = self.semantic_chunker.identify_semantic_boundaries(content, self.config.max_chunk_size)
        
        chunks = []
        start_pos = 0
        
        for i, boundary in enumerate(boundaries + [len(content)]):
            chunk_content = content[start_pos:boundary].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    start_position=start_pos,
                    end_position=boundary,
                    chunk_type=ChunkType.MIXED,
                    structure=structure
                )
                chunks.append(chunk)
            
            start_pos = boundary
        
        return chunks
    
    def _fixed_size_chunking(self, content: str, document_id: str) -> List[ContentChunk]:
        """Fixed-size chunking with overlap"""
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = min(start_pos + self.config.max_chunk_size, len(content))
            
            # Adjust end position to preserve sentences if configured
            if self.config.preserve_sentences and end_pos < len(content):
                end_pos = self._find_sentence_boundary(content, end_pos, backward=True)
            
            chunk_content = content[start_pos:end_pos].strip()
            
            if chunk_content:
                chunk = self._create_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    start_position=start_pos,
                    end_position=end_pos,
                    chunk_type=ChunkType.MIXED
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            start_pos = end_pos - self.config.overlap_size
            if start_pos <= 0:
                break
        
        return chunks
    
    def _sentence_based_chunking(self, content: str, document_id: str) -> List[ContentChunk]:
        """Sentence-based chunking"""
        
        sentences = self._split_sentences(content)
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.config.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_content)
                
                chunk = self._create_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    start_position=start_pos,
                    end_position=end_pos,
                    chunk_type=ChunkType.SENTENCE
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.config.overlap_size // 50:] if self.config.overlap_size > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
                start_pos = end_pos - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Handle remaining sentences
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            end_pos = start_pos + len(chunk_content)
            
            chunk = self._create_chunk(
                content=chunk_content,
                document_id=document_id,
                start_position=start_pos,
                end_position=end_pos,
                chunk_type=ChunkType.SENTENCE
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_based_chunking(self, content: str, document_id: str, structure: Dict[str, Any]) -> List[ContentChunk]:
        """Paragraph-based chunking"""
        
        paragraphs = structure.get("paragraphs", [])
        chunks = []
        current_chunk_paras = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para["content"])
            
            if current_size + para_size > self.config.max_chunk_size and current_chunk_paras:
                # Create chunk from current paragraphs
                chunk_content = '\n\n'.join([p["content"] for p in current_chunk_paras])
                start_pos = current_chunk_paras[0]["start_position"]
                end_pos = current_chunk_paras[-1]["end_position"]
                
                chunk = self._create_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    start_position=start_pos,
                    end_position=end_pos,
                    chunk_type=ChunkType.PARAGRAPH,
                    structure=structure
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_paras = current_chunk_paras[-1:] if self.config.overlap_size > 0 else []
                current_chunk_paras = overlap_paras + [para]
                current_size = sum(len(p["content"]) for p in current_chunk_paras)
            else:
                current_chunk_paras.append(para)
                current_size += para_size
        
        # Handle remaining paragraphs
        if current_chunk_paras:
            chunk_content = '\n\n'.join([p["content"] for p in current_chunk_paras])
            start_pos = current_chunk_paras[0]["start_position"]
            end_pos = current_chunk_paras[-1]["end_position"]
            
            chunk = self._create_chunk(
                content=chunk_content,
                document_id=document_id,
                start_position=start_pos,
                end_position=end_pos,
                chunk_type=ChunkType.PARAGRAPH,
                structure=structure
            )
            chunks.append(chunk)
        
        return chunks
    
    def _structure_aware_chunking(self, content: str, document_id: str, structure: Dict[str, Any]) -> List[ContentChunk]:
        """Structure-aware chunking based on document sections"""
        
        sections = structure.get("sections", [])
        chunks = []
        
        for section in sections:
            section_content = section["content"]
            
            if len(section_content) <= self.config.max_chunk_size:
                # Section fits in one chunk
                chunk = self._create_chunk(
                    content=section_content,
                    document_id=document_id,
                    start_position=section["start_position"],
                    end_position=section["end_position"],
                    chunk_type=ChunkType.SECTION,
                    structure=structure
                )
                chunk.metadata.section_title = section["title"]
                chunk.metadata.section_level = section["level"]
                chunks.append(chunk)
            else:
                # Split large section into smaller chunks
                section_chunks = self._fixed_size_chunking(section_content, document_id)
                
                # Update metadata for section chunks
                for chunk in section_chunks:
                    chunk.chunk_type = ChunkType.SECTION
                    chunk.metadata.section_title = section["title"]
                    chunk.metadata.section_level = section["level"]
                    # Adjust positions relative to full document
                    chunk.start_position += section["start_position"]
                    chunk.end_position += section["start_position"]
                
                chunks.extend(section_chunks)
        
        return chunks
    
    def _hybrid_chunking(self, content: str, document_id: str, structure: Dict[str, Any]) -> List[ContentChunk]:
        """Hybrid chunking combining multiple strategies"""
        
        # Start with structure-aware chunking
        chunks = self._structure_aware_chunking(content, document_id, structure)
        
        # Refine with semantic boundaries if available
        if self.semantic_chunker:
            refined_chunks = []
            
            for chunk in chunks:
                if len(chunk.content) > self.config.max_chunk_size * 1.5:
                    # Further split large chunks using semantic boundaries
                    semantic_boundaries = self.semantic_chunker.identify_semantic_boundaries(
                        chunk.content, self.config.max_chunk_size
                    )
                    
                    if semantic_boundaries:
                        sub_chunks = self._split_chunk_at_boundaries(
                            chunk, semantic_boundaries
                        )
                        refined_chunks.extend(sub_chunks)
                    else:
                        refined_chunks.append(chunk)
                else:
                    refined_chunks.append(chunk)
            
            chunks = refined_chunks
        
        return chunks
    
    def _split_chunk_at_boundaries(self, chunk: ContentChunk, boundaries: List[int]) -> List[ContentChunk]:
        """Split a chunk at specified boundaries"""
        sub_chunks = []
        content = chunk.content
        start_pos = 0
        
        for boundary in boundaries + [len(content)]:
            if boundary > start_pos:
                sub_content = content[start_pos:boundary].strip()
                
                if len(sub_content) >= self.config.min_chunk_size:
                    sub_chunk = self._create_chunk(
                        content=sub_content,
                        document_id=chunk.document_id,
                        start_position=chunk.start_position + start_pos,
                        end_position=chunk.start_position + boundary,
                        chunk_type=chunk.chunk_type
                    )
                    
                    # Copy metadata from parent chunk
                    sub_chunk.metadata.section_title = chunk.metadata.section_title
                    sub_chunk.metadata.section_level = chunk.metadata.section_level
                    sub_chunk.metadata.page_number = chunk.metadata.page_number
                    
                    sub_chunks.append(sub_chunk)
                
                start_pos = boundary
        
        return sub_chunks
    
    def _create_chunk(
        self, 
        content: str, 
        document_id: str, 
        start_position: int, 
        end_position: int, 
        chunk_type: ChunkType,
        structure: Optional[Dict[str, Any]] = None
    ) -> ContentChunk:
        """Create a content chunk with metadata"""
        
        metadata = ChunkMetadata(
            language=self.config.language,
            confidence_score=1.0
        )
        
        # Determine page number from structure if available
        if structure and "sections" in structure:
            for section in structure["sections"]:
                if (section["start_position"] <= start_position <= section["end_position"]):
                    metadata.section_title = section.get("title")
                    metadata.section_level = section.get("level", 0)
                    break
        
        chunk = ContentChunk(
            id="",  # Will be auto-generated
            content=content,
            document_id=document_id,
            chunk_type=chunk_type,
            start_position=start_position,
            end_position=end_position,
            metadata=metadata
        )
        
        return chunk
    
    def _post_process_chunks(
        self, 
        chunks: List[ContentChunk], 
        original_content: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContentChunk]:
        """Post-process chunks to add relationships and final metadata"""
        
        if not chunks:
            return chunks
        
        # Add chunk relationships
        for i, chunk in enumerate(chunks):
            relationships = []
            
            if i > 0:
                relationships.append(f"previous:{chunks[i-1].id}")
                # Calculate overlap with previous chunk
                chunk.overlap_with_previous = self._calculate_overlap(
                    chunks[i-1].content, chunk.content
                )
            
            if i < len(chunks) - 1:
                relationships.append(f"next:{chunks[i+1].id}")
                # Calculate overlap with next chunk
                chunk.overlap_with_next = self._calculate_overlap(
                    chunk.content, chunks[i+1].content
                )
            
            chunk.metadata.relationships = relationships
            chunk.metadata.paragraph_index = i
        
        # Add document-level metadata if provided
        if document_metadata:
            for chunk in chunks:
                if "page_number" in document_metadata:
                    chunk.metadata.page_number = document_metadata["page_number"]
                if "language" in document_metadata:
                    chunk.metadata.language = document_metadata["language"]
                if "tags" in document_metadata:
                    chunk.metadata.tags.extend(document_metadata["tags"])
        
        return chunks
    
    def _calculate_overlap(self, content1: str, content2: str) -> int:
        """Calculate character overlap between two content strings"""
        if not content1 or not content2:
            return 0
        
        # Simple overlap calculation - find common suffix/prefix
        max_overlap = min(len(content1), len(content2), self.config.overlap_size)
        
        # Check suffix of content1 with prefix of content2
        for i in range(max_overlap, 0, -1):
            if content1[-i:] == content2[:i]:
                return i
        
        return 0
    
    def _validate_chunks(self, chunks: List[ContentChunk], original_content: str) -> None:
        """Validate generated chunks"""
        
        if not chunks:
            raise ChunkingError("No chunks generated from content")
        
        if len(chunks) > self.config.max_chunks_per_document:
            raise ChunkingError(
                f"Too many chunks generated: {len(chunks)} > {self.config.max_chunks_per_document}"
            )
        
        # Validate chunk sizes
        for i, chunk in enumerate(chunks):
            if len(chunk.content) > self.config.max_chunk_size * 1.5:  # Allow some tolerance
                self.logger.warning(
                    f"Chunk {i} exceeds maximum size",
                    chunk_id=chunk.id,
                    size=len(chunk.content),
                    max_size=self.config.max_chunk_size
                )
        
        # Validate chunk IDs are unique
        chunk_ids = [chunk.id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ChunkingError("Duplicate chunk IDs generated")
        
        # Validate content coverage (chunks should cover most of original content)
        total_chunk_length = sum(len(chunk.content) for chunk in chunks)
        coverage_ratio = total_chunk_length / len(original_content) if original_content else 0
        
        if coverage_ratio < 0.8:  # Should cover at least 80% of original content
            self.logger.warning(
                f"Low content coverage: {coverage_ratio:.2%}",
                total_chunk_length=total_chunk_length,
                original_length=len(original_content)
            )
    
    def _find_sentence_boundary(self, content: str, position: int, backward: bool = True) -> int:
        """Find the nearest sentence boundary from a given position"""
        
        sentence_endings = '.!?'
        
        if backward:
            # Look backward for sentence ending
            for i in range(position, max(0, position - 200), -1):
                if i < len(content) and content[i] in sentence_endings:
                    # Make sure it's followed by whitespace or end of content
                    if i + 1 >= len(content) or content[i + 1].isspace():
                        return i + 1
            return max(0, position - 100)  # Fallback
        else:
            # Look forward for sentence ending
            for i in range(position, min(len(content), position + 200)):
                if content[i] in sentence_endings:
                    # Make sure it's followed by whitespace or end of content
                    if i + 1 >= len(content) or content[i + 1].isspace():
                        return i + 1
            return min(len(content), position + 100)  # Fallback
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using improved logic"""
        
        # Handle common abbreviations that shouldn't split sentences
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 'Corp.']
        
        # Temporarily replace abbreviations
        temp_text = text
        for i, abbrev in enumerate(abbreviations):
            temp_text = temp_text.replace(abbrev, f"__ABBREV_{i}__")
        
        # Split on sentence endings followed by whitespace and capital letter
        sentence_pattern = r'[.!?]+(?:\s+(?=[A-Z])|$)'
        sentences = re.split(sentence_pattern, temp_text)
        
        # Restore abbreviations and clean up
        result_sentences = []
        for sentence in sentences:
            if sentence.strip():
                # Restore abbreviations
                for i, abbrev in enumerate(abbreviations):
                    sentence = sentence.replace(f"__ABBREV_{i}__", abbrev)
                result_sentences.append(sentence.strip())
        
        return result_sentences
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about the chunking process"""
        return {
            "strategy": self.config.strategy.value,
            "max_chunk_size": self.config.max_chunk_size,
            "min_chunk_size": self.config.min_chunk_size,
            "overlap_size": self.config.overlap_size,
            "preserve_sentences": self.config.preserve_sentences,
            "preserve_paragraphs": self.config.preserve_paragraphs,
            "use_llm_semantic": self.config.use_llm_for_semantic_boundaries,
            "language": self.config.language,
            "semantic_chunker_available": self.semantic_chunker is not None
        }
    
    def optimize_chunking_config(self, sample_content: str, target_chunk_count: int = 10) -> ChunkingConfig:
        """Optimize chunking configuration based on sample content"""
        
        content_length = len(sample_content)
        target_chunk_size = content_length // target_chunk_count
        
        # Adjust chunk size based on content characteristics
        structure = self.structure_analyzer.analyze_structure(sample_content)
        
        avg_paragraph_length = 0
        if structure["paragraphs"]:
            avg_paragraph_length = sum(len(p["content"]) for p in structure["paragraphs"]) // len(structure["paragraphs"])
        
        # Optimize based on content structure
        if avg_paragraph_length > 0:
            # Adjust chunk size to align with paragraph boundaries
            optimal_chunk_size = max(
                self.config.min_chunk_size,
                min(target_chunk_size, avg_paragraph_length * 3)
            )
        else:
            optimal_chunk_size = target_chunk_size
        
        # Create optimized config
        optimized_config = ChunkingConfig(
            strategy=self.config.strategy,
            max_chunk_size=optimal_chunk_size,
            min_chunk_size=max(50, optimal_chunk_size // 10),
            overlap_size=max(25, optimal_chunk_size // 20),
            preserve_sentences=self.config.preserve_sentences,
            preserve_paragraphs=self.config.preserve_paragraphs,
            use_llm_for_semantic_boundaries=self.config.use_llm_for_semantic_boundaries,
            language=self.config.language
        )
        
        self.logger.info(
            f"Optimized chunking config",
            original_chunk_size=self.config.max_chunk_size,
            optimized_chunk_size=optimal_chunk_size,
            content_length=content_length,
            target_chunks=target_chunk_count
        )
        
        return optimized_config
        if structure and "sections" in structure:
            section_info = self._find_containing_section(start_position, structure["sections"])
            if section_info:
                metadata.section_title = section_info["title"]
                metadata.section_level = section_info["level"]
        
        chunk = ContentChunk(
            id="",  # Will be generated in __post_init__
            content=content,
            document_id=document_id,
            chunk_type=chunk_type,
            start_position=start_position,
            end_position=end_position,
            metadata=metadata
        )
        
        return chunk
    
    def _post_process_chunks(self, chunks: List[ContentChunk], 
                           original_content: str, 
                           document_metadata: Optional[Dict[str, Any]]) -> List[ContentChunk]:
        """Post-process chunks to add overlap and relationships"""
        
        # Add overlap information
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Calculate overlap with previous chunk
                prev_chunk = chunks[i - 1]
                overlap_start = max(chunk.start_position - self.config.overlap_size, prev_chunk.start_position)
                overlap_end = min(chunk.start_position, prev_chunk.end_position)
                chunk.overlap_with_previous = max(0, overlap_end - overlap_start)
            
            if i < len(chunks) - 1:
                # Calculate overlap with next chunk
                next_chunk = chunks[i + 1]
                overlap_start = max(chunk.end_position, next_chunk.start_position - self.config.overlap_size)
                overlap_end = min(chunk.end_position + self.config.overlap_size, next_chunk.start_position)
                chunk.overlap_with_next = max(0, overlap_end - overlap_start)
        
        # Add relationship information
        for i, chunk in enumerate(chunks):
            relationships = []
            
            if i > 0:
                relationships.append(f"previous:{chunks[i-1].id}")
            if i < len(chunks) - 1:
                relationships.append(f"next:{chunks[i+1].id}")
            
            chunk.metadata.relationships = relationships
        
        return chunks
    
    def _validate_chunks(self, chunks: List[ContentChunk], original_content: str):
        """Validate chunk integrity"""
        
        if not chunks:
            raise ChunkingError("No chunks generated from content")
        
        # Check chunk count limit
        if len(chunks) > self.config.max_chunks_per_document:
            raise ChunkingError(f"Too many chunks generated: {len(chunks)} > {self.config.max_chunks_per_document}")
        
        # Check chunk sizes
        for chunk in chunks:
            if len(chunk.content) < self.config.min_chunk_size:
                self.logger.warning(f"Chunk {chunk.id} is smaller than minimum size: {len(chunk.content)}")
            
            if len(chunk.content) > self.config.max_chunk_size * 1.2:  # Allow 20% tolerance
                self.logger.warning(f"Chunk {chunk.id} exceeds maximum size: {len(chunk.content)}")
        
        # Check for duplicate chunks
        chunk_contents = [chunk.content for chunk in chunks]
        if len(chunk_contents) != len(set(chunk_contents)):
            self.logger.warning("Duplicate chunks detected")
        
        self.logger.debug(f"Chunk validation completed", 
                         chunk_count=len(chunks),
                         avg_size=sum(len(c.content) for c in chunks) // len(chunks),
                         min_size=min(len(c.content) for c in chunks),
                         max_size=max(len(c.content) for c in chunks))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Enhanced sentence splitting
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _find_sentence_boundary(self, content: str, position: int, backward: bool = True) -> int:
        """Find the nearest sentence boundary"""
        
        if backward:
            # Look backward for sentence ending
            for i in range(position, max(0, position - 200), -1):
                if content[i:i+1] in '.!?':
                    # Make sure it's not an abbreviation
                    if i < len(content) - 1 and content[i+1].isspace():
                        return i + 1
            return position
        else:
            # Look forward for sentence ending
            for i in range(position, min(len(content), position + 200)):
                if content[i:i+1] in '.!?':
                    if i < len(content) - 1 and content[i+1].isspace():
                        return i + 1
            return position
    
    def _get_page_number(self, position: int, page_info: List[Dict[str, Any]]) -> Optional[int]:
        """Get page number for a given position"""
        for page in page_info:
            if page["start_position"] <= position <= page["end_position"]:
                return page["page_number"]
        return None
    
    def _find_containing_section(self, position: int, sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the section containing a given position"""
        for section in sections:
            if section["start_position"] <= position <= section["end_position"]:
                return section
        return None
    
    def _split_chunk_at_boundaries(self, chunk: ContentChunk, boundaries: List[int]) -> List[ContentChunk]:
        """Split a chunk at semantic boundaries"""
        sub_chunks = []
        content = chunk.content
        start_pos = 0
        
        for boundary in boundaries + [len(content)]:
            if boundary > start_pos:
                sub_content = content[start_pos:boundary].strip()
                
                if len(sub_content) >= self.config.min_chunk_size:
                    sub_chunk = ContentChunk(
                        id="",
                        content=sub_content,
                        document_id=chunk.document_id,
                        chunk_type=chunk.chunk_type,
                        start_position=chunk.start_position + start_pos,
                        end_position=chunk.start_position + boundary,
                        metadata=ChunkMetadata(
                            section_title=chunk.metadata.section_title,
                            section_level=chunk.metadata.section_level,
                            language=chunk.metadata.language
                        )
                    )
                    sub_chunks.append(sub_chunk)
                
                start_pos = boundary
        
        return sub_chunks if sub_chunks else [chunk]