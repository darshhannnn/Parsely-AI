"""
Tests for Intelligent Content Chunker
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline.stages.stage2_llm_parser.content_chunker import (
    ChunkType,
    ChunkingStrategy,
    ChunkMetadata,
    ContentChunk,
    ChunkingConfig,
    DocumentStructureAnalyzer,
    SemanticChunker,
    IntelligentContentChunker
)
from src.pipeline.stages.stage2_llm_parser.llm_integration import (
    LLMManager,
    LLMConfig,
    LLMProvider,
    LLMRequest,
    LLMResponse
)
from src.pipeline.core.exceptions import ChunkingError


class TestChunkMetadata:
    """Test chunk metadata"""
    
    def test_chunk_metadata_creation(self):
        """Test chunk metadata creation with defaults"""
        metadata = ChunkMetadata()
        
        assert metadata.page_number is None
        assert metadata.section_title is None
        assert metadata.section_level == 0
        assert metadata.paragraph_index == 0
        assert metadata.sentence_count == 0
        assert metadata.word_count == 0
        assert metadata.char_count == 0
        assert metadata.language == "en"
        assert metadata.confidence_score == 1.0
        assert metadata.relationships == []
        assert metadata.tags == []
        assert isinstance(metadata.created_at, datetime)
    
    def test_chunk_metadata_with_values(self):
        """Test chunk metadata with custom values"""
        metadata = ChunkMetadata(
            page_number=5,
            section_title="Introduction",
            section_level=1,
            word_count=150,
            relationships=["next:chunk_2"],
            tags=["important"]
        )
        
        assert metadata.page_number == 5
        assert metadata.section_title == "Introduction"
        assert metadata.section_level == 1
        assert metadata.word_count == 150
        assert metadata.relationships == ["next:chunk_2"]
        assert metadata.tags == ["important"]


class TestContentChunk:
    """Test content chunk"""
    
    def test_content_chunk_creation(self):
        """Test content chunk creation"""
        content = "This is a test chunk with multiple sentences. It contains important information."
        metadata = ChunkMetadata(page_number=1)
        
        chunk = ContentChunk(
            id="test_chunk_1",
            content=content,
            document_id="doc_123",
            chunk_type=ChunkType.PARAGRAPH,
            start_position=0,
            end_position=len(content),
            metadata=metadata
        )
        
        assert chunk.id == "test_chunk_1"
        assert chunk.content == content
        assert chunk.document_id == "doc_123"
        assert chunk.chunk_type == ChunkType.PARAGRAPH
        assert chunk.start_position == 0
        assert chunk.end_position == len(content)
        assert chunk.metadata.char_count == len(content)
        assert chunk.metadata.word_count == len(content.split())
        assert chunk.metadata.sentence_count == 2  # Two sentences
    
    def test_chunk_id_generation(self):
        """Test automatic chunk ID generation"""
        chunk = ContentChunk(
            id="",  # Empty ID should trigger generation
            content="Test content",
            document_id="doc_123",
            chunk_type=ChunkType.PARAGRAPH,
            start_position=0,
            end_position=12,
            metadata=ChunkMetadata()
        )
        
        assert chunk.id != ""
        assert chunk.id.startswith("doc_123_0_")
        assert len(chunk.id.split("_")) == 4  # doc_123_0_hash
    
    def test_chunk_context_window(self):
        """Test chunk context window"""
        chunk = ContentChunk(
            id="test_chunk",
            content="Test content",
            document_id="doc_123",
            chunk_type=ChunkType.PARAGRAPH,
            start_position=100,
            end_position=112,
            metadata=ChunkMetadata()
        )
        
        context = chunk.get_context_window(50)
        assert "[Context: 50-162]" in context
        assert "Test content" in context
    
    def test_chunk_to_dict(self):
        """Test chunk serialization to dictionary"""
        metadata = ChunkMetadata(page_number=2, section_title="Test Section")
        chunk = ContentChunk(
            id="test_chunk",
            content="Test content",
            document_id="doc_123",
            chunk_type=ChunkType.PARAGRAPH,
            start_position=0,
            end_position=12,
            metadata=metadata
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict["id"] == "test_chunk"
        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["document_id"] == "doc_123"
        assert chunk_dict["chunk_type"] == "paragraph"
        assert chunk_dict["metadata"]["page_number"] == 2
        assert chunk_dict["metadata"]["section_title"] == "Test Section"


class TestChunkingConfig:
    """Test chunking configuration"""
    
    def test_chunking_config_defaults(self):
        """Test chunking config with defaults"""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.HYBRID
        assert config.max_chunk_size == 1000
        assert config.min_chunk_size == 100
        assert config.overlap_size == 50
        assert config.preserve_sentences is True
        assert config.preserve_paragraphs is True
        assert config.use_llm_for_semantic_boundaries is True
        assert config.language == "en"
    
    def test_chunking_config_validation(self):
        """Test chunking config validation"""
        # Valid config
        config = ChunkingConfig(max_chunk_size=1000, min_chunk_size=100, overlap_size=50)
        assert config.max_chunk_size == 1000
        
        # Invalid: max_chunk_size <= min_chunk_size
        with pytest.raises(ValueError, match="max_chunk_size must be greater than min_chunk_size"):
            ChunkingConfig(max_chunk_size=100, min_chunk_size=100)
        
        # Invalid: overlap_size >= max_chunk_size
        with pytest.raises(ValueError, match="overlap_size must be less than max_chunk_size"):
            ChunkingConfig(max_chunk_size=100, overlap_size=100)


class TestDocumentStructureAnalyzer:
    """Test document structure analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create document structure analyzer"""
        return DocumentStructureAnalyzer()
    
    @pytest.fixture
    def sample_document(self):
        """Sample document with structure"""
        return """# Introduction

This is the introduction paragraph. It provides an overview of the document.

## Section 1: Background

This section contains background information.

### Subsection 1.1: History

Historical information goes here.

## Section 2: Methods

1. First method step
2. Second method step
3. Third method step

The methods section continues with more details.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

## Conclusion

This is the conclusion paragraph.
"""
    
    def test_analyze_structure(self, analyzer, sample_document):
        """Test complete structure analysis"""
        structure = analyzer.analyze_structure(sample_document)
        
        assert "headings" in structure
        assert "paragraphs" in structure
        assert "lists" in structure
        assert "tables" in structure
        assert "sections" in structure
        
        # Check headings
        headings = structure["headings"]
        assert len(headings) >= 4  # Introduction, Section 1, Subsection 1.1, Section 2, Conclusion
        
        # Check first heading
        assert headings[0]["title"] == "Introduction"
        assert headings[0]["level"] == 1
    
    def test_extract_headings(self, analyzer):
        """Test heading extraction"""
        content = """# Main Title
## Subtitle
### Sub-subtitle
1. Numbered Section
1.1 Subsection
2.3.1 Deep subsection"""
        
        headings = analyzer._extract_headings(content)
        
        # Should find markdown headings and numbered headings
        assert len(headings) >= 3
        
        # Check markdown headings
        markdown_headings = [h for h in headings if "number" not in h]
        assert len(markdown_headings) == 3
        assert markdown_headings[0]["title"] == "Main Title"
        assert markdown_headings[0]["level"] == 1
        assert markdown_headings[1]["title"] == "Subtitle"
        assert markdown_headings[1]["level"] == 2
    
    def test_extract_paragraphs(self, analyzer):
        """Test paragraph extraction"""
        content = """First paragraph with some content.

Second paragraph with more content.
This continues the second paragraph.

Third paragraph is shorter."""
        
        paragraphs = analyzer._extract_paragraphs(content)
        
        assert len(paragraphs) == 3
        assert "First paragraph" in paragraphs[0]["content"]
        assert "Second paragraph" in paragraphs[1]["content"]
        assert "Third paragraph" in paragraphs[2]["content"]
        
        # Check metadata
        assert paragraphs[0]["word_count"] > 0
        assert paragraphs[0]["sentence_count"] > 0
    
    def test_extract_lists(self, analyzer):
        """Test list extraction"""
        content = """Some text before.

- First bullet point
- Second bullet point
- Third bullet point

More text.

1. First numbered item
2. Second numbered item
3. Third numbered item

Final text."""
        
        lists = analyzer._extract_lists(content)
        
        assert len(lists) == 2
        
        # Check bullet list
        bullet_list = lists[0]
        assert bullet_list["type"] == "bullet"
        assert len(bullet_list["items"]) == 3
        assert bullet_list["items"][0]["text"] == "First bullet point"
        
        # Check numbered list
        numbered_list = lists[1]
        assert numbered_list["type"] == "numbered"
        assert len(numbered_list["items"]) == 3
        assert numbered_list["items"][0]["text"] == "First numbered item"
    
    def test_extract_tables(self, analyzer):
        """Test table extraction"""
        content = """Some text before.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

More text after."""
        
        tables = analyzer._extract_tables(content)
        
        assert len(tables) == 1
        table = tables[0]
        assert table["row_count"] == 3  # Header + 2 data rows
        assert "Header 1" in table["content"]
        assert "Data 1" in table["content"]
    
    def test_identify_sections(self, analyzer, sample_document):
        """Test section identification"""
        structure = analyzer.analyze_structure(sample_document)
        sections = structure["sections"]
        
        assert len(sections) >= 4
        
        # Check first section (Introduction)
        intro_section = sections[0]
        assert intro_section["title"] == "Introduction"
        assert intro_section["level"] == 1
        assert "introduction paragraph" in intro_section["content"].lower()


class TestSemanticChunker:
    """Test semantic chunker"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock(spec=LLMManager)
        
        # Mock response with boundary positions
        mock_response = Mock()
        mock_response.content = "150, 320, 480, 650"
        manager.generate.return_value = mock_response
        
        return manager
    
    @pytest.fixture
    def semantic_chunker(self, mock_llm_manager):
        """Create semantic chunker with mock LLM"""
        return SemanticChunker(mock_llm_manager)
    
    def test_identify_semantic_boundaries(self, semantic_chunker, mock_llm_manager):
        """Test semantic boundary identification"""
        content = "This is a test document. " * 100  # Create content of reasonable length
        
        boundaries = semantic_chunker.identify_semantic_boundaries(content, max_chunk_size=1000)
        
        # Should call LLM manager
        mock_llm_manager.generate.assert_called_once()
        
        # Should return parsed boundaries
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0
        assert all(isinstance(b, int) for b in boundaries)
        assert all(0 < b < len(content) for b in boundaries)
    
    def test_parse_boundary_positions(self, semantic_chunker):
        """Test parsing boundary positions from LLM response"""
        response = "The boundaries are at positions 150, 320, 480, and 650."
        content_length = 1000
        
        positions = semantic_chunker._parse_boundary_positions(response, content_length)
        
        assert positions == [150, 320, 480, 650]
    
    def test_parse_boundary_positions_invalid(self, semantic_chunker):
        """Test parsing with invalid positions"""
        response = "Boundaries at 50, 1500, -10, abc, 300"  # Mix of valid/invalid
        content_length = 1000
        
        positions = semantic_chunker._parse_boundary_positions(response, content_length)
        
        # Should only include valid positions (50, 300)
        assert positions == [50, 300]
    
    def test_fallback_boundaries(self, semantic_chunker):
        """Test fallback boundary detection"""
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        boundaries = semantic_chunker._fallback_boundaries(content, max_chunk_size=1000)
        
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0
    
    def test_llm_failure_fallback(self, mock_llm_manager, semantic_chunker):
        """Test fallback when LLM fails"""
        # Make LLM manager raise exception
        mock_llm_manager.generate.side_effect = Exception("LLM API error")
        
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        boundaries = semantic_chunker.identify_semantic_boundaries(content, max_chunk_size=1000)
        
        # Should still return boundaries (from fallback)
        assert isinstance(boundaries, list)


class TestIntelligentContentChunker:
    """Test intelligent content chunker"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic chunking configuration"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH_BASED,
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=25,
            use_llm_for_semantic_boundaries=False
        )
    
    @pytest.fixture
    def semantic_config(self):
        """Semantic chunking configuration"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=25,
            use_llm_for_semantic_boundaries=True
        )
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock(spec=LLMManager)
        mock_response = Mock()
        mock_response.content = "250, 500, 750"
        manager.generate.return_value = mock_response
        return manager
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for chunking"""
        return """# Introduction

This is the introduction section of the document. It provides an overview of what will be covered in the following sections.

## Background

The background section contains important context information. This information helps readers understand the motivation and context for the work described in this document.

## Methods

The methods section describes the approach taken. It includes detailed steps and procedures that were followed during the research or development process.

### Data Collection

Data was collected using various methods. The collection process was systematic and thorough to ensure data quality and completeness.

### Analysis

The analysis phase involved processing the collected data. Multiple analytical techniques were applied to extract meaningful insights from the raw data.

## Results

The results section presents the findings from the analysis. Key discoveries and patterns are highlighted and discussed in detail.

## Conclusion

The conclusion summarizes the main findings and their implications. It also suggests areas for future research and development."""
    
    def test_chunker_initialization(self, basic_config):
        """Test chunker initialization"""
        chunker = IntelligentContentChunker(basic_config)
        
        assert chunker.config == basic_config
        assert chunker.llm_manager is None
        assert chunker.semantic_chunker is None
        assert chunker.structure_analyzer is not None
    
    def test_chunker_with_llm(self, semantic_config, mock_llm_manager):
        """Test chunker initialization with LLM"""
        chunker = IntelligentContentChunker(semantic_config, mock_llm_manager)
        
        assert chunker.config == semantic_config
        assert chunker.llm_manager == mock_llm_manager
        assert chunker.semantic_chunker is not None
    
    def test_chunk_content_empty(self, basic_config):
        """Test chunking empty content"""
        chunker = IntelligentContentChunker(basic_config)
        
        with pytest.raises(ChunkingError, match="Empty content provided"):
            chunker.chunk_content("", "doc_123")
        
        with pytest.raises(ChunkingError, match="Empty content provided"):
            chunker.chunk_content("   ", "doc_123")
    
    def test_fixed_size_chunking(self, sample_content):
        """Test fixed-size chunking"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=300,
            min_chunk_size=50,
            overlap_size=50,
            preserve_sentences=True
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ContentChunk) for chunk in chunks)
        assert all(chunk.document_id == "doc_123" for chunk in chunks)
        assert all(len(chunk.content) <= config.max_chunk_size * 1.2 for chunk in chunks)  # Allow tolerance
        
        # Check that chunks cover the content
        total_content_length = sum(len(chunk.content) for chunk in chunks)
        assert total_content_length > 0
    
    def test_sentence_based_chunking(self, sample_content):
        """Test sentence-based chunking"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE_BASED,
            max_chunk_size=400,
            min_chunk_size=50,
            overlap_size=25
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(chunk.chunk_type == ChunkType.SENTENCE for chunk in chunks)
        
        # Check that sentences are preserved (no broken sentences)
        for chunk in chunks:
            # Should end with sentence-ending punctuation or be at document end
            assert chunk.content.strip()[-1] in '.!?' or chunk == chunks[-1]
    
    def test_paragraph_based_chunking(self, sample_content):
        """Test paragraph-based chunking"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH_BASED,
            max_chunk_size=600,
            min_chunk_size=50,
            overlap_size=25
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(chunk.chunk_type == ChunkType.PARAGRAPH for chunk in chunks)
        
        # Check that paragraphs are preserved
        for chunk in chunks:
            # Should contain complete paragraphs (double newlines or single paragraphs)
            assert chunk.content.strip()
    
    def test_structure_aware_chunking(self, sample_content):
        """Test structure-aware chunking"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            max_chunk_size=800,
            min_chunk_size=50,
            overlap_size=25
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(chunk.chunk_type == ChunkType.SECTION for chunk in chunks)
        
        # Check that section metadata is preserved
        section_chunks = [chunk for chunk in chunks if chunk.metadata.section_title]
        assert len(section_chunks) > 0
        
        # Should have chunks for major sections
        section_titles = [chunk.metadata.section_title for chunk in section_chunks]
        assert any("Introduction" in title for title in section_titles if title)
    
    def test_hybrid_chunking(self, sample_content):
        """Test hybrid chunking"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=500,
            min_chunk_size=50,
            overlap_size=25,
            use_llm_for_semantic_boundaries=False  # Disable LLM for testing
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ContentChunk) for chunk in chunks)
        
        # Should have a mix of chunk types or at least section-based chunks
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert ChunkType.SECTION in chunk_types or ChunkType.MIXED in chunk_types
    
    def test_semantic_chunking_with_llm(self, sample_content, mock_llm_manager):
        """Test semantic chunking with LLM"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=400,
            min_chunk_size=50,
            overlap_size=25,
            use_llm_for_semantic_boundaries=True
        )
        
        chunker = IntelligentContentChunker(config, mock_llm_manager)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        assert len(chunks) > 0
        assert mock_llm_manager.generate.called
        
        # Check chunk properties
        for chunk in chunks:
            assert chunk.document_id == "doc_123"
            assert len(chunk.content) >= config.min_chunk_size
    
    def test_chunk_metadata_population(self, sample_content):
        """Test that chunk metadata is properly populated"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.STRUCTURE_AWARE,
            max_chunk_size=500,
            min_chunk_size=50
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        # Check that metadata is populated
        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata.word_count > 0
            assert chunk.metadata.char_count > 0
            assert chunk.metadata.sentence_count > 0
            assert chunk.metadata.language == "en"
            assert isinstance(chunk.metadata.created_at, datetime)
    
    def test_chunk_relationships(self, sample_content):
        """Test that chunk relationships are established"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH_BASED,
            max_chunk_size=300,
            min_chunk_size=50,
            overlap_size=25
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        if len(chunks) > 1:
            # First chunk should have next relationship
            assert any("next:" in rel for rel in chunks[0].metadata.relationships)
            
            # Middle chunks should have both previous and next relationships
            if len(chunks) > 2:
                middle_chunk = chunks[1]
                assert any("previous:" in rel for rel in middle_chunk.metadata.relationships)
                assert any("next:" in rel for rel in middle_chunk.metadata.relationships)
            
            # Last chunk should have previous relationship
            assert any("previous:" in rel for rel in chunks[-1].metadata.relationships)
    
    def test_chunk_overlap_calculation(self, sample_content):
        """Test chunk overlap calculation"""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_chunk_size=300,
            min_chunk_size=50,
            overlap_size=50
        )
        
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        if len(chunks) > 1:
            # Check overlap information
            for i, chunk in enumerate(chunks):
                if i > 0:
                    assert chunk.overlap_with_previous >= 0
                if i < len(chunks) - 1:
                    assert chunk.overlap_with_next >= 0
    
    def test_chunk_validation(self):
        """Test chunk validation"""
        config = ChunkingConfig(max_chunks_per_document=5)  # Low limit for testing
        chunker = IntelligentContentChunker(config)
        
        # Create content that would generate many chunks
        long_content = "This is a paragraph. " * 1000
        
        with pytest.raises(ChunkingError, match="Too many chunks generated"):
            chunker.chunk_content(long_content, "doc_123")
    
    def test_unsupported_strategy(self):
        """Test unsupported chunking strategy"""
        # This would require modifying the enum, so we'll test with a mock
        config = ChunkingConfig()
        config.strategy = "unsupported_strategy"  # Invalid strategy
        
        chunker = IntelligentContentChunker(config)
        
        with pytest.raises(ChunkingError, match="Unsupported chunking strategy"):
            chunker.chunk_content("Test content", "doc_123")
    
    def test_chunk_to_dict_serialization(self, sample_content):
        """Test chunk serialization to dictionary"""
        config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH_BASED)
        chunker = IntelligentContentChunker(config)
        chunks = chunker.chunk_content(sample_content, "doc_123")
        
        # Test serialization
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            
            assert isinstance(chunk_dict, dict)
            assert "id" in chunk_dict
            assert "content" in chunk_dict
            assert "document_id" in chunk_dict
            assert "chunk_type" in chunk_dict
            assert "metadata" in chunk_dict
            
            # Check metadata serialization
            metadata = chunk_dict["metadata"]
            assert isinstance(metadata, dict)
            assert "word_count" in metadata
            assert "char_count" in metadata
            assert "created_at" in metadata


class TestChunkingEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_very_short_content(self):
        """Test chunking very short content"""
        config = ChunkingConfig(min_chunk_size=50)
        chunker = IntelligentContentChunker(config)
        
        short_content = "Short."
        chunks = chunker.chunk_content(short_content, "doc_123")
        
        # Should still create a chunk even if below minimum size
        assert len(chunks) >= 1
    
    def test_content_with_special_characters(self):
        """Test content with special characters and unicode"""
        config = ChunkingConfig()
        chunker = IntelligentContentChunker(config)
        
        special_content = """
        Content with special characters: @#$%^&*()
        Unicode characters: café, naïve, résumé
        Emojis: 😀 🎉 📄
        Mathematical symbols: ∑ ∫ ∞ ≈
        """
        
        chunks = chunker.chunk_content(special_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
    
    def test_content_with_only_whitespace_paragraphs(self):
        """Test content with paragraphs containing only whitespace"""
        config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH_BASED)
        chunker = IntelligentContentChunker(config)
        
        content_with_empty_paras = """
        First paragraph with content.
        
        
        
        Second paragraph after empty ones.
        
        
        Third paragraph.
        """
        
        chunks = chunker.chunk_content(content_with_empty_paras, "doc_123")
        
        assert len(chunks) > 0
        # Should skip empty paragraphs
        assert all(chunk.content.strip() for chunk in chunks)
    
    def test_content_without_clear_structure(self):
        """Test content without clear paragraph or sentence structure"""
        config = ChunkingConfig()
        chunker = IntelligentContentChunker(config)
        
        unstructured_content = "word1 word2 word3 " * 200  # No clear sentences or paragraphs
        
        chunks = chunker.chunk_content(unstructured_content, "doc_123")
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__])