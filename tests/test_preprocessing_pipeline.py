"""
Unit tests for preprocessing pipeline components
"""

import pytest
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipeline.stages.preprocessing.metadata_extractor import MetadataExtractor
from src.pipeline.stages.preprocessing.content_normalizer import (
    ContentNormalizer, NormalizationOptions, NormalizationResult
)
from src.pipeline.stages.preprocessing.temp_file_manager import (
    TempFileManager, TempFileInfo, get_temp_file_manager
)
from src.pipeline.stages.preprocessing.preprocessing_pipeline import (
    PreprocessingPipeline, PreprocessingOptions, PreprocessingResult
)
from src.pipeline.core.models import DocumentContent, ExtractedContent
from src.pipeline.core.interfaces import DocumentType


class TestMetadataExtractor:
    """Test suite for MetadataExtractor"""
    
    @pytest.fixture
    def extractor(self):
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_document(self):
        return DocumentContent(
            url='https://example.com/test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 sample content',
            size_bytes=1024,
            metadata={
                'download_timestamp': time.time(),
                'original_filename': 'test.pdf'
            }
        )
    
    @pytest.fixture
    def sample_extracted_content(self):
        return ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This is a sample document with some content for testing purposes.',
            pages=['Page 1 content'],
            sections={'page_1': 'Page 1 content'},
            metadata={
                'page_count': 1,
                'extraction_method': 'PyPDF2'
            }
        )
    
    def test_extract_comprehensive_metadata(self, extractor, sample_document, sample_extracted_content):
        """Test comprehensive metadata extraction"""
        result = extractor.extract_comprehensive_metadata(sample_document, sample_extracted_content)
        
        assert isinstance(result, dict)
        assert 'source' in result
        assert 'content_analysis' in result
        assert 'technical' in result
        assert 'processing' in result
        assert 'security' in result
        assert 'quality' in result
        assert 'accessibility' in result
        assert 'compliance' in result
    
    def test_extract_source_metadata(self, extractor, sample_document):
        """Test source metadata extraction"""
        result = extractor._extract_source_metadata(sample_document)
        
        assert result['url'] == 'https://example.com/test.pdf'
        assert result['domain'] == 'example.com'
        assert result['filename'] == 'test.pdf'
        assert result['content_type'] == 'application/pdf'
        assert result['size_bytes'] == 1024
        assert 'content_hash' in result
        assert 'md5' in result['content_hash']
        assert 'sha256' in result['content_hash']
    
    def test_analyze_content(self, extractor, sample_extracted_content):
        """Test content analysis"""
        result = extractor._analyze_content(sample_extracted_content)
        
        assert 'statistics' in result
        assert 'language' in result
        assert 'structure' in result
        assert 'readability' in result
        assert 'content_patterns' in result
        
        stats = result['statistics']
        assert stats['word_count'] > 0
        assert stats['character_counts']['total'] > 0
    
    def test_analyze_language_characteristics(self, extractor):
        """Test language analysis"""
        english_text = "The quick brown fox jumps over the lazy dog."
        result = extractor._analyze_language_characteristics(english_text)
        
        assert result['primary_language'] == 'english'
        assert result['confidence'] > 0
    
    def test_calculate_readability_metrics(self, extractor):
        """Test readability calculation"""
        text = "This is a simple sentence. This is another sentence for testing."
        result = extractor._calculate_readability_metrics(text)
        
        assert 'flesch_score' in result
        assert 'reading_level' in result
        assert isinstance(result['flesch_score'], float)
    
    def test_identify_content_patterns(self, extractor):
        """Test content pattern identification"""
        text = "Contact us at test@example.com or call 123-456-7890. Visit https://example.com"
        result = extractor._identify_content_patterns(text)
        
        assert result['email_addresses'] == 1
        assert result['phone_numbers'] == 1
        assert result['urls'] == 1
    
    def test_extract_format_specific_metadata_pdf(self, extractor, sample_extracted_content):
        """Test PDF-specific metadata extraction"""
        sample_extracted_content.metadata = {
            'page_count': 5,
            'pdf_metadata': {'Title': 'Test PDF', 'Author': 'Test Author'}
        }
        
        result = extractor._extract_pdf_metadata(sample_extracted_content)
        
        assert result['page_count'] == 5
        assert result['title'] == 'Test PDF'
        assert result['author'] == 'Test Author'


class TestContentNormalizer:
    """Test suite for ContentNormalizer"""
    
    @pytest.fixture
    def normalizer(self):
        return ContentNormalizer()
    
    @pytest.fixture
    def sample_content(self):
        return ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This  is   a  test   with   extra    spaces\r\nand\r\ndifferent\nline\nendings.',
            metadata={}
        )
    
    def test_normalize_content(self, normalizer, sample_content):
        """Test content normalization"""
        result = normalizer.normalize_content(sample_content)
        
        assert isinstance(result, NormalizationResult)
        assert result.normalized_content != sample_content.text_content
        assert result.original_length > 0
        assert result.normalized_length > 0
        assert len(result.changes_made) > 0
    
    def test_remove_control_characters(self, normalizer):
        """Test control character removal"""
        text_with_control = "Normal text\x00with\x01control\x02characters"
        result, count = normalizer._remove_control_characters(text_with_control)
        
        assert count == 3
        assert '\x00' not in result
        assert '\x01' not in result
        assert '\x02' not in result
    
    def test_normalize_unicode(self, normalizer):
        """Test Unicode normalization"""
        text_with_unicode = "café naïve résumé"  # Contains accented characters
        result, changes = normalizer._normalize_unicode(text_with_unicode)
        
        assert isinstance(result, str)
        assert changes >= 0
    
    def test_normalize_line_endings(self, normalizer):
        """Test line ending normalization"""
        text_with_mixed_endings = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result, changes = normalizer._normalize_line_endings(text_with_mixed_endings)
        
        assert '\r\n' not in result
        assert '\r' not in result
        assert result.count('\n') == 3
        assert changes == 2  # \r\n and \r
    
    def test_normalize_whitespace(self, normalizer):
        """Test whitespace normalization"""
        text_with_special_spaces = "Normal\u00A0space\u2000and\u2003more"
        result, changes = normalizer._normalize_whitespace(text_with_special_spaces)
        
        assert changes > 0
        assert '\u00A0' not in result
        assert '\u2000' not in result
        assert '\u2003' not in result
    
    def test_remove_extra_spaces(self, normalizer):
        """Test extra space removal"""
        text_with_extra_spaces = "This  has    too     many      spaces"
        result, changes = normalizer._remove_extra_spaces(text_with_extra_spaces)
        
        assert changes > 0
        assert '  ' not in result  # No double spaces
    
    def test_normalize_quotes(self, normalizer):
        """Test quote normalization"""
        text_with_smart_quotes = ""Hello" and 'world' with «quotes»"
        result, changes = normalizer._normalize_quotes(text_with_smart_quotes)
        
        assert changes > 0
        assert '"' not in result  # Smart quotes removed
        assert '"' not in result
        assert ''' not in result
        assert ''' not in result
    
    def test_normalize_dashes(self, normalizer):
        """Test dash normalization"""
        text_with_dashes = "This—is—an—em—dash and this–is–an–en–dash"
        result, changes = normalizer._normalize_dashes(text_with_dashes)
        
        assert changes > 0
        assert '—' not in result  # Em dashes removed
        assert '–' not in result  # En dashes removed
    
    def test_calculate_quality_improvements(self, normalizer):
        """Test quality improvement calculation"""
        original = "This  has   ""smart quotes""  and—dashes"
        normalized = "This has \"smart quotes\" and-dashes"
        
        result = normalizer._calculate_quality_improvements(original, normalized)
        
        assert isinstance(result, dict)
        assert 'character_consistency' in result
        assert 'whitespace_consistency' in result
        assert 'quote_consistency' in result


class TestTempFileManager:
    """Test suite for TempFileManager"""
    
    @pytest.fixture
    def temp_manager(self):
        manager = TempFileManager(
            max_age_hours=1,
            max_total_size_mb=10,
            cleanup_interval_minutes=1
        )
        yield manager
        manager.shutdown()
    
    def test_create_temp_file(self, temp_manager):
        """Test temporary file creation"""
        file_path = temp_manager.create_temp_file(suffix='.txt', purpose='test')
        
        assert os.path.exists(file_path)
        assert file_path.endswith('.txt')
        assert temp_manager.get_file_count() == 1
        
        # Cleanup
        temp_manager.cleanup_file(file_path)
    
    def test_temp_file_context_manager(self, temp_manager):
        """Test temporary file context manager"""
        with temp_manager.temp_file(suffix='.txt', purpose='test') as file_path:
            assert os.path.exists(file_path)
            assert file_path.endswith('.txt')
            
            # Write some content
            with open(file_path, 'w') as f:
                f.write("test content")
        
        # File should be cleaned up after context
        assert not os.path.exists(file_path)
    
    def test_update_file_size(self, temp_manager):
        """Test file size updating"""
        file_path = temp_manager.create_temp_file(purpose='test')
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write("test content")
        
        # Update size
        success = temp_manager.update_file_size(file_path)
        assert success
        
        file_info = temp_manager.get_file_info(file_path)
        assert file_info.size_bytes > 0
        
        # Cleanup
        temp_manager.cleanup_file(file_path)
    
    def test_cleanup_expired_files(self, temp_manager):
        """Test expired file cleanup"""
        # Create file with short expiration
        file_path = temp_manager.create_temp_file(
            purpose='test',
            cleanup_after_hours=0.001  # Very short expiration
        )
        
        # Wait a bit
        time.sleep(0.1)
        
        # Force cleanup
        cleaned_count = temp_manager.cleanup_expired_files()
        
        assert cleaned_count >= 0
        assert not os.path.exists(file_path)
    
    def test_get_statistics(self, temp_manager):
        """Test statistics retrieval"""
        # Create some files
        file1 = temp_manager.create_temp_file(purpose='test1')
        file2 = temp_manager.create_temp_file(purpose='test2')
        
        stats = temp_manager.get_statistics()
        
        assert stats['total_files'] == 2
        assert 'test1' in stats['files_by_purpose']
        assert 'test2' in stats['files_by_purpose']
        
        # Cleanup
        temp_manager.cleanup_file(file1)
        temp_manager.cleanup_file(file2)
    
    def test_force_cleanup_by_purpose(self, temp_manager):
        """Test cleanup by purpose"""
        # Create files with different purposes
        file1 = temp_manager.create_temp_file(purpose='test1')
        file2 = temp_manager.create_temp_file(purpose='test2')
        file3 = temp_manager.create_temp_file(purpose='test1')
        
        # Cleanup only test1 files
        cleaned_count = temp_manager.force_cleanup_by_purpose('test1')
        
        assert cleaned_count == 2
        assert not os.path.exists(file1)
        assert not os.path.exists(file3)
        assert os.path.exists(file2)
        
        # Cleanup remaining
        temp_manager.cleanup_file(file2)


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        options = PreprocessingOptions(
            extract_comprehensive_metadata=True,
            normalize_content=True,
            preserve_original_content=True
        )
        return PreprocessingPipeline(options)
    
    @pytest.fixture
    def sample_document(self):
        return DocumentContent(
            url='https://example.com/test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 sample content',
            size_bytes=1024
        )
    
    @pytest.fixture
    def sample_extracted_content(self):
        return ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This  is  a  test   document  with   extra   spaces.',
            pages=['Page 1 content'],
            sections={'page_1': 'Page 1 content'},
            metadata={'page_count': 1}
        )
    
    def test_process_pipeline(self, pipeline, sample_document, sample_extracted_content):
        """Test complete preprocessing pipeline"""
        result = pipeline.process(sample_document, sample_extracted_content)
        
        assert isinstance(result, PreprocessingResult)
        assert result.success
        assert result.processed_content is not None
        assert result.comprehensive_metadata is not None
        assert result.normalization_result is not None
        assert result.processing_time_ms > 0
    
    def test_preserve_original_content(self, pipeline, sample_document, sample_extracted_content):
        """Test original content preservation"""
        result = pipeline.process(sample_document, sample_extracted_content)
        
        assert result.original_content is not None
        assert result.original_content.text_content == sample_extracted_content.text_content
        assert result.processed_content.text_content != sample_extracted_content.text_content
    
    def test_metadata_extraction(self, pipeline, sample_document, sample_extracted_content):
        """Test metadata extraction in pipeline"""
        result = pipeline.process(sample_document, sample_extracted_content)
        
        assert len(result.comprehensive_metadata) > 0
        assert 'source' in result.comprehensive_metadata
        assert 'content_analysis' in result.comprehensive_metadata
        assert 'technical' in result.comprehensive_metadata
    
    def test_content_normalization(self, pipeline, sample_document, sample_extracted_content):
        """Test content normalization in pipeline"""
        result = pipeline.process(sample_document, sample_extracted_content)
        
        assert result.normalization_result is not None
        assert len(result.normalization_result.changes_made) > 0
        assert result.processed_content.text_content != sample_extracted_content.text_content
    
    def test_get_pipeline_summary(self, pipeline, sample_document, sample_extracted_content):
        """Test pipeline summary generation"""
        result = pipeline.process(sample_document, sample_extracted_content)
        summary = pipeline.get_pipeline_summary(result)
        
        assert 'success' in summary
        assert 'processing_time_ms' in summary
        assert 'content_changes' in summary
        assert 'metadata_extracted' in summary
        assert 'normalization_applied' in summary
    
    def test_validate_preprocessing_result(self, pipeline, sample_document, sample_extracted_content):
        """Test preprocessing result validation"""
        result = pipeline.process(sample_document, sample_extracted_content)
        validation_report = pipeline.validate_preprocessing_result(result)
        
        assert 'is_valid' in validation_report
        assert 'validation_errors' in validation_report
        assert 'validation_warnings' in validation_report
        assert 'quality_checks' in validation_report
    
    def test_pipeline_with_disabled_options(self):
        """Test pipeline with disabled options"""
        options = PreprocessingOptions(
            extract_comprehensive_metadata=False,
            normalize_content=False,
            preserve_original_content=False
        )
        pipeline = PreprocessingPipeline(options)
        
        sample_document = DocumentContent(
            url='https://example.com/test.pdf',
            content_type='application/pdf',
            raw_content=b'test content',
            size_bytes=100
        )
        
        sample_content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='Test content',
            metadata={}
        )
        
        result = pipeline.process(sample_document, sample_content)
        
        assert result.success
        assert len(result.comprehensive_metadata) == 0
        assert result.normalization_result is None
        assert result.original_content is None


# Integration tests
class TestPreprocessingIntegration:
    """Integration tests for preprocessing components"""
    
    def test_end_to_end_preprocessing(self):
        """Test end-to-end preprocessing workflow"""
        # Create sample data
        document = DocumentContent(
            url='https://example.com/test.pdf',
            content_type='application/pdf',
            raw_content=b'%PDF-1.4 sample content with unicode café and extra  spaces',
            size_bytes=1024
        )
        
        extracted_content = ExtractedContent(
            document_id='test-id',
            document_type=DocumentType.PDF.value,
            text_content='This  is  a  test   document  with   café  and   extra   spaces.',
            pages=['Page 1 content'],
            sections={'page_1': 'Page 1 content'},
            metadata={'page_count': 1}
        )
        
        # Create pipeline
        options = PreprocessingOptions(
            extract_comprehensive_metadata=True,
            normalize_content=True,
            preserve_original_content=True
        )
        
        with TempFileManager() as temp_manager:
            pipeline = PreprocessingPipeline(options, temp_manager)
            
            # Process
            result = pipeline.process(document, extracted_content)
            
            # Verify results
            assert result.success
            assert result.processed_content.text_content != extracted_content.text_content
            assert len(result.comprehensive_metadata) > 0
            assert result.normalization_result is not None
            assert result.original_content is not None
            
            # Verify metadata categories
            assert 'source' in result.comprehensive_metadata
            assert 'content_analysis' in result.comprehensive_metadata
            assert 'technical' in result.comprehensive_metadata
            assert 'quality' in result.comprehensive_metadata
            
            # Verify normalization
            assert len(result.normalization_result.changes_made) > 0
            assert result.normalization_result.normalized_length <= result.normalization_result.original_length
    
    def test_global_temp_file_manager(self):
        """Test global temp file manager"""
        from src.pipeline.stages.preprocessing.temp_file_manager import (
            get_temp_file_manager, create_managed_temp_file, cleanup_managed_temp_file
        )
        
        # Create file using global manager
        file_path = create_managed_temp_file(suffix='.txt', purpose='test')
        
        assert os.path.exists(file_path)
        
        # Get manager and check file is tracked
        manager = get_temp_file_manager()
        assert manager.get_file_count() >= 1
        
        # Cleanup using global function
        success = cleanup_managed_temp_file(file_path)
        assert success
        assert not os.path.exists(file_path)


if __name__ == '__main__':
    pytest.main([__file__])