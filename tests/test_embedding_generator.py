"""
Tests for the Embedding Generation System
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import torch

from src.pipeline.stages.stage3_embedding_search.embedding_generator import (
    EmbeddingGenerator, EmbeddingCache, BatchProcessingResult
)
from src.pipeline.core.models import ContentChunk, Embedding
from src.pipeline.core.exceptions import EmbeddingGenerationError


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Mock configuration for testing"""
        config = Mock()
        config.embedding.model_name = "all-MiniLM-L6-v2"
        config.embedding.model_cache_dir = temp_dir
        config.embedding.batch_size = 4
        config.embedding.max_sequence_length = 512
        config.embedding.device = "cpu"
        config.database.cache_ttl_seconds = 3600
        return config
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample content chunks for testing"""
        return [
            ContentChunk(
                id="chunk1",
                content="This is a test document about machine learning.",
                document_id="doc1",
                page_number=1
            ),
            ContentChunk(
                id="chunk2", 
                content="Natural language processing is a subset of AI.",
                document_id="doc1",
                page_number=1
            ),
            ContentChunk(
                id="chunk3",
                content="Deep learning models require large datasets.",
                document_id="doc1",
                page_number=2
            )
        ]
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.max_seq_length = 512
        
        # Mock encode method to return consistent embeddings
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            
            # Return consistent embeddings based on text length
            embeddings = []
            for text in texts:
                # Create deterministic embedding based on text hash
                embedding = np.random.RandomState(hash(text) % 2**32).random(384).astype(np.float32)
                embeddings.append(embedding)
            
            if len(embeddings) == 1:
                return torch.tensor(embeddings[0])
            return torch.tensor(embeddings)
        
        mock_model.encode = mock_encode
        return mock_model
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_initialization(self, mock_st_class, mock_get_config, mock_config, temp_dir):
        """Test EmbeddingGenerator initialization"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = Mock()
        
        generator = EmbeddingGenerator()
        
        assert generator.config == mock_config
        assert generator.model is not None
        assert generator.embeddings_cache_dir.exists()
        assert generator.stats['total_embeddings_generated'] == 0
        mock_st_class.assert_called_once()
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_single_embedding_generation(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test single embedding generation"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        content = "This is a test sentence for embedding generation."
        
        embedding = generator.generate_single_embedding(content)
        
        assert isinstance(embedding, Embedding)
        assert len(embedding.vector) == 384  # MiniLM embedding dimension
        assert embedding.model_name == "all-MiniLM-L6-v2"
        assert embedding.dimension == 384
        assert generator.stats['total_embeddings_generated'] == 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_batch_embedding_generation(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test batch embedding generation"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        contents = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        result = generator.generate_batch_embeddings(contents)
        
        assert isinstance(result, BatchProcessingResult)
        assert len(result.embeddings) == 3
        assert result.cache_hits == 0
        assert result.cache_misses == 3
        assert result.processing_time > 0
        assert len(result.errors) == 0
        
        # Check individual embeddings
        for embedding in result.embeddings:
            assert isinstance(embedding, Embedding)
            assert len(embedding.vector) == 384
            assert embedding.model_name == "all-MiniLM-L6-v2"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_caching_functionality(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test embedding caching"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        content = "This is a test sentence for caching."
        
        # First generation - should be cache miss
        embedding1 = generator.generate_single_embedding(content)
        assert generator.stats['cache_misses'] == 1
        assert generator.stats['cache_hits'] == 0
        
        # Second generation - should be cache hit
        embedding2 = generator.generate_single_embedding(content)
        assert generator.stats['cache_misses'] == 1
        assert generator.stats['cache_hits'] == 1
        
        # Embeddings should be identical
        assert embedding1.vector == embedding2.vector
        assert embedding1.model_name == embedding2.model_name
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_content_chunks_processing(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer, sample_chunks):
        """Test processing ContentChunk objects"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        embeddings = generator.create_embeddings(sample_chunks)
        
        assert len(embeddings) == 3
        
        # Check that chunk IDs are properly assigned
        for i, (chunk, embedding) in enumerate(zip(sample_chunks, embeddings)):
            assert embedding.chunk_id == chunk.id
            assert len(embedding.vector) == 384
            assert embedding.model_name == "all-MiniLM-L6-v2"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_empty_content_handling(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test handling of empty content"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Test empty string
        with pytest.raises(EmbeddingGenerationError):
            generator.generate_single_embedding("")
        
        # Test whitespace only
        with pytest.raises(EmbeddingGenerationError):
            generator.generate_single_embedding("   ")
        
        # Test batch with empty contents
        contents = ["Valid content", "", "   ", "Another valid content"]
        result = generator.generate_batch_embeddings(contents)
        
        # Should have 2 valid embeddings and 2 errors
        assert len(result.embeddings) == 2
        assert len(result.errors) == 2
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_embedding_validation(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test embedding quality validation"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Test valid embeddings
        valid_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        valid_contents = ["content1", "content2"]
        
        validation = generator._validate_embedding_quality(valid_embeddings, valid_contents)
        assert validation["valid"] is True
        assert validation["dimension"] == 3
        assert validation["count"] == 2
        
        # Test inconsistent dimensions
        invalid_embeddings = [[0.1, 0.2], [0.4, 0.5, 0.6]]
        validation = generator._validate_embedding_quality(invalid_embeddings, valid_contents)
        assert validation["valid"] is False
        assert "Inconsistent embedding dimensions" in validation["reason"]
        
        # Test NaN values
        nan_embeddings = [[0.1, float('nan'), 0.3]]
        validation = generator._validate_embedding_quality(nan_embeddings, ["content"])
        assert validation["valid"] is False
        assert "Invalid values" in validation["reason"]
        
        # Test zero vectors
        zero_embeddings = [[0.0, 0.0, 0.0]]
        validation = generator._validate_embedding_quality(zero_embeddings, ["content"])
        assert validation["valid"] is False
        assert "Zero or near-zero embedding" in validation["reason"]
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_cache_management(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test cache management functionality"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Generate some embeddings to populate cache
        contents = ["content1", "content2", "content3"]
        generator.generate_batch_embeddings(contents)
        
        # Check cache stats
        stats = generator.get_embedding_stats()
        assert stats['memory_cache_size'] == 3
        assert stats['cache_hit_rate'] == 0.0  # First generation, all misses
        
        # Clear cache
        generator.clear_cache()
        stats = generator.get_embedding_stats()
        assert stats['memory_cache_size'] == 0
        
        # Test cache expiration cleanup
        # Add entries to cache with old timestamps
        old_time = datetime.now() - timedelta(hours=2)
        cache_key = generator._get_cache_key("old_content", "test_model")
        
        with generator.cache_lock:
            generator.memory_cache[cache_key] = EmbeddingCache(
                embedding=[0.1, 0.2, 0.3],
                model_name="test_model",
                content_hash="hash",
                created_at=old_time
            )
        
        # Cleanup should remove expired entries
        removed_count = generator.cleanup_expired_cache()
        assert removed_count >= 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_performance_stats(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test performance statistics tracking"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Generate some embeddings
        contents = ["content1", "content2", "content3"]
        result = generator.generate_batch_embeddings(contents)
        
        stats = generator.get_embedding_stats()
        
        # Check basic stats
        assert stats['total_embeddings_generated'] == 3
        assert stats['total_processing_time'] > 0
        assert stats['batch_operations'] == 1
        assert stats['average_batch_size'] == 3.0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['average_processing_time_per_embedding'] > 0
        
        # Check model info
        assert stats['model_name'] == "all-MiniLM-L6-v2"
        assert stats['model_device'] == "cpu"
        assert stats['batch_size'] == 4
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_error_handling(self, mock_st_class, mock_get_config, mock_config):
        """Test error handling in embedding generation"""
        mock_get_config.return_value = mock_config
        
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_st_class.return_value = mock_model
        
        generator = EmbeddingGenerator()
        
        # Test single embedding error
        with pytest.raises(EmbeddingGenerationError):
            generator.generate_single_embedding("test content")
        
        # Test batch embedding error handling
        contents = ["content1", "content2"]
        result = generator.generate_batch_embeddings(contents)
        
        # Should handle errors gracefully
        assert len(result.embeddings) == 0
        assert len(result.errors) > 0
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_disk_cache_persistence(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer, temp_dir):
        """Test disk cache persistence"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        # Create first generator instance
        generator1 = EmbeddingGenerator()
        content = "test content for disk cache"
        
        # Generate embedding (should create disk cache)
        embedding1 = generator1.generate_single_embedding(content)
        
        # Create second generator instance (simulating restart)
        generator2 = EmbeddingGenerator()
        
        # Generate same embedding (should hit disk cache)
        embedding2 = generator2.generate_single_embedding(content)
        
        # Should be cache hit for second generator
        assert generator2.stats['cache_hits'] == 1
        assert generator2.stats['cache_misses'] == 0
        
        # Embeddings should be identical
        assert embedding1.vector == embedding2.vector
    
    @pytest.mark.asyncio
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    async def test_async_embedding_generation(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer, sample_chunks):
        """Test async embedding generation"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        embeddings = await generator.create_embeddings_async(sample_chunks)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, Embedding)
            assert len(embedding.vector) == 384


class TestEmbeddingCache:
    """Test suite for EmbeddingCache"""
    
    def test_cache_entry_creation(self):
        """Test EmbeddingCache creation"""
        embedding = [0.1, 0.2, 0.3, 0.4]
        cache_entry = EmbeddingCache(
            embedding=embedding,
            model_name="test-model",
            content_hash="abc123",
            created_at=datetime.now()
        )
        
        assert cache_entry.embedding == embedding
        assert cache_entry.model_name == "test-model"
        assert cache_entry.content_hash == "abc123"
        assert cache_entry.access_count == 0
        assert isinstance(cache_entry.last_accessed, datetime)


class TestBatchProcessingResult:
    """Test suite for BatchProcessingResult"""
    
    def test_result_creation(self):
        """Test BatchProcessingResult creation"""
        embeddings = [
            Embedding(chunk_id="1", vector=[0.1, 0.2], model_name="test", dimension=2),
            Embedding(chunk_id="2", vector=[0.3, 0.4], model_name="test", dimension=2)
        ]
        
        result = BatchProcessingResult(
            embeddings=embeddings,
            processing_time=1.5,
            cache_hits=1,
            cache_misses=1,
            errors=["test error"]
        )
        
        assert len(result.embeddings) == 2
        assert result.processing_time == 1.5
        assert result.cache_hits == 1
        assert result.cache_misses == 1
        assert result.errors == ["test error"]


if __name__ == "__main__":
    pytest.main([__file__])