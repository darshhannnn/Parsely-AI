"""
Tests for the FAISS Search Engine
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline.stages.stage3_embedding_search.faiss_search_engine import (
    FAISSSearchEngine, FAISSIndexMetadata, SearchConfiguration
)
from src.pipeline.core.models import ContentChunk, Embedding, VectorIndex, SearchResult
from src.pipeline.core.exceptions import FAISSError, SearchError


class TestFAISSSearchEngine:
    """Test suite for FAISSSearchEngine"""
    
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
        config.database.faiss_index_path = temp_dir
        config.database.vector_db_type = "faiss"
        return config
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing"""
        embeddings = []
        for i in range(10):
            # Create deterministic embeddings
            vector = np.random.RandomState(i).random(384).astype(np.float32).tolist()
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector,
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        return embeddings
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample content chunks for testing"""
        chunks = []
        for i in range(10):
            chunk = ContentChunk(
                id=f"chunk_{i}",
                content=f"This is test content for chunk {i}",
                document_id=f"doc_{i // 3}",
                page_number=(i // 3) + 1
            )
            chunks.append(chunk)
        return chunks
    
    @patch('src.pipeline.core.config.get_config')
    def test_initialization(self, mock_get_config, mock_config):
        """Test FAISSSearchEngine initialization"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        
        assert engine.config == mock_config
        assert engine.index is None
        assert engine.index_metadata is None
        assert len(engine.chunk_id_mapping) == 0
        assert engine.stats['total_searches'] == 0
        assert engine.index_dir.exists()
    
    @patch('src.pipeline.core.config.get_config')
    def test_build_flat_index(self, mock_get_config, mock_config, sample_embeddings):
        """Test building a flat FAISS index"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        vector_index = engine.build_index(sample_embeddings)
        
        assert isinstance(vector_index, VectorIndex)
        assert vector_index.index_type == "faiss"
        assert vector_index.dimension == 384
        assert vector_index.total_vectors == 10
        assert engine.index is not None
        assert engine.index.ntotal == 10
        assert len(engine.chunk_id_mapping) == 10
    
    @patch('src.pipeline.core.config.get_config')
    def test_build_ivf_index(self, mock_get_config, mock_config):
        """Test building an IVF FAISS index"""
        mock_get_config.return_value = mock_config
        
        # Create larger dataset to trigger IVF index
        embeddings = []
        for i in range(1500):
            vector = np.random.RandomState(i).random(384).astype(np.float32).tolist()
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector,
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        
        engine = FAISSSearchEngine()
        vector_index = engine.build_index(embeddings)
        
        assert vector_index.total_vectors == 1500
        assert engine.index.is_trained
        assert hasattr(engine.index, 'nprobe')
    
    @patch('src.pipeline.core.config.get_config')
    def test_vector_search(self, mock_get_config, mock_config, sample_embeddings):
        """Test vector similarity search"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        # Use first embedding as query (should return itself as top result)
        query_vector = sample_embeddings[0].vector
        results = engine.search_similar_by_vector(query_vector, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].chunk_id == "chunk_0"  # Should find itself first
        assert results[0].similarity_score > 0.9  # Should be very similar to itself
        
        # Check that results are sorted by similarity
        for i in range(1, len(results)):
            assert results[i-1].similarity_score >= results[i].similarity_score
    
    @patch('src.pipeline.core.config.get_config')
    def test_similarity_threshold(self, mock_get_config, mock_config, sample_embeddings):
        """Test similarity threshold filtering"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        query_vector = sample_embeddings[0].vector
        
        # High threshold should return fewer results
        high_threshold_results = engine.search_similar_by_vector(
            query_vector, top_k=10, similarity_threshold=0.8
        )
        
        # Low threshold should return more results
        low_threshold_results = engine.search_similar_by_vector(
            query_vector, top_k=10, similarity_threshold=0.1
        )
        
        assert len(high_threshold_results) <= len(low_threshold_results)
        
        # All results should meet the threshold
        for result in high_threshold_results:
            assert result.similarity_score >= 0.8
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_persistence(self, mock_get_config, mock_config, sample_embeddings):
        """Test saving and loading index"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        vector_index = engine.build_index(sample_embeddings)
        index_id = vector_index.index_id
        
        # Save index
        success = engine.save_index(index_id)
        assert success
        
        # Verify files exist
        assert engine._get_index_file_path(index_id).exists()
        assert engine._get_metadata_file_path(index_id).exists()
        assert engine._get_mappings_file_path(index_id).exists()
        
        # Clear current index
        engine.clear_index()
        assert engine.index is None
        
        # Load index
        success = engine.load_index(index_id)
        assert success
        assert engine.index is not None
        assert engine.index.ntotal == 10
        assert len(engine.chunk_id_mapping) == 10
        
        # Test search still works
        query_vector = sample_embeddings[0].vector
        results = engine.search_similar_by_vector(query_vector, top_k=3)
        assert len(results) > 0
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_update(self, mock_get_config, mock_config, sample_embeddings):
        """Test updating existing index with new embeddings"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings[:5])  # Build with first 5
        
        assert engine.index.ntotal == 5
        
        # Update with remaining embeddings
        engine.update_index(sample_embeddings[5:])
        
        assert engine.index.ntotal == 10
        assert len(engine.chunk_id_mapping) == 10
        
        # Test search works with all embeddings
        query_vector = sample_embeddings[7].vector  # Use one from the updated batch
        results = engine.search_similar_by_vector(query_vector, top_k=3)
        assert results[0].chunk_id == "chunk_7"
    
    @patch('src.pipeline.core.config.get_config')
    def test_vector_removal(self, mock_get_config, mock_config, sample_embeddings):
        """Test removing vectors from index"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        assert engine.index.ntotal == 10
        
        # Remove some chunks
        chunks_to_remove = ["chunk_0", "chunk_1", "chunk_2"]
        removed_count = engine.remove_vectors(chunks_to_remove)
        
        assert removed_count == 3
        assert engine.index.ntotal == 7
        
        # Verify removed chunks are not in mappings
        for chunk_id in chunks_to_remove:
            assert chunk_id not in engine.chunk_metadata
            assert chunk_id not in engine.embeddings_store
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_optimization(self, mock_get_config, mock_config):
        """Test index optimization"""
        mock_get_config.return_value = mock_config
        
        # Create IVF index for optimization testing
        embeddings = []
        for i in range(1500):
            vector = np.random.RandomState(i).random(384).astype(np.float32).tolist()
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector,
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        
        engine = FAISSSearchEngine()
        engine.build_index(embeddings)
        
        # Optimize index
        optimization_stats = engine.optimize_index()
        
        assert 'original_size' in optimization_stats
        assert 'optimization_time' in optimization_stats
        assert 'optimizations_applied' in optimization_stats
        assert optimization_stats['original_size'] == 1500
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_info(self, mock_get_config, mock_config, sample_embeddings):
        """Test getting index information"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        
        # Test with no index
        info = engine.get_index_info()
        assert info["status"] == "no_index"
        
        # Test with index
        engine.build_index(sample_embeddings)
        info = engine.get_index_info()
        
        assert info["status"] == "ready"
        assert info["total_vectors"] == 10
        assert info["dimension"] == 384
        assert "index_type" in info
        assert "memory_usage_bytes" in info
        assert info["chunk_mappings"] == 10
    
    @patch('src.pipeline.core.config.get_config')
    def test_search_configuration(self, mock_get_config, mock_config, sample_embeddings):
        """Test search with custom configuration"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        # Create custom search configuration
        search_config = SearchConfiguration(
            top_k=5,
            similarity_threshold=0.2,
            search_type="cosine",
            include_metadata=True,
            return_scores=True
        )
        
        query_vector = sample_embeddings[0].vector
        results = engine.search_similar_by_vector(
            query_vector, 
            top_k=search_config.top_k,
            similarity_threshold=search_config.similarity_threshold,
            search_config=search_config
        )
        
        assert len(results) <= search_config.top_k
        for result in results:
            assert result.similarity_score >= search_config.similarity_threshold
            assert 'faiss_index' in result.metadata
            assert 'search_time' in result.metadata
    
    @patch('src.pipeline.core.config.get_config')
    def test_search_stats(self, mock_get_config, mock_config, sample_embeddings):
        """Test search statistics tracking"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        # Perform multiple searches
        query_vector = sample_embeddings[0].vector
        for _ in range(5):
            engine.search_similar_by_vector(query_vector, top_k=3)
        
        stats = engine.get_search_stats()
        
        assert stats['total_searches'] == 5
        assert stats['total_search_time'] > 0
        assert stats['average_search_time'] > 0
        assert stats['index_builds'] == 1
        assert stats['total_vectors_indexed'] == 10
    
    @patch('src.pipeline.core.config.get_config')
    def test_saved_index_management(self, mock_get_config, mock_config, sample_embeddings):
        """Test saved index management operations"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        vector_index = engine.build_index(sample_embeddings)
        index_id = vector_index.index_id
        
        # Save index
        engine.save_index(index_id)
        
        # List saved indexes
        saved_indexes = engine.list_saved_indexes()
        assert index_id in saved_indexes
        
        # Delete saved index
        success = engine.delete_saved_index(index_id)
        assert success
        
        # Verify deletion
        saved_indexes = engine.list_saved_indexes()
        assert index_id not in saved_indexes
    
    @patch('src.pipeline.core.config.get_config')
    def test_error_handling(self, mock_get_config, mock_config):
        """Test error handling in various scenarios"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        
        # Test building index with empty embeddings
        with pytest.raises(FAISSError):
            engine.build_index([])
        
        # Test search without index
        with pytest.raises(FAISSError):
            engine.search_similar_by_vector([0.1] * 384)
        
        # Test saving without index
        with pytest.raises(FAISSError):
            engine.save_index("test_id")
        
        # Test loading non-existent index
        with pytest.raises(FAISSError):
            engine.load_index("non_existent_id")
        
        # Test updating without index
        with pytest.raises(FAISSError):
            engine.update_index([])
        
        # Test optimization without index
        with pytest.raises(FAISSError):
            engine.optimize_index()
    
    @patch('src.pipeline.core.config.get_config')
    def test_thread_safety(self, mock_get_config, mock_config, sample_embeddings):
        """Test thread safety of operations"""
        import threading
        import time
        
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        engine.build_index(sample_embeddings)
        
        results = []
        errors = []
        
        def search_worker():
            try:
                query_vector = sample_embeddings[0].vector
                result = engine.search_similar_by_vector(query_vector, top_k=3)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple searches concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        # All results should be similar (same query)
        for result in results:
            assert len(result) > 0
            assert result[0].chunk_id == "chunk_0"
    
    @patch('src.pipeline.core.config.get_config')
    def test_large_scale_performance(self, mock_get_config, mock_config):
        """Test performance with larger dataset"""
        mock_get_config.return_value = mock_config
        
        # Create larger dataset
        embeddings = []
        for i in range(5000):
            vector = np.random.RandomState(i).random(384).astype(np.float32).tolist()
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector,
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        
        engine = FAISSSearchEngine()
        
        # Measure build time
        start_time = time.time()
        vector_index = engine.build_index(embeddings)
        build_time = time.time() - start_time
        
        assert vector_index.total_vectors == 5000
        assert build_time < 30  # Should build within 30 seconds
        
        # Measure search time
        query_vector = embeddings[0].vector
        start_time = time.time()
        results = engine.search_similar_by_vector(query_vector, top_k=10)
        search_time = time.time() - start_time
        
        assert len(results) == 10
        assert search_time < 1.0  # Should search within 1 second
        
        print(f"Large scale test: {5000} vectors, build_time={build_time:.3f}s, search_time={search_time:.4f}s")


class TestFAISSIndexMetadata:
    """Test suite for FAISSIndexMetadata"""
    
    def test_metadata_creation(self):
        """Test FAISSIndexMetadata creation"""
        metadata = FAISSIndexMetadata(
            index_id="test_index",
            index_type="flat",
            dimension=384,
            total_vectors=100,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        assert metadata.index_id == "test_index"
        assert metadata.index_type == "flat"
        assert metadata.dimension == 384
        assert metadata.total_vectors == 100
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.last_updated, datetime)


class TestSearchConfiguration:
    """Test suite for SearchConfiguration"""
    
    def test_search_config_creation(self):
        """Test SearchConfiguration creation"""
        config = SearchConfiguration(
            top_k=10,
            similarity_threshold=0.5,
            search_type="cosine",
            nprobe=20,
            ef_search=128,
            include_metadata=True,
            return_scores=True
        )
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.search_type == "cosine"
        assert config.nprobe == 20
        assert config.ef_search == 128
        assert config.include_metadata is True
        assert config.return_scores is True
    
    def test_search_config_defaults(self):
        """Test SearchConfiguration default values"""
        config = SearchConfiguration()
        
        assert config.top_k == 5
        assert config.similarity_threshold == 0.1
        assert config.search_type == "cosine"
        assert config.nprobe == 10
        assert config.ef_search == 64
        assert config.include_metadata is True
        assert config.return_scores is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])