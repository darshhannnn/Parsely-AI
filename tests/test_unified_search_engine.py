"""
Tests for Unified Search Engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np

from src.pipeline.stages.stage3_embedding_search.unified_search_engine import (
    UnifiedSearchEngine, VectorDBType, UnifiedSearchConfiguration
)
from src.pipeline.core.models import ContentChunk, Embedding, VectorIndex, SearchResult
from src.pipeline.core.exceptions import VectorIndexError, SearchError


class TestUnifiedSearchEngine:
    """Test suite for UnifiedSearchEngine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock()
        config.database.vector_db_type = "auto"
        config.database.pinecone_api_key = "test-api-key"
        config.database.pinecone_environment = "test-env"
        config.database.pinecone_index_name = "test-index"
        config.database.faiss_index_path = "./test_indexes"
        config.enable_caching = True
        return config
    
    @pytest.fixture
    def mock_faiss_engine(self):
        """Mock FAISS search engine"""
        mock_engine = Mock()
        mock_engine.get_index_info.return_value = {
            "status": "ready",
            "total_vectors": 100,
            "dimension": 384,
            "index_type": "IndexFlatIP"
        }
        mock_engine.get_search_stats.return_value = {
            "total_searches": 10,
            "total_search_time": 1.0
        }
        return mock_engine
    
    @pytest.fixture
    def mock_pinecone_engine(self):
        """Mock Pinecone search engine"""
        mock_engine = Mock()
        mock_engine.get_index_info.return_value = {
            "status": "ready",
            "total_vectors": 200,
            "dimension": 384,
            "current_namespace": "default"
        }
        mock_engine.get_search_stats.return_value = {
            "total_searches": 5,
            "total_search_time": 0.5
        }
        return mock_engine
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing"""
        embeddings = []
        for i in range(3):
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=np.random.random(384).tolist(),
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        return embeddings
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results"""
        results = []
        for i in range(2):
            result = SearchResult(
                chunk_id=f"chunk_{i}",
                content=f"Test content {i}",
                similarity_score=0.9 - i * 0.1,
                rank=i + 1,
                metadata={"source": "test"}
            )
            results.append(result)
        return results
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_initialization_auto_with_both_engines(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                                  mock_config, mock_faiss_engine, mock_pinecone_engine):
        """Test initialization with auto mode and both engines available"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        engine = UnifiedSearchEngine()
        
        assert engine.faiss_engine is not None
        assert engine.pinecone_engine is not None
        assert engine.active_engine == mock_pinecone_engine  # Should prefer Pinecone in auto mode
        assert engine.active_db_type == VectorDBType.PINECONE
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_initialization_faiss_only(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                      mock_config, mock_faiss_engine):
        """Test initialization with FAISS only"""
        mock_config.database.vector_db_type = "faiss"
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        engine = UnifiedSearchEngine()
        
        assert engine.faiss_engine is not None
        assert engine.pinecone_engine is None
        assert engine.active_engine == mock_faiss_engine
        assert engine.active_db_type == VectorDBType.FAISS
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_initialization_pinecone_only(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                         mock_config, mock_pinecone_engine):
        """Test initialization with Pinecone only"""
        mock_config.database.vector_db_type = "pinecone"
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = Mock()
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        engine = UnifiedSearchEngine()
        
        assert engine.faiss_engine is not None  # Always initialized as fallback
        assert engine.pinecone_engine is not None
        assert engine.active_engine == mock_pinecone_engine
        assert engine.active_db_type == VectorDBType.PINECONE
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_build_index(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                        mock_config, mock_faiss_engine, sample_embeddings):
        """Test building index with active engine"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        # Mock build_index return
        mock_vector_index = VectorIndex(
            index_id="test-index",
            index_type="faiss",
            dimension=384,
            total_vectors=3
        )
        mock_faiss_engine.build_index.return_value = mock_vector_index
        
        engine = UnifiedSearchEngine()
        result = engine.build_index(sample_embeddings)
        
        assert result == mock_vector_index
        mock_faiss_engine.build_index.assert_called_once_with(sample_embeddings)
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_build_index_with_fallback(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                      mock_config, mock_faiss_engine, mock_pinecone_engine, sample_embeddings):
        """Test building index with fallback on failure"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        # Mock Pinecone failure and FAISS success
        mock_pinecone_engine.build_index.side_effect = Exception("Pinecone error")
        mock_vector_index = VectorIndex(
            index_id="test-index",
            index_type="faiss",
            dimension=384,
            total_vectors=3
        )
        mock_faiss_engine.build_index.return_value = mock_vector_index
        
        engine = UnifiedSearchEngine()
        result = engine.build_index(sample_embeddings)
        
        assert result == mock_vector_index
        assert engine.active_db_type == VectorDBType.FAISS  # Should have fallen back
        assert engine.stats['fallback_events'] == 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_search_similar_by_vector_faiss(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                           mock_config, mock_faiss_engine, sample_search_results):
        """Test vector search with FAISS engine"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        mock_faiss_engine.search_similar_by_vector.return_value = sample_search_results
        
        engine = UnifiedSearchEngine()
        query_vector = np.random.random(384).tolist()
        
        results = engine.search_similar_by_vector(query_vector)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_0"
        assert results[0].similarity_score == 0.9
        
        mock_faiss_engine.search_similar_by_vector.assert_called_once()
        assert engine.stats['total_searches'] == 1
        assert engine.stats['faiss_searches'] == 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_search_similar_by_vector_pinecone(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                              mock_config, mock_faiss_engine, mock_pinecone_engine, sample_search_results):
        """Test vector search with Pinecone engine"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        mock_pinecone_engine.search_similar_by_vector.return_value = sample_search_results
        
        engine = UnifiedSearchEngine()
        query_vector = np.random.random(384).tolist()
        
        results = engine.search_similar_by_vector(query_vector)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_0"
        
        mock_pinecone_engine.search_similar_by_vector.assert_called_once()
        assert engine.stats['total_searches'] == 1
        assert engine.stats['pinecone_searches'] == 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_search_with_fallback(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                 mock_config, mock_faiss_engine, mock_pinecone_engine, sample_search_results):
        """Test search with fallback on failure"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        # Mock Pinecone failure and FAISS success
        mock_pinecone_engine.search_similar_by_vector.side_effect = Exception("Pinecone error")
        mock_faiss_engine.search_similar_by_vector.return_value = sample_search_results
        
        engine = UnifiedSearchEngine()
        query_vector = np.random.random(384).tolist()
        
        search_config = UnifiedSearchConfiguration(enable_fallback=True)
        results = engine.search_similar_by_vector(query_vector, search_config)
        
        assert len(results) == 2
        assert engine.active_db_type == VectorDBType.FAISS  # Should have fallen back
        assert engine.stats['fallback_events'] == 1
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_update_index(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                         mock_config, mock_faiss_engine, sample_embeddings):
        """Test updating index with new embeddings"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        engine = UnifiedSearchEngine()
        engine.update_index(sample_embeddings)
        
        mock_faiss_engine.update_index.assert_called_once_with(sample_embeddings)
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_remove_vectors(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                           mock_config, mock_faiss_engine):
        """Test removing vectors from index"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        mock_faiss_engine.remove_vectors.return_value = 3
        
        engine = UnifiedSearchEngine()
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        removed_count = engine.remove_vectors(chunk_ids)
        
        assert removed_count == 3
        mock_faiss_engine.remove_vectors.assert_called_once_with(chunk_ids)
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_switch_engine(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                          mock_config, mock_faiss_engine, mock_pinecone_engine):
        """Test manually switching between engines"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        engine = UnifiedSearchEngine()
        
        # Should start with Pinecone (auto mode prefers Pinecone)
        assert engine.active_db_type == VectorDBType.PINECONE
        
        # Switch to FAISS
        result = engine.switch_engine(VectorDBType.FAISS)
        assert result is True
        assert engine.active_db_type == VectorDBType.FAISS
        assert engine.active_engine == mock_faiss_engine
        
        # Switch back to Pinecone
        result = engine.switch_engine(VectorDBType.PINECONE)
        assert result is True
        assert engine.active_db_type == VectorDBType.PINECONE
        assert engine.active_engine == mock_pinecone_engine
        
        assert engine.stats['engine_switches'] == 2
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_switch_engine_unavailable(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                      mock_config, mock_faiss_engine):
        """Test switching to unavailable engine"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        engine = UnifiedSearchEngine()
        
        # Try to switch to unavailable Pinecone
        result = engine.switch_engine(VectorDBType.PINECONE)
        assert result is False
        assert engine.active_db_type == VectorDBType.FAISS  # Should remain FAISS
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_get_engine_status(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                              mock_config, mock_faiss_engine, mock_pinecone_engine):
        """Test getting engine status"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        engine = UnifiedSearchEngine()
        status = engine.get_engine_status()
        
        assert status["active_engine"] == "pinecone"
        assert status["engines"]["faiss"]["available"] is True
        assert status["engines"]["faiss"]["status"] == "ready"
        assert status["engines"]["faiss"]["total_vectors"] == 100
        assert status["engines"]["pinecone"]["available"] is True
        assert status["engines"]["pinecone"]["status"] == "ready"
        assert status["engines"]["pinecone"]["total_vectors"] == 200
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_get_unified_stats(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                              mock_config, mock_faiss_engine, mock_pinecone_engine):
        """Test getting unified statistics"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.return_value = mock_pinecone_engine
        
        engine = UnifiedSearchEngine()
        
        # Simulate some searches
        engine.stats['total_searches'] = 10
        engine.stats['total_search_time'] = 2.0
        engine.stats['faiss_searches'] = 6
        engine.stats['faiss_search_time'] = 1.2
        engine.stats['pinecone_searches'] = 4
        engine.stats['pinecone_search_time'] = 0.8
        
        stats = engine.get_unified_stats()
        
        assert stats['total_searches'] == 10
        assert stats['average_search_time'] == 0.2
        assert stats['average_faiss_search_time'] == 0.2
        assert stats['average_pinecone_search_time'] == 0.2
        assert 'faiss_engine_stats' in stats
        assert 'pinecone_engine_stats' in stats
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_optimize_active_index(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                  mock_config, mock_faiss_engine):
        """Test optimizing the active index"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        mock_faiss_engine.optimize_index.return_value = {"optimizations_applied": ["nprobe=10"]}
        
        engine = UnifiedSearchEngine()
        result = engine.optimize_active_index()
        
        assert "optimizations_applied" in result
        mock_faiss_engine.optimize_index.assert_called_once()
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_save_and_load_index(self, mock_pinecone_class, mock_faiss_class, mock_get_config, 
                                mock_config, mock_faiss_engine):
        """Test saving and loading index (FAISS only)"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.return_value = mock_faiss_engine
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        mock_faiss_engine.save_index.return_value = True
        mock_faiss_engine.load_index.return_value = True
        
        engine = UnifiedSearchEngine()
        
        # Test saving
        result = engine.save_active_index("test-index")
        assert result is True
        mock_faiss_engine.save_index.assert_called_once_with("test-index")
        
        # Test loading
        result = engine.load_index("test-index")
        assert result is True
        mock_faiss_engine.load_index.assert_called_once_with("test-index")
    
    def test_unified_search_configuration(self):
        """Test UnifiedSearchConfiguration dataclass"""
        config = UnifiedSearchConfiguration(
            top_k=10,
            similarity_threshold=0.3,
            preferred_db=VectorDBType.FAISS,
            enable_fallback=False,
            namespace="test-namespace",
            include_metadata=False
        )
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.3
        assert config.preferred_db == VectorDBType.FAISS
        assert config.enable_fallback is False
        assert config.namespace == "test-namespace"
        assert config.include_metadata is False
    
    def test_vector_db_type_enum(self):
        """Test VectorDBType enum"""
        assert VectorDBType.FAISS.value == "faiss"
        assert VectorDBType.PINECONE.value == "pinecone"
        assert VectorDBType.AUTO.value == "auto"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.FAISSSearchEngine')
    @patch('src.pipeline.stages.stage3_embedding_search.unified_search_engine.PineconeSearchEngine')
    def test_no_engines_available(self, mock_pinecone_class, mock_faiss_class, mock_get_config, mock_config):
        """Test initialization when no engines are available"""
        mock_get_config.return_value = mock_config
        mock_faiss_class.side_effect = Exception("FAISS not available")
        mock_pinecone_class.side_effect = Exception("Pinecone not available")
        
        with pytest.raises(VectorIndexError, match="No search engines available"):
            UnifiedSearchEngine()


if __name__ == "__main__":
    pytest.main([__file__])