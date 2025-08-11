"""
Tests for Pinecone Integration
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np

from src.pipeline.stages.stage3_embedding_search.pinecone_search_engine import (
    PineconeSearchEngine, PineconeIndexMetadata, PineconeSearchConfiguration
)
from src.pipeline.core.models import ContentChunk, Embedding, VectorIndex
from src.pipeline.core.exceptions import PineconeError


class TestPineconeSearchEngine:
    """Test suite for PineconeSearchEngine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock()
        config.database.pinecone_api_key = "test-api-key"
        config.database.pinecone_environment = "test-env"
        config.database.pinecone_index_name = "test-index"
        return config
    
    @pytest.fixture
    def mock_pinecone_client(self):
        """Mock Pinecone client"""
        mock_pc = Mock()
        mock_index = Mock()
        
        # Mock list_indexes
        mock_pc.list_indexes.return_value = []
        
        # Mock create_index
        mock_pc.create_index.return_value = None
        
        # Mock describe_index
        mock_index_info = Mock()
        mock_index_info.status.ready = True
        mock_pc.describe_index.return_value = mock_index_info
        
        # Mock Index
        mock_pc.Index.return_value = mock_index
        
        # Mock index operations
        mock_index.upsert.return_value = None
        mock_index.query.return_value = Mock(matches=[])
        mock_index.delete.return_value = None
        
        # Mock index stats
        mock_stats = Mock()
        mock_stats.total_vector_count = 0
        mock_stats.dimension = 384
        mock_stats.namespaces = {}
        mock_index.describe_index_stats.return_value = mock_stats
        
        return mock_pc, mock_index
    
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
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_initialization_success(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test successful PineconeSearchEngine initialization"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        engine = PineconeSearchEngine()
        
        assert engine.pc is not None
        assert engine.current_namespace == "default"
        assert engine.stats['total_searches'] == 0
        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
    
    @patch('src.pipeline.core.config.get_config')
    def test_initialization_no_api_key(self, mock_get_config):
        """Test initialization failure without API key"""
        config = Mock()
        config.database.pinecone_api_key = ""
        mock_get_config.return_value = config
        
        with pytest.raises(PineconeError, match="Pinecone API key is required"):
            PineconeSearchEngine()
    
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.PINECONE_AVAILABLE', False)
    def test_initialization_pinecone_not_available(self):
        """Test initialization when Pinecone is not installed"""
        with pytest.raises(PineconeError, match="Pinecone client not available"):
            PineconeSearchEngine()
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_build_index_new(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client, sample_embeddings):
        """Test building a new Pinecone index"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        result = engine.build_index(sample_embeddings)
        
        assert isinstance(result, VectorIndex)
        assert result.index_type == "pinecone"
        assert result.dimension == 384
        assert result.total_vectors == 3
        
        # Verify upsert was called
        mock_index.upsert.assert_called()
        
        # Check metadata
        assert engine.index_metadata is not None
        assert engine.index_metadata.total_vectors == 3
        assert engine.index_metadata.dimension == 384
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_build_index_existing(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client, sample_embeddings):
        """Test building index when index already exists"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock existing index
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        result = engine.build_index(sample_embeddings)
        
        assert isinstance(result, VectorIndex)
        # Should not call create_index since it exists
        mock_pc.create_index.assert_not_called()
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_search_similar_by_vector(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test vector similarity search"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock search results
        mock_match = Mock()
        mock_match.id = "chunk_1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "test content", "chunk_id": "chunk_1"}
        
        mock_search_response = Mock()
        mock_search_response.matches = [mock_match]
        mock_index.query.return_value = mock_search_response
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        query_vector = np.random.random(384).tolist()
        results = engine.search_similar_by_vector(query_vector, top_k=5)
        
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"
        assert results[0].similarity_score == 0.95
        assert results[0].content == "test content"
        assert results[0].rank == 1
        
        # Verify query was called with correct parameters
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args[1]['vector'] == query_vector
        assert call_args[1]['top_k'] == 5
        assert call_args[1]['namespace'] == "default"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_search_with_threshold(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test search with similarity threshold filtering"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock search results with different scores
        mock_match1 = Mock()
        mock_match1.id = "chunk_1"
        mock_match1.score = 0.95  # Above threshold
        mock_match1.metadata = {"content": "high similarity"}
        
        mock_match2 = Mock()
        mock_match2.id = "chunk_2"
        mock_match2.score = 0.05  # Below threshold
        mock_match2.metadata = {"content": "low similarity"}
        
        mock_search_response = Mock()
        mock_search_response.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_search_response
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        query_vector = np.random.random(384).tolist()
        results = engine.search_similar_by_vector(
            query_vector, 
            top_k=5, 
            similarity_threshold=0.1
        )
        
        # Should only return results above threshold
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"
        assert results[0].similarity_score == 0.95
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_update_index(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client, sample_embeddings):
        """Test updating index with new embeddings"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        engine.index_metadata = PineconeIndexMetadata(
            index_name="test-index",
            namespace="default",
            dimension=384,
            total_vectors=5,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        new_embeddings = sample_embeddings[:2]  # Add 2 new embeddings
        engine.update_index(new_embeddings)
        
        # Verify upsert was called
        mock_index.upsert.assert_called()
        
        # Check metadata was updated
        assert engine.index_metadata.total_vectors == 7  # 5 + 2
        assert engine.stats['total_upserts'] == 1
        assert engine.stats['total_vectors_upserted'] == 2
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_remove_vectors(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test removing vectors from index"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        engine.index_metadata = PineconeIndexMetadata(
            index_name="test-index",
            namespace="default",
            dimension=384,
            total_vectors=10,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
        removed_count = engine.remove_vectors(chunk_ids)
        
        assert removed_count == 3
        
        # Verify delete was called
        mock_index.delete.assert_called_once()
        call_args = mock_index.delete.call_args
        assert call_args[1]['ids'] == chunk_ids
        assert call_args[1]['namespace'] == "default"
        
        # Check metadata was updated
        assert engine.index_metadata.total_vectors == 7  # 10 - 3
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_namespace_operations(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test namespace management operations"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        # Test setting namespace
        engine.set_namespace("test-namespace")
        assert engine.current_namespace == "test-namespace"
        
        # Test creating namespace (should succeed)
        result = engine.create_namespace("new-namespace")
        assert result is True
        
        # Test deleting namespace
        result = engine.delete_namespace("old-namespace")
        assert result is True
        mock_index.delete.assert_called_with(delete_all=True, namespace="old-namespace")
        
        # Test listing namespaces
        mock_stats = Mock()
        mock_stats.namespaces = {"ns1": Mock(), "ns2": Mock()}
        mock_index.describe_index_stats.return_value = mock_stats
        
        namespaces = engine.list_namespaces()
        assert "ns1" in namespaces
        assert "ns2" in namespaces
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_get_index_info(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test getting index information"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock index stats
        mock_stats = Mock()
        mock_stats.total_vector_count = 100
        mock_stats.dimension = 384
        mock_stats.namespaces = {
            "default": Mock(vector_count=50),
            "test": Mock(vector_count=50)
        }
        mock_index.describe_index_stats.return_value = mock_stats
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        engine.index_metadata = PineconeIndexMetadata(
            index_name="test-index",
            namespace="default",
            dimension=384,
            total_vectors=100,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        info = engine.get_index_info()
        
        assert info["status"] == "ready"
        assert info["total_vectors"] == 100
        assert info["dimension"] == 384
        assert info["index_type"] == "pinecone"
        assert info["current_namespace"] == "default"
        assert "default" in info["namespaces"]
        assert "test" in info["namespaces"]
        assert info["namespace_stats"]["default"]["vector_count"] == 50
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_test_connection(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test connection testing functionality"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock successful connection
        mock_pc.list_indexes.return_value = [Mock(name="index1"), Mock(name="index2")]
        
        mock_stats = Mock()
        mock_stats.total_vector_count = 100
        mock_stats.dimension = 384
        mock_index.describe_index_stats.return_value = mock_stats
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        engine.index_metadata = PineconeIndexMetadata(
            index_name="test-index",
            namespace="default",
            dimension=384,
            total_vectors=100,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        result = engine.test_connection()
        
        assert result["connected"] is True
        assert result["available_indexes"] == 2
        assert result["current_index"] == "test-index"
        assert result["current_namespace"] == "default"
        assert result["index_stats"]["total_vectors"] == 100
        assert result["index_stats"]["dimension"] == 384
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_error_handling(self, mock_pinecone_class, mock_get_config, mock_config):
        """Test error handling in various operations"""
        mock_get_config.return_value = mock_config
        
        # Mock Pinecone client that raises exceptions
        mock_pc = Mock()
        mock_pc.list_indexes.side_effect = Exception("Connection error")
        mock_pinecone_class.return_value = mock_pc
        
        with pytest.raises(PineconeError, match="Client initialization failed"):
            PineconeSearchEngine()
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.pinecone_search_engine.Pinecone')
    def test_search_configuration(self, mock_pinecone_class, mock_get_config, mock_config, mock_pinecone_client):
        """Test search with custom configuration"""
        mock_get_config.return_value = mock_config
        mock_pc, mock_index = mock_pinecone_client
        mock_pinecone_class.return_value = mock_pc
        
        # Mock search results
        mock_match = Mock()
        mock_match.id = "chunk_1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "test content"}
        
        mock_search_response = Mock()
        mock_search_response.matches = [mock_match]
        mock_index.query.return_value = mock_search_response
        
        engine = PineconeSearchEngine()
        engine.index = mock_index
        
        # Test with custom search configuration
        search_config = PineconeSearchConfiguration(
            top_k=10,
            similarity_threshold=0.2,
            namespace="custom-namespace",
            include_metadata=True,
            include_values=False,
            filter_conditions={"category": "test"}
        )
        
        query_vector = np.random.random(384).tolist()
        results = engine.search_similar_by_vector(
            query_vector, 
            top_k=10, 
            similarity_threshold=0.2,
            search_config=search_config
        )
        
        # Verify query was called with custom parameters
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args[1]['top_k'] == 10
        assert call_args[1]['namespace'] == "custom-namespace"
        assert call_args[1]['include_metadata'] is True
        assert call_args[1]['include_values'] is False
        assert call_args[1]['filter'] == {"category": "test"}
    
    def test_search_configuration_dataclass(self):
        """Test PineconeSearchConfiguration dataclass"""
        config = PineconeSearchConfiguration(
            top_k=10,
            similarity_threshold=0.3,
            namespace="test-ns",
            include_metadata=False,
            filter_conditions={"type": "document"}
        )
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.3
        assert config.namespace == "test-ns"
        assert config.include_metadata is False
        assert config.filter_conditions == {"type": "document"}
    
    def test_index_metadata_dataclass(self):
        """Test PineconeIndexMetadata dataclass"""
        now = datetime.now()
        metadata = PineconeIndexMetadata(
            index_name="test-index",
            namespace="test-ns",
            dimension=512,
            total_vectors=1000,
            created_at=now,
            last_updated=now,
            index_parameters={"metric": "cosine"},
            performance_stats={"build_time": 5.0}
        )
        
        assert metadata.index_name == "test-index"
        assert metadata.namespace == "test-ns"
        assert metadata.dimension == 512
        assert metadata.total_vectors == 1000
        assert metadata.created_at == now
        assert metadata.index_parameters["metric"] == "cosine"
        assert metadata.performance_stats["build_time"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__])