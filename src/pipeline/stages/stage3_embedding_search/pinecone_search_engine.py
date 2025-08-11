"""
Pinecone Cloud Vector Database Integration for Document Processing Pipeline

This module implements cloud-based vector similarity search using Pinecone
with namespace management, upsert operations, and comprehensive search capabilities.
"""

import os
import time
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from dataclasses import dataclass, field
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None
    Pinecone = None
    ServerlessSpec = None

import numpy as np

# Local imports
from ...core.models import ContentChunk, Embedding, VectorIndex, SearchResult, ProcessingMetadata
from ...core.interfaces import IEmbeddingSearchEngine
from ...core.config import get_config
from ...core.exceptions import VectorIndexError, PineconeError, SearchError
from ...core.logging_utils import get_logger


@dataclass
class PineconeIndexMetadata:
    """Metadata for Pinecone index"""
    index_name: str
    namespace: str
    dimension: int
    total_vectors: int
    created_at: datetime
    last_updated: datetime
    index_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class PineconeSearchConfiguration:
    """Configuration for Pinecone search operations"""
    top_k: int = 5
    similarity_threshold: float = 0.1
    namespace: str = ""
    include_metadata: bool = True
    include_values: bool = False
    filter_conditions: Optional[Dict[str, Any]] = None


class PineconeSearchEngine:
    """
    Cloud-based Pinecone vector search engine.
    
    Features:
    - Serverless Pinecone integration
    - Namespace management for document organization
    - Vector upsert and query operations
    - Metadata filtering and search
    - Async operations support
    - Comprehensive error handling
    - Fallback logic integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Pinecone search engine"""
        if not PINECONE_AVAILABLE:
            raise PineconeError("Pinecone client not available. Install with: pip install pinecone-client")
        
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Override config if provided
        if config:
            for key, value in config.items():
                if hasattr(self.config.database, key):
                    setattr(self.config.database, key, value)
        
        # Validate required configuration
        if not self.config.database.pinecone_api_key:
            raise PineconeError("Pinecone API key is required")
        
        # Initialize Pinecone client
        self.pc = None
        self.index = None
        self.index_metadata: Optional[PineconeIndexMetadata] = None
        self.current_namespace = "default"
        
        # Thread safety
        self.client_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'average_search_time': 0.0,
            'total_upserts': 0,
            'total_vectors_upserted': 0,
            'total_deletes': 0,
            'connection_errors': 0,
            'api_calls': 0
        }
        
        # Initialize client
        self._initialize_client()
        
        self.logger.info("PineconeSearchEngine initialized")
    
    def _initialize_client(self) -> None:
        """Initialize Pinecone client and connection"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.config.database.pinecone_api_key)
            
            # Test connection by listing indexes
            indexes = self.pc.list_indexes()
            self.logger.info(f"Connected to Pinecone. Available indexes: {len(indexes)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone client: {e}")
            raise PineconeError(f"Client initialization failed: {e}")
    
    def _get_or_create_index(self, dimension: int, index_name: Optional[str] = None) -> str:
        """Get existing index or create new one"""
        index_name = index_name or self.config.database.pinecone_index_name
        
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name in existing_indexes:
                self.logger.info(f"Using existing Pinecone index: {index_name}")
                return index_name
            
            # Create new index
            self.logger.info(f"Creating new Pinecone index: {index_name}")
            
            # Use serverless spec for cost efficiency
            spec = ServerlessSpec(
                cloud=self.config.database.pinecone_environment or "aws",
                region="us-east-1"  # Default region
            )
            
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=spec
            )
            
            # Wait for index to be ready
            self._wait_for_index_ready(index_name)
            
            self.logger.info(f"Created Pinecone index: {index_name}")
            return index_name
            
        except Exception as e:
            self.logger.error(f"Failed to get or create Pinecone index: {e}")
            raise PineconeError(f"Index creation failed: {e}")
    
    def _wait_for_index_ready(self, index_name: str, timeout: int = 60) -> None:
        """Wait for index to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                index_info = self.pc.describe_index(index_name)
                if index_info.status.ready:
                    return
                
                time.sleep(2)
                
            except Exception as e:
                self.logger.warning(f"Error checking index status: {e}")
                time.sleep(2)
        
        raise PineconeError(f"Index {index_name} not ready after {timeout} seconds")
    
    def _connect_to_index(self, index_name: str) -> None:
        """Connect to Pinecone index"""
        try:
            self.index = self.pc.Index(index_name)
            
            # Test connection
            stats = self.index.describe_index_stats()
            self.logger.info(f"Connected to Pinecone index: {index_name}, vectors: {stats.total_vector_count}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pinecone index: {e}")
            raise PineconeError(f"Index connection failed: {e}")
    
    def _prepare_vectors_for_upsert(self, embeddings: List[Embedding]) -> List[Dict[str, Any]]:
        """Prepare vectors for Pinecone upsert"""
        vectors = []
        
        for embedding in embeddings:
            vector_data = {
                "id": embedding.chunk_id,
                "values": embedding.vector,
                "metadata": {
                    "model_name": embedding.model_name,
                    "dimension": embedding.dimension,
                    "created_at": embedding.created_at.isoformat(),
                    "chunk_id": embedding.chunk_id
                }
            }
            
            # Add chunk metadata if available
            if hasattr(embedding, 'chunk_metadata') and embedding.chunk_metadata:
                chunk_meta = embedding.chunk_metadata
                if hasattr(chunk_meta, '__dict__'):
                    # Convert chunk metadata to serializable format
                    chunk_dict = {}
                    for key, value in chunk_meta.__dict__.items():
                        if isinstance(value, (str, int, float, bool)):
                            chunk_dict[key] = value
                        elif isinstance(value, datetime):
                            chunk_dict[key] = value.isoformat()
                        else:
                            chunk_dict[key] = str(value)
                    
                    vector_data["metadata"].update(chunk_dict)
            
            vectors.append(vector_data)
        
        return vectors
    
    def build_index(self, embeddings: List[Embedding]) -> VectorIndex:
        """Build Pinecone index from embeddings"""
        if not embeddings:
            raise PineconeError("Cannot build index from empty embeddings list")
        
        try:
            with self.client_lock:
                # Get dimension from first embedding
                dimension = embeddings[0].dimension
                
                # Get or create index
                index_name = self._get_or_create_index(dimension)
                self._connect_to_index(index_name)
                
                # Prepare vectors for upsert
                vectors = self._prepare_vectors_for_upsert(embeddings)
                
                # Upsert vectors in batches
                batch_size = 100  # Pinecone recommended batch size
                start_time = time.time()
                
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    
                    self.index.upsert(
                        vectors=batch,
                        namespace=self.current_namespace
                    )
                    
                    self.logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                
                build_time = time.time() - start_time
                
                # Create metadata
                self.index_metadata = PineconeIndexMetadata(
                    index_name=index_name,
                    namespace=self.current_namespace,
                    dimension=dimension,
                    total_vectors=len(embeddings),
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    index_parameters={
                        'index_name': index_name,
                        'namespace': self.current_namespace,
                        'dimension': dimension,
                        'metric': 'cosine'
                    },
                    performance_stats={
                        'build_time': build_time,
                        'vectors_per_second': len(embeddings) / build_time if build_time > 0 else 0,
                        'batch_size': batch_size
                    }
                )
                
                # Update stats
                self.stats['total_upserts'] += 1
                self.stats['total_vectors_upserted'] += len(embeddings)
                self.stats['api_calls'] += (len(vectors) - 1) // batch_size + 1
                
                self.logger.info(
                    f"Built Pinecone index: {len(embeddings)} vectors, {dimension}D, "
                    f"namespace={self.current_namespace}, time={build_time:.3f}s"
                )
                
                return VectorIndex(
                    index_id=f"{index_name}:{self.current_namespace}",
                    index_type="pinecone",
                    dimension=dimension,
                    total_vectors=len(embeddings),
                    metadata={
                        'index_name': index_name,
                        'namespace': self.current_namespace,
                        'build_time': build_time,
                        'performance_stats': self.index_metadata.performance_stats
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to build Pinecone index: {e}")
            self.stats['connection_errors'] += 1
            raise PineconeError(f"Index building failed: {e}")
    
    def search_similar_by_vector(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        search_config: Optional[PineconeSearchConfiguration] = None
    ) -> List[SearchResult]:
        """Search for similar vectors using query vector"""
        if not self.index:
            raise PineconeError("No index available for search")
        
        try:
            start_time = time.time()
            
            # Prepare search configuration
            namespace = search_config.namespace if search_config else self.current_namespace
            include_metadata = search_config.include_metadata if search_config else True
            include_values = search_config.include_values if search_config else False
            filter_conditions = search_config.filter_conditions if search_config else None
            
            # Perform search
            search_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                include_values=include_values,
                filter=filter_conditions
            )
            
            search_time = time.time() - start_time
            
            # Process results
            results = []
            for i, match in enumerate(search_response.matches):
                similarity_score = float(match.score)
                
                # Apply similarity threshold
                if similarity_score < similarity_threshold:
                    continue
                
                # Extract metadata
                metadata = match.metadata or {}
                chunk_id = match.id
                content = metadata.get('content', '')
                
                # Create search result
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    similarity_score=similarity_score,
                    rank=i + 1,
                    metadata={
                        'pinecone_id': match.id,
                        'namespace': namespace,
                        'search_time': search_time,
                        'pinecone_metadata': metadata
                    }
                )
                results.append(result)
            
            # Update stats
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += search_time
            self.stats['average_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
            self.stats['api_calls'] += 1
            
            self.logger.info(
                f"Pinecone search completed: {len(results)} results, "
                f"time={search_time:.4f}s, namespace={namespace}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pinecone search failed: {e}")
            self.stats['connection_errors'] += 1
            raise SearchError(f"Vector search failed: {e}")
    
    def update_index(self, new_embeddings: List[Embedding]) -> None:
        """Update existing index with new embeddings"""
        if not self.index:
            raise PineconeError("No existing index to update")
        
        try:
            # Prepare vectors for upsert
            vectors = self._prepare_vectors_for_upsert(new_embeddings)
            
            # Upsert vectors in batches
            batch_size = 100
            start_time = time.time()
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                self.index.upsert(
                    vectors=batch,
                    namespace=self.current_namespace
                )
            
            update_time = time.time() - start_time
            
            # Update metadata
            if self.index_metadata:
                self.index_metadata.total_vectors += len(new_embeddings)
                self.index_metadata.last_updated = datetime.now()
                self.index_metadata.performance_stats['last_update_time'] = update_time
            
            # Update stats
            self.stats['total_upserts'] += 1
            self.stats['total_vectors_upserted'] += len(new_embeddings)
            self.stats['api_calls'] += (len(vectors) - 1) // batch_size + 1
            
            self.logger.info(f"Updated Pinecone index with {len(new_embeddings)} new vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to update Pinecone index: {e}")
            self.stats['connection_errors'] += 1
            raise PineconeError(f"Index update failed: {e}")
    
    def remove_vectors(self, chunk_ids: List[str], namespace: Optional[str] = None) -> int:
        """Remove vectors from index by chunk IDs"""
        if not self.index or not chunk_ids:
            return 0
        
        try:
            namespace = namespace or self.current_namespace
            
            # Delete vectors in batches
            batch_size = 1000  # Pinecone delete batch limit
            deleted_count = 0
            
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                
                self.index.delete(
                    ids=batch,
                    namespace=namespace
                )
                
                deleted_count += len(batch)
            
            # Update stats
            self.stats['total_deletes'] += 1
            self.stats['api_calls'] += (len(chunk_ids) - 1) // batch_size + 1
            
            # Update metadata
            if self.index_metadata:
                self.index_metadata.total_vectors = max(0, self.index_metadata.total_vectors - deleted_count)
                self.index_metadata.last_updated = datetime.now()
            
            self.logger.info(f"Removed {deleted_count} vectors from Pinecone index")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove vectors from Pinecone index: {e}")
            self.stats['connection_errors'] += 1
            raise PineconeError(f"Vector removal failed: {e}")
    
    def create_namespace(self, namespace: str) -> bool:
        """Create a new namespace (Pinecone creates namespaces automatically on first use)"""
        try:
            # Pinecone creates namespaces automatically, so we just validate the name
            if not namespace or not isinstance(namespace, str):
                raise PineconeError("Invalid namespace name")
            
            # Test namespace by querying (will create if doesn't exist)
            if self.index:
                self.index.describe_index_stats(filter={})
            
            self.logger.info(f"Namespace ready: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create namespace {namespace}: {e}")
            return False
    
    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace"""
        if not self.index:
            return False
        
        try:
            # Delete all vectors in namespace
            self.index.delete(delete_all=True, namespace=namespace)
            
            self.logger.info(f"Deleted namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete namespace {namespace}: {e}")
            return False
    
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        if not self.index:
            return []
        
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            return namespaces
            
        except Exception as e:
            self.logger.error(f"Failed to list namespaces: {e}")
            return []
    
    def set_namespace(self, namespace: str) -> None:
        """Set the current working namespace"""
        self.current_namespace = namespace
        self.logger.info(f"Set current namespace to: {namespace}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get comprehensive index information"""
        if not self.index:
            return {"status": "no_index"}
        
        try:
            stats = self.index.describe_index_stats()
            
            info = {
                "status": "ready",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_type": "pinecone",
                "current_namespace": self.current_namespace,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else [],
                "namespace_stats": {}
            }
            
            # Add namespace-specific stats
            if stats.namespaces:
                for ns_name, ns_stats in stats.namespaces.items():
                    info["namespace_stats"][ns_name] = {
                        "vector_count": ns_stats.vector_count
                    }
            
            if self.index_metadata:
                info.update({
                    "index_name": self.index_metadata.index_name,
                    "created_at": self.index_metadata.created_at.isoformat(),
                    "last_updated": self.index_metadata.last_updated.isoformat(),
                    "index_parameters": self.index_metadata.index_parameters,
                    "performance_stats": self.index_metadata.performance_stats
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get index info: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return self.stats.copy()
    
    def clear_index(self, namespace: Optional[str] = None) -> None:
        """Clear all vectors in the current or specified namespace"""
        if not self.index:
            return
        
        namespace = namespace or self.current_namespace
        
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            
            if self.index_metadata and namespace == self.current_namespace:
                self.index_metadata.total_vectors = 0
                self.index_metadata.last_updated = datetime.now()
            
            self.logger.info(f"Cleared Pinecone index namespace: {namespace}")
            
        except Exception as e:
            self.logger.error(f"Failed to clear index: {e}")
            raise PineconeError(f"Index clearing failed: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Pinecone connection and return status"""
        try:
            # Test basic connection
            indexes = self.pc.list_indexes()
            
            result = {
                "connected": True,
                "available_indexes": len(indexes),
                "current_index": self.index_metadata.index_name if self.index_metadata else None,
                "current_namespace": self.current_namespace
            }
            
            # Test index connection if available
            if self.index:
                stats = self.index.describe_index_stats()
                result["index_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension
                }
            
            return result
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    async def search_similar_async(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        search_config: Optional[PineconeSearchConfiguration] = None
    ) -> List[SearchResult]:
        """Async version of vector search"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(
                executor, 
                self.search_similar_by_vector, 
                query_vector, 
                top_k, 
                similarity_threshold, 
                search_config
            )
    
    async def update_index_async(self, new_embeddings: List[Embedding]) -> None:
        """Async version of index update"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, self.update_index, new_embeddings)
    
    # Implement IEmbeddingSearchEngine interface methods
    def search_similar(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar vectors using query string"""
        # This method would typically use an embedding generator to convert query to vector
        raise NotImplementedError("search_similar with query string requires embedding generator integration")
    
    def create_embeddings(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """This method should be handled by EmbeddingGenerator"""
        raise NotImplementedError("create_embeddings should be handled by EmbeddingGenerator")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'logger') and hasattr(self, 'stats'):
                self.logger.info(f"PineconeSearchEngine final stats: {self.get_search_stats()}")
        except Exception:
            pass