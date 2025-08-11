"""
Unified Search Engine with FAISS and Pinecone Support

This module provides a unified interface for vector search that can use either
FAISS (local) or Pinecone (cloud) with automatic fallback logic.
"""

import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Local imports
from ...core.models import ContentChunk, Embedding, VectorIndex, SearchResult
from ...core.interfaces import IEmbeddingSearchEngine
from ...core.config import get_config
from ...core.exceptions import VectorIndexError, SearchError
from ...core.logging_utils import get_logger

from .faiss_search_engine import FAISSSearchEngine, SearchConfiguration as FAISSSearchConfig
from .pinecone_search_engine import PineconeSearchEngine, PineconeSearchConfiguration


class VectorDBType(Enum):
    """Supported vector database types"""
    FAISS = "faiss"
    PINECONE = "pinecone"
    AUTO = "auto"


@dataclass
class UnifiedSearchConfiguration:
    """Configuration for unified search operations"""
    top_k: int = 5
    similarity_threshold: float = 0.1
    preferred_db: VectorDBType = VectorDBType.AUTO
    enable_fallback: bool = True
    namespace: str = "default"
    include_metadata: bool = True


class UnifiedSearchEngine(IEmbeddingSearchEngine):
    """
    Unified search engine that supports both FAISS and Pinecone with fallback logic.
    
    Features:
    - Automatic database selection based on configuration
    - Fallback from Pinecone to FAISS on connection issues
    - Unified interface for both vector databases
    - Performance monitoring and comparison
    - Namespace management for Pinecone
    - Index persistence for FAISS
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified search engine"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Override config if provided
        if config:
            for key, value in config.items():
                if hasattr(self.config.database, key):
                    setattr(self.config.database, key, value)
        
        # Initialize search engines
        self.faiss_engine: Optional[FAISSSearchEngine] = None
        self.pinecone_engine: Optional[PineconeSearchEngine] = None
        self.active_engine: Optional[Union[FAISSSearchEngine, PineconeSearchEngine]] = None
        self.active_db_type: Optional[VectorDBType] = None
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'faiss_searches': 0,
            'pinecone_searches': 0,
            'fallback_events': 0,
            'total_search_time': 0.0,
            'faiss_search_time': 0.0,
            'pinecone_search_time': 0.0,
            'engine_switches': 0
        }
        
        # Initialize engines based on configuration
        self._initialize_engines()
        
        self.logger.info(f"UnifiedSearchEngine initialized with active engine: {self.active_db_type}")
    
    def _initialize_engines(self) -> None:
        """Initialize available search engines"""
        db_type = self.config.database.vector_db_type.lower()
        
        try:
            # Always initialize FAISS as fallback
            self.faiss_engine = FAISSSearchEngine()
            self.logger.info("FAISS engine initialized successfully")
            
            # Initialize Pinecone if configured
            if db_type in ["pinecone", "auto"] and self.config.database.pinecone_api_key:
                try:
                    self.pinecone_engine = PineconeSearchEngine()
                    self.logger.info("Pinecone engine initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Pinecone engine: {e}")
                    if db_type == "pinecone":
                        # If Pinecone is explicitly requested but fails, raise error
                        raise VectorIndexError(f"Failed to initialize required Pinecone engine: {e}")
            
            # Set active engine based on configuration
            self._set_active_engine(db_type)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize search engines: {e}")
            # Fallback to FAISS if available
            if self.faiss_engine:
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
                self.logger.info("Falling back to FAISS engine")
            else:
                raise VectorIndexError(f"No search engines available: {e}")
    
    def _set_active_engine(self, db_type: str) -> None:
        """Set the active search engine based on configuration"""
        if db_type == "pinecone" and self.pinecone_engine:
            self.active_engine = self.pinecone_engine
            self.active_db_type = VectorDBType.PINECONE
        elif db_type == "faiss" and self.faiss_engine:
            self.active_engine = self.faiss_engine
            self.active_db_type = VectorDBType.FAISS
        elif db_type == "auto":
            # Prefer Pinecone if available, fallback to FAISS
            if self.pinecone_engine:
                self.active_engine = self.pinecone_engine
                self.active_db_type = VectorDBType.PINECONE
            elif self.faiss_engine:
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
        else:
            # Default to FAISS
            if self.faiss_engine:
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
        
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
    
    def _attempt_fallback(self) -> bool:
        """Attempt to fallback to alternative engine"""
        if not self.config.enable_caching:  # Using enable_caching as fallback flag
            return False
        
        try:
            if self.active_db_type == VectorDBType.PINECONE and self.faiss_engine:
                self.logger.warning("Falling back from Pinecone to FAISS")
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
                self.stats['fallback_events'] += 1
                self.stats['engine_switches'] += 1
                return True
            elif self.active_db_type == VectorDBType.FAISS and self.pinecone_engine:
                self.logger.warning("Falling back from FAISS to Pinecone")
                self.active_engine = self.pinecone_engine
                self.active_db_type = VectorDBType.PINECONE
                self.stats['fallback_events'] += 1
                self.stats['engine_switches'] += 1
                return True
        except Exception as e:
            self.logger.error(f"Fallback attempt failed: {e}")
        
        return False
    
    def create_embeddings(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """Create embeddings for content chunks (delegates to embedding generator)"""
        # This should be handled by the EmbeddingGenerator
        raise NotImplementedError("create_embeddings should be handled by EmbeddingGenerator")
    
    def build_index(self, embeddings: List[Embedding]) -> VectorIndex:
        """Build vector index from embeddings"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        try:
            start_time = time.time()
            result = self.active_engine.build_index(embeddings)
            build_time = time.time() - start_time
            
            self.logger.info(
                f"Built index using {self.active_db_type.value}: "
                f"{len(embeddings)} vectors, time={build_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to build index with {self.active_db_type.value}: {e}")
            
            # Attempt fallback
            if self._attempt_fallback():
                return self.build_index(embeddings)
            
            raise VectorIndexError(f"Index building failed: {e}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar content using query string"""
        # This requires embedding generation from query string
        raise NotImplementedError("search_similar with query string requires embedding generator integration")
    
    def search_similar_by_vector(
        self, 
        query_vector: List[float], 
        search_config: Optional[UnifiedSearchConfiguration] = None
    ) -> List[SearchResult]:
        """Search for similar vectors using query vector"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        config = search_config or UnifiedSearchConfiguration()
        
        try:
            start_time = time.time()
            
            # Prepare engine-specific search configuration
            if self.active_db_type == VectorDBType.FAISS:
                faiss_config = FAISSSearchConfig(
                    top_k=config.top_k,
                    similarity_threshold=config.similarity_threshold,
                    include_metadata=config.include_metadata,
                    return_scores=True
                )
                results = self.active_engine.search_similar_by_vector(
                    query_vector, 
                    config.top_k, 
                    config.similarity_threshold, 
                    faiss_config
                )
                
            elif self.active_db_type == VectorDBType.PINECONE:
                pinecone_config = PineconeSearchConfiguration(
                    top_k=config.top_k,
                    similarity_threshold=config.similarity_threshold,
                    namespace=config.namespace,
                    include_metadata=config.include_metadata
                )
                results = self.active_engine.search_similar_by_vector(
                    query_vector, 
                    config.top_k, 
                    config.similarity_threshold, 
                    pinecone_config
                )
            else:
                raise VectorIndexError(f"Unsupported active engine type: {self.active_db_type}")
            
            search_time = time.time() - start_time
            
            # Update stats
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += search_time
            
            if self.active_db_type == VectorDBType.FAISS:
                self.stats['faiss_searches'] += 1
                self.stats['faiss_search_time'] += search_time
            elif self.active_db_type == VectorDBType.PINECONE:
                self.stats['pinecone_searches'] += 1
                self.stats['pinecone_search_time'] += search_time
            
            self.logger.info(
                f"Search completed using {self.active_db_type.value}: "
                f"{len(results)} results, time={search_time:.4f}s"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed with {self.active_db_type.value}: {e}")
            
            # Attempt fallback
            if config.enable_fallback and self._attempt_fallback():
                return self.search_similar_by_vector(query_vector, config)
            
            raise SearchError(f"Vector search failed: {e}")
    
    def update_index(self, new_embeddings: List[Embedding]) -> None:
        """Update existing index with new embeddings"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        try:
            self.active_engine.update_index(new_embeddings)
            
            self.logger.info(
                f"Updated index using {self.active_db_type.value}: "
                f"{len(new_embeddings)} new vectors"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update index with {self.active_db_type.value}: {e}")
            
            # Attempt fallback
            if self._attempt_fallback():
                self.update_index(new_embeddings)
                return
            
            raise VectorIndexError(f"Index update failed: {e}")
    
    def remove_vectors(self, chunk_ids: List[str]) -> int:
        """Remove vectors from index by chunk IDs"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        try:
            removed_count = self.active_engine.remove_vectors(chunk_ids)
            
            self.logger.info(
                f"Removed {removed_count} vectors using {self.active_db_type.value}"
            )
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove vectors with {self.active_db_type.value}: {e}")
            
            # Attempt fallback
            if self._attempt_fallback():
                return self.remove_vectors(chunk_ids)
            
            raise VectorIndexError(f"Vector removal failed: {e}")
    
    def switch_engine(self, db_type: VectorDBType) -> bool:
        """Manually switch to a different engine"""
        try:
            if db_type == VectorDBType.FAISS and self.faiss_engine:
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
                self.stats['engine_switches'] += 1
                self.logger.info("Switched to FAISS engine")
                return True
            elif db_type == VectorDBType.PINECONE and self.pinecone_engine:
                self.active_engine = self.pinecone_engine
                self.active_db_type = VectorDBType.PINECONE
                self.stats['engine_switches'] += 1
                self.logger.info("Switched to Pinecone engine")
                return True
            else:
                self.logger.warning(f"Cannot switch to {db_type.value}: engine not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to switch to {db_type.value}: {e}")
            return False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all available engines"""
        status = {
            "active_engine": self.active_db_type.value if self.active_db_type else None,
            "engines": {}
        }
        
        # FAISS engine status
        if self.faiss_engine:
            try:
                faiss_info = self.faiss_engine.get_index_info()
                status["engines"]["faiss"] = {
                    "available": True,
                    "status": faiss_info.get("status", "unknown"),
                    "total_vectors": faiss_info.get("total_vectors", 0),
                    "dimension": faiss_info.get("dimension", 0)
                }
            except Exception as e:
                status["engines"]["faiss"] = {
                    "available": True,
                    "status": "error",
                    "error": str(e)
                }
        else:
            status["engines"]["faiss"] = {"available": False}
        
        # Pinecone engine status
        if self.pinecone_engine:
            try:
                pinecone_info = self.pinecone_engine.get_index_info()
                status["engines"]["pinecone"] = {
                    "available": True,
                    "status": pinecone_info.get("status", "unknown"),
                    "total_vectors": pinecone_info.get("total_vectors", 0),
                    "dimension": pinecone_info.get("dimension", 0),
                    "current_namespace": pinecone_info.get("current_namespace", "")
                }
            except Exception as e:
                status["engines"]["pinecone"] = {
                    "available": True,
                    "status": "error",
                    "error": str(e)
                }
        else:
            status["engines"]["pinecone"] = {"available": False}
        
        return status
    
    def get_unified_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all engines"""
        unified_stats = self.stats.copy()
        
        # Add engine-specific stats
        if self.faiss_engine:
            unified_stats["faiss_engine_stats"] = self.faiss_engine.get_search_stats()
        
        if self.pinecone_engine:
            unified_stats["pinecone_engine_stats"] = self.pinecone_engine.get_search_stats()
        
        # Calculate derived metrics
        if unified_stats['total_searches'] > 0:
            unified_stats['average_search_time'] = unified_stats['total_search_time'] / unified_stats['total_searches']
        
        if unified_stats['faiss_searches'] > 0:
            unified_stats['average_faiss_search_time'] = unified_stats['faiss_search_time'] / unified_stats['faiss_searches']
        
        if unified_stats['pinecone_searches'] > 0:
            unified_stats['average_pinecone_search_time'] = unified_stats['pinecone_search_time'] / unified_stats['pinecone_searches']
        
        return unified_stats
    
    def optimize_active_index(self) -> Dict[str, Any]:
        """Optimize the currently active index"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        try:
            if hasattr(self.active_engine, 'optimize_index'):
                result = self.active_engine.optimize_index()
                self.logger.info(f"Optimized {self.active_db_type.value} index")
                return result
            else:
                return {"message": f"{self.active_db_type.value} engine does not support optimization"}
                
        except Exception as e:
            self.logger.error(f"Failed to optimize {self.active_db_type.value} index: {e}")
            raise VectorIndexError(f"Index optimization failed: {e}")
    
    def save_active_index(self, index_id: str) -> bool:
        """Save the currently active index (FAISS only)"""
        if not self.active_engine:
            raise VectorIndexError("No active search engine available")
        
        try:
            if self.active_db_type == VectorDBType.FAISS and hasattr(self.active_engine, 'save_index'):
                result = self.active_engine.save_index(index_id)
                self.logger.info(f"Saved FAISS index: {index_id}")
                return result
            else:
                self.logger.info(f"{self.active_db_type.value} engine does not require manual saving")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_id: str) -> bool:
        """Load an index (FAISS only)"""
        if not self.faiss_engine:
            raise VectorIndexError("FAISS engine not available for loading")
        
        try:
            result = self.faiss_engine.load_index(index_id)
            if result:
                # Switch to FAISS engine after successful load
                self.active_engine = self.faiss_engine
                self.active_db_type = VectorDBType.FAISS
                self.logger.info(f"Loaded FAISS index: {index_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'logger') and hasattr(self, 'stats'):
                self.logger.info(f"UnifiedSearchEngine final stats: {self.get_unified_stats()}")
        except Exception:
            pass