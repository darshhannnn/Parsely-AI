"""
FAISS Vector Database Integration for Document Processing Pipeline

This module implements high-performance vector similarity search using FAISS
with index persistence, optimization, and comprehensive search capabilities.
"""

import os
import time
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, field
import threading
import json

# Third-party imports
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from ...core.models import ContentChunk, Embedding, VectorIndex, SearchResult, ProcessingMetadata
from ...core.interfaces import IEmbeddingSearchEngine
from ...core.config import get_config
from ...core.exceptions import VectorIndexError, FAISSError, SearchError
from ...core.logging_utils import get_logger


@dataclass
class FAISSIndexMetadata:
    """Metadata for FAISS index"""
    index_id: str
    index_type: str  # "flat", "ivf", "hnsw", "pq"
    dimension: int
    total_vectors: int
    created_at: datetime
    last_updated: datetime
    index_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchConfiguration:
    """Configuration for search operations"""
    top_k: int = 5
    similarity_threshold: float = 0.1
    search_type: str = "cosine"  # "cosine", "l2", "inner_product"
    nprobe: int = 10  # For IVF indexes
    ef_search: int = 64  # For HNSW indexes
    include_metadata: bool = True
    return_scores: bool = True


class FAISSSearchEngine:
    """
    High-performance FAISS-based vector search engine.
    
    Features:
    - Multiple index types (Flat, IVF, HNSW, PQ)
    - Index persistence and loading
    - Configurable similarity search
    - Performance optimization
    - Thread-safe operations
    - Comprehensive error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the FAISS search engine"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Override config if provided
        if config:
            for key, value in config.items():
                if hasattr(self.config.database, key):
                    setattr(self.config.database, key, value)
        
        # Initialize FAISS components
        self.index: Optional[faiss.Index] = None
        self.index_metadata: Optional[FAISSIndexMetadata] = None
        self.chunk_id_mapping: Dict[int, str] = {}  # Maps FAISS index positions to chunk IDs
        self.chunk_metadata: Dict[str, ContentChunk] = {}  # Maps chunk IDs to chunk objects
        self.embeddings_store: Dict[str, List[float]] = {}  # Maps chunk IDs to embeddings
        
        # Thread safety
        self.index_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'average_search_time': 0.0,
            'index_builds': 0,
            'total_vectors_indexed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Setup directories
        self._setup_directories()
        
        self.logger.info("FAISSSearchEngine initialized")
    
    def _setup_directories(self) -> None:
        """Setup required directories for index storage"""
        index_dir = Path(self.config.database.faiss_index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = index_dir
    
    def _get_index_file_path(self, index_id: str) -> Path:
        """Get file path for index storage"""
        return self.index_dir / f"{index_id}.faiss"
    
    def _get_metadata_file_path(self, index_id: str) -> Path:
        """Get file path for metadata storage"""
        return self.index_dir / f"{index_id}_metadata.pkl"
    
    def _get_mappings_file_path(self, index_id: str) -> Path:
        """Get file path for chunk mappings storage"""
        return self.index_dir / f"{index_id}_mappings.pkl"
    
    def _create_index(self, dimension: int, index_type: str = "flat", **kwargs) -> faiss.Index:
        """Create FAISS index based on type and parameters"""
        try:
            if index_type.lower() == "flat":
                # Flat index for exact search
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
            elif index_type.lower() == "ivf":
                # IVF index for faster approximate search
                nlist = kwargs.get('nlist', min(100, max(1, dimension // 4)))
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
            elif index_type.lower() == "hnsw":
                # HNSW index for very fast approximate search
                m = kwargs.get('m', 16)
                index = faiss.IndexHNSWFlat(dimension, m)
                index.hnsw.efConstruction = kwargs.get('ef_construction', 200)
                
            elif index_type.lower() == "pq":
                # Product Quantization for memory efficiency
                m = kwargs.get('m', min(8, dimension // 4))
                nbits = kwargs.get('nbits', 8)
                index = faiss.IndexPQ(dimension, m, nbits)
                
            else:
                raise FAISSError(f"Unsupported index type: {index_type}")
            
            self.logger.info(f"Created FAISS index: type={index_type}, dimension={dimension}")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}")
            raise FAISSError(f"Index creation failed: {e}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def build_index(self, embeddings: List[Embedding]) -> VectorIndex:
        """Build FAISS index from embeddings"""
        if not embeddings:
            raise FAISSError("Cannot build index from empty embeddings list")
        
        try:
            with self.index_lock:
                # Extract vectors and metadata
                vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
                dimension = vectors.shape[1]
                
                # Normalize vectors for cosine similarity
                vectors = self._normalize_embeddings(vectors)
                
                # Determine index type based on size
                num_vectors = len(embeddings)
                if num_vectors < 1000:
                    index_type = "flat"
                elif num_vectors < 10000:
                    index_type = "ivf"
                else:
                    index_type = "hnsw"
                
                # Create index
                self.index = self._create_index(dimension, index_type)
                
                # Train index if needed (for IVF)
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    self.logger.info("Training FAISS index...")
                    self.index.train(vectors)
                
                # Add vectors to index
                start_time = time.time()
                self.index.add(vectors)
                build_time = time.time() - start_time
                
                # Update mappings
                self.chunk_id_mapping.clear()
                self.chunk_metadata.clear()
                self.embeddings_store.clear()
                
                for i, embedding in enumerate(embeddings):
                    self.chunk_id_mapping[i] = embedding.chunk_id
                    if hasattr(embedding, 'chunk_metadata'):
                        self.chunk_metadata[embedding.chunk_id] = embedding.chunk_metadata
                    self.embeddings_store[embedding.chunk_id] = embedding.vector
                
                # Create metadata
                index_id = f"faiss_index_{int(time.time())}"
                self.index_metadata = FAISSIndexMetadata(
                    index_id=index_id,
                    index_type=index_type,
                    dimension=dimension,
                    total_vectors=num_vectors,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    index_parameters={
                        'index_type': index_type,
                        'dimension': dimension,
                        'num_vectors': num_vectors
                    },
                    performance_stats={
                        'build_time': build_time,
                        'vectors_per_second': num_vectors / build_time if build_time > 0 else 0
                    }
                )
                
                # Update stats
                self.stats['index_builds'] += 1
                self.stats['total_vectors_indexed'] += num_vectors
                
                self.logger.info(
                    f"Built FAISS index: {num_vectors} vectors, {dimension}D, "
                    f"type={index_type}, time={build_time:.3f}s"
                )
                
                return VectorIndex(
                    index_id=index_id,
                    index_type="faiss",
                    dimension=dimension,
                    total_vectors=num_vectors,
                    metadata={
                        'faiss_index_type': index_type,
                        'build_time': build_time,
                        'performance_stats': self.index_metadata.performance_stats
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to build FAISS index: {e}")
            raise FAISSError(f"Index building failed: {e}")
    
    def save_index(self, index_id: str) -> bool:
        """Save FAISS index and metadata to disk"""
        if not self.index or not self.index_metadata:
            raise FAISSError("No index to save")
        
        try:
            with self.index_lock:
                # Save FAISS index
                index_file = self._get_index_file_path(index_id)
                faiss.write_index(self.index, str(index_file))
                
                # Save metadata
                metadata_file = self._get_metadata_file_path(index_id)
                with open(metadata_file, 'wb') as f:
                    pickle.dump(self.index_metadata, f)
                
                # Save mappings
                mappings_file = self._get_mappings_file_path(index_id)
                mappings_data = {
                    'chunk_id_mapping': self.chunk_id_mapping,
                    'chunk_metadata': self.chunk_metadata,
                    'embeddings_store': self.embeddings_store
                }
                with open(mappings_file, 'wb') as f:
                    pickle.dump(mappings_data, f)
                
                self.logger.info(f"Saved FAISS index: {index_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")
            raise FAISSError(f"Index saving failed: {e}")
    
    def load_index(self, index_id: str) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            with self.index_lock:
                # Load FAISS index
                index_file = self._get_index_file_path(index_id)
                if not index_file.exists():
                    raise FAISSError(f"Index file not found: {index_file}")
                
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                metadata_file = self._get_metadata_file_path(index_id)
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        self.index_metadata = pickle.load(f)
                else:
                    # Create basic metadata if not found
                    self.index_metadata = FAISSIndexMetadata(
                        index_id=index_id,
                        index_type="unknown",
                        dimension=self.index.d,
                        total_vectors=self.index.ntotal,
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                
                # Load mappings
                mappings_file = self._get_mappings_file_path(index_id)
                if mappings_file.exists():
                    with open(mappings_file, 'rb') as f:
                        mappings_data = pickle.load(f)
                        self.chunk_id_mapping = mappings_data.get('chunk_id_mapping', {})
                        self.chunk_metadata = mappings_data.get('chunk_metadata', {})
                        self.embeddings_store = mappings_data.get('embeddings_store', {})
                
                self.logger.info(f"Loaded FAISS index: {index_id}, {self.index.ntotal} vectors")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            raise FAISSError(f"Index loading failed: {e}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar vectors using query string"""
        # This method would typically use an embedding generator to convert query to vector
        # For now, we'll raise an error indicating this needs to be implemented with embedding generator
        raise NotImplementedError("search_similar with query string requires embedding generator integration")
    
    def search_similar_by_vector(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        search_config: Optional[SearchConfiguration] = None
    ) -> List[SearchResult]:
        """Search for similar vectors using query vector"""
        if not self.index:
            raise FAISSError("No index available for search")
        
        try:
            with self.index_lock:
                start_time = time.time()
                
                # Prepare query vector
                query_array = np.array([query_vector], dtype=np.float32)
                query_array = self._normalize_embeddings(query_array)
                
                # Configure search parameters
                if search_config:
                    if hasattr(self.index, 'nprobe'):
                        self.index.nprobe = search_config.nprobe
                    if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'efSearch'):
                        self.index.hnsw.efSearch = search_config.ef_search
                
                # Perform search
                scores, indices = self.index.search(query_array, top_k)
                search_time = time.time() - start_time
                
                # Process results
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1:  # FAISS returns -1 for invalid results
                        continue
                    
                    # Convert inner product back to cosine similarity
                    similarity_score = float(score)
                    
                    # Apply similarity threshold
                    if similarity_score < similarity_threshold:
                        continue
                    
                    # Get chunk information
                    chunk_id = self.chunk_id_mapping.get(int(idx), f"unknown_{idx}")
                    chunk_metadata = self.chunk_metadata.get(chunk_id, {})
                    
                    # Create search result
                    result = SearchResult(
                        chunk_id=chunk_id,
                        content=getattr(chunk_metadata, 'content', '') if hasattr(chunk_metadata, 'content') else '',
                        similarity_score=similarity_score,
                        rank=i + 1,
                        metadata={
                            'faiss_index': int(idx),
                            'search_time': search_time,
                            'chunk_metadata': chunk_metadata.__dict__ if hasattr(chunk_metadata, '__dict__') else {}
                        }
                    )
                    results.append(result)
                
                # Update stats
                self.stats['total_searches'] += 1
                self.stats['total_search_time'] += search_time
                self.stats['average_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
                
                self.logger.info(
                    f"FAISS search completed: {len(results)} results, "
                    f"time={search_time:.4f}s, top_k={top_k}"
                )
                
                return results
                
        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}")
            raise SearchError(f"Vector search failed: {e}")
    
    def update_index(self, new_embeddings: List[Embedding]) -> None:
        """Update existing index with new embeddings"""
        if not self.index:
            raise FAISSError("No existing index to update")
        
        try:
            with self.index_lock:
                # Extract new vectors
                new_vectors = np.array([emb.vector for emb in new_embeddings], dtype=np.float32)
                new_vectors = self._normalize_embeddings(new_vectors)
                
                # Add to index
                start_idx = self.index.ntotal
                self.index.add(new_vectors)
                
                # Update mappings
                for i, embedding in enumerate(new_embeddings):
                    faiss_idx = start_idx + i
                    self.chunk_id_mapping[faiss_idx] = embedding.chunk_id
                    if hasattr(embedding, 'chunk_metadata'):
                        self.chunk_metadata[embedding.chunk_id] = embedding.chunk_metadata
                    self.embeddings_store[embedding.chunk_id] = embedding.vector
                
                # Update metadata
                if self.index_metadata:
                    self.index_metadata.total_vectors = self.index.ntotal
                    self.index_metadata.last_updated = datetime.now()
                
                self.logger.info(f"Updated FAISS index with {len(new_embeddings)} new vectors")
                
        except Exception as e:
            self.logger.error(f"Failed to update FAISS index: {e}")
            raise FAISSError(f"Index update failed: {e}")
    
    def remove_vectors(self, chunk_ids: List[str]) -> int:
        """Remove vectors from index by chunk IDs"""
        # FAISS doesn't support direct removal, so we need to rebuild
        # This is a limitation of FAISS - for frequent updates, consider other solutions
        if not self.index or not chunk_ids:
            return 0
        
        try:
            with self.index_lock:
                # Find indices to remove
                indices_to_remove = set()
                for faiss_idx, chunk_id in self.chunk_id_mapping.items():
                    if chunk_id in chunk_ids:
                        indices_to_remove.add(faiss_idx)
                
                if not indices_to_remove:
                    return 0
                
                # Rebuild index without removed vectors
                remaining_embeddings = []
                for faiss_idx, chunk_id in self.chunk_id_mapping.items():
                    if faiss_idx not in indices_to_remove:
                        if chunk_id in self.embeddings_store:
                            embedding = Embedding(
                                chunk_id=chunk_id,
                                vector=self.embeddings_store[chunk_id],
                                model_name="unknown",
                                dimension=len(self.embeddings_store[chunk_id])
                            )
                            if chunk_id in self.chunk_metadata:
                                embedding.chunk_metadata = self.chunk_metadata[chunk_id]
                            remaining_embeddings.append(embedding)
                
                # Rebuild index
                if remaining_embeddings:
                    self.build_index(remaining_embeddings)
                else:
                    # Clear index if no vectors remain
                    self.index = None
                    self.index_metadata = None
                    self.chunk_id_mapping.clear()
                    self.chunk_metadata.clear()
                    self.embeddings_store.clear()
                
                # Clean up removed chunk data
                for chunk_id in chunk_ids:
                    self.chunk_metadata.pop(chunk_id, None)
                    self.embeddings_store.pop(chunk_id, None)
                
                self.logger.info(f"Removed {len(indices_to_remove)} vectors from FAISS index")
                return len(indices_to_remove)
                
        except Exception as e:
            self.logger.error(f"Failed to remove vectors from FAISS index: {e}")
            raise FAISSError(f"Vector removal failed: {e}")
    
    def optimize_index(self) -> Dict[str, Any]:
        """Optimize index for better performance"""
        if not self.index:
            raise FAISSError("No index to optimize")
        
        try:
            with self.index_lock:
                optimization_stats = {
                    'original_size': self.index.ntotal,
                    'optimization_time': 0.0,
                    'optimizations_applied': []
                }
                
                start_time = time.time()
                
                # For IVF indexes, we can optimize nprobe
                if hasattr(self.index, 'nprobe'):
                    # Set optimal nprobe based on index size
                    optimal_nprobe = min(max(1, self.index.nlist // 10), 100)
                    self.index.nprobe = optimal_nprobe
                    optimization_stats['optimizations_applied'].append(f'nprobe={optimal_nprobe}')
                
                # For HNSW indexes, we can optimize efSearch
                if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'efSearch'):
                    # Set optimal efSearch based on accuracy/speed tradeoff
                    optimal_ef = min(max(16, self.index.ntotal // 100), 512)
                    self.index.hnsw.efSearch = optimal_ef
                    optimization_stats['optimizations_applied'].append(f'efSearch={optimal_ef}')
                
                optimization_time = time.time() - start_time
                optimization_stats['optimization_time'] = optimization_time
                
                # Update metadata
                if self.index_metadata:
                    self.index_metadata.performance_stats['last_optimization'] = optimization_time
                    self.index_metadata.performance_stats['optimizations_applied'] = len(optimization_stats['optimizations_applied'])
                
                self.logger.info(f"Optimized FAISS index: {optimization_stats}")
                return optimization_stats
                
        except Exception as e:
            self.logger.error(f"Failed to optimize FAISS index: {e}")
            raise FAISSError(f"Index optimization failed: {e}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get comprehensive index information"""
        if not self.index:
            return {"status": "no_index"}
        
        info = {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, 'is_trained', True),
            "memory_usage_bytes": self.index.ntotal * self.index.d * 4,  # Approximate
            "chunk_mappings": len(self.chunk_id_mapping),
            "metadata_entries": len(self.chunk_metadata),
            "embeddings_stored": len(self.embeddings_store)
        }
        
        if self.index_metadata:
            info.update({
                "index_id": self.index_metadata.index_id,
                "created_at": self.index_metadata.created_at.isoformat(),
                "last_updated": self.index_metadata.last_updated.isoformat(),
                "index_parameters": self.index_metadata.index_parameters,
                "performance_stats": self.index_metadata.performance_stats
            })
        
        # Add FAISS-specific info
        if hasattr(self.index, 'nprobe'):
            info["nprobe"] = self.index.nprobe
        if hasattr(self.index, 'nlist'):
            info["nlist"] = self.index.nlist
        if hasattr(self.index, 'hnsw'):
            info["hnsw_m"] = getattr(self.index.hnsw, 'M', None)
            info["hnsw_ef_construction"] = getattr(self.index.hnsw, 'efConstruction', None)
            info["hnsw_ef_search"] = getattr(self.index.hnsw, 'efSearch', None)
        
        return info
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return self.stats.copy()
    
    def clear_index(self) -> None:
        """Clear the current index and all associated data"""
        with self.index_lock:
            self.index = None
            self.index_metadata = None
            self.chunk_id_mapping.clear()
            self.chunk_metadata.clear()
            self.embeddings_store.clear()
            
        self.logger.info("Cleared FAISS index and all associated data")
    
    def list_saved_indexes(self) -> List[str]:
        """List all saved index IDs"""
        try:
            index_files = list(self.index_dir.glob("*.faiss"))
            index_ids = [f.stem for f in index_files]
            return sorted(index_ids)
        except Exception as e:
            self.logger.error(f"Failed to list saved indexes: {e}")
            return []
    
    def delete_saved_index(self, index_id: str) -> bool:
        """Delete a saved index and its associated files"""
        try:
            # Delete index file
            index_file = self._get_index_file_path(index_id)
            if index_file.exists():
                index_file.unlink()
            
            # Delete metadata file
            metadata_file = self._get_metadata_file_path(index_id)
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Delete mappings file
            mappings_file = self._get_mappings_file_path(index_id)
            if mappings_file.exists():
                mappings_file.unlink()
            
            self.logger.info(f"Deleted saved index: {index_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete saved index {index_id}: {e}")
            return False
    
    # Implement IEmbeddingSearchEngine interface methods
    def create_embeddings(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """This method should be handled by EmbeddingGenerator"""
        raise NotImplementedError("create_embeddings should be handled by EmbeddingGenerator")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, 'logger') and hasattr(self, 'stats'):
                self.logger.info(f"FAISSSearchEngine final stats: {self.get_search_stats()}")
        except Exception:
            pass