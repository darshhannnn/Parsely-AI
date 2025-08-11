"""
Embedding Generation System for Document Processing Pipeline

This module implements high-quality embedding generation using sentence-transformers
with batch processing, caching, and performance optimization.
"""

import os
import time
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Third-party imports
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from ...core.models import ContentChunk, Embedding, ProcessingMetadata
from ...core.interfaces import IEmbeddingSearchEngine
from ...core.config import get_config
from ...core.exceptions import EmbeddingGenerationError, CacheError
from ...core.logging_utils import get_logger


@dataclass
class EmbeddingCache:
    """Cache entry for embeddings"""
    embedding: List[float]
    model_name: str
    content_hash: str
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class BatchProcessingResult:
    """Result of batch embedding processing"""
    embeddings: List[Embedding]
    processing_time: float
    cache_hits: int
    cache_misses: int
    errors: List[str] = field(default_factory=list)


class EmbeddingGenerator:
    """
    High-performance embedding generation system with caching and batch processing.
    
    Features:
    - Sentence-transformers integration with multiple model support
    - Intelligent batch processing for efficiency
    - Multi-level caching (memory + disk)
    - Quality validation and consistency checking
    - Performance monitoring and optimization
    - Async processing support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding generator"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Override config if provided
        if config:
            for key, value in config.items():
                if hasattr(self.config.embedding, key):
                    setattr(self.config.embedding, key, value)
        
        # Initialize model and caching
        self.model: Optional[SentenceTransformer] = None
        self.model_lock = threading.Lock()
        self.memory_cache: Dict[str, EmbeddingCache] = {}
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            'total_embeddings_generated': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'average_batch_size': 0.0,
            'model_load_time': 0.0
        }
        
        # Setup directories
        self._setup_directories()
        
        # Initialize model
        self._initialize_model()
        
        self.logger.info(f"EmbeddingGenerator initialized with model: {self.config.embedding.model_name}")
    
    def _setup_directories(self) -> None:
        """Setup required directories"""
        cache_dir = Path(self.config.embedding.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings cache directory
        self.embeddings_cache_dir = cache_dir / "embeddings_cache"
        self.embeddings_cache_dir.mkdir(exist_ok=True)
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model"""
        try:
            start_time = time.time()
            
            with self.model_lock:
                self.model = SentenceTransformer(
                    self.config.embedding.model_name,
                    cache_folder=self.config.embedding.model_cache_dir,
                    device=self.config.embedding.device
                )
                
                # Set model to evaluation mode for consistency
                self.model.eval()
                
                # Configure model settings
                if hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = self.config.embedding.max_sequence_length
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            
            self.logger.info(f"Model loaded in {load_time:.2f}s. Device: {self.config.embedding.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise EmbeddingGenerationError(f"Model initialization failed: {e}")
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content to use as cache key"""
        content_bytes = content.encode('utf-8')
        return hashlib.sha256(content_bytes).hexdigest()[:16]
    
    def _get_cache_key(self, content: str, model_name: str) -> str:
        """Generate cache key for embedding"""
        content_hash = self._get_content_hash(content)
        return f"{model_name}:{content_hash}"
    
    def _load_disk_cache(self, cache_key: str) -> Optional[EmbeddingCache]:
        """Load embedding from disk cache"""
        try:
            cache_file = self.embeddings_cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)
                    
                # Check if cache entry is still valid (not expired)
                if isinstance(cache_entry, EmbeddingCache):
                    cache_age = datetime.now() - cache_entry.created_at
                    if cache_age.total_seconds() < self.config.database.cache_ttl_seconds:
                        cache_entry.access_count += 1
                        cache_entry.last_accessed = datetime.now()
                        return cache_entry
                    else:
                        # Remove expired cache
                        cache_file.unlink()
                        
        except Exception as e:
            self.logger.warning(f"Failed to load disk cache for {cache_key}: {e}")
        
        return None
    
    def _save_disk_cache(self, cache_key: str, cache_entry: EmbeddingCache) -> None:
        """Save embedding to disk cache"""
        try:
            cache_file = self.embeddings_cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
        except Exception as e:
            self.logger.warning(f"Failed to save disk cache for {cache_key}: {e}")
    
    def _get_cached_embedding(self, content: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache (memory or disk)"""
        cache_key = self._get_cache_key(content, model_name)
        
        # Check memory cache first
        with self.cache_lock:
            if cache_key in self.memory_cache:
                cache_entry = self.memory_cache[cache_key]
                cache_entry.access_count += 1
                cache_entry.last_accessed = datetime.now()
                self.stats['cache_hits'] += 1
                return cache_entry.embedding
        
        # Check disk cache
        cache_entry = self._load_disk_cache(cache_key)
        if cache_entry:
            # Add to memory cache for faster access
            with self.cache_lock:
                self.memory_cache[cache_key] = cache_entry
            self.stats['cache_hits'] += 1
            return cache_entry.embedding
        
        self.stats['cache_misses'] += 1
        return None
    
    def _cache_embedding(self, content: str, model_name: str, embedding: List[float]) -> None:
        """Cache embedding in memory and disk"""
        cache_key = self._get_cache_key(content, model_name)
        content_hash = self._get_content_hash(content)
        
        cache_entry = EmbeddingCache(
            embedding=embedding,
            model_name=model_name,
            content_hash=content_hash,
            created_at=datetime.now()
        )
        
        # Add to memory cache
        with self.cache_lock:
            self.memory_cache[cache_key] = cache_entry
            
            # Limit memory cache size (keep most recently used)
            if len(self.memory_cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
                for key, _ in sorted_items[:100]:  # Remove 100 oldest
                    del self.memory_cache[key]
        
        # Save to disk cache
        self._save_disk_cache(cache_key, cache_entry)
    
    def _validate_embedding_quality(self, embeddings: List[List[float]], contents: List[str]) -> Dict[str, Any]:
        """Validate embedding quality and consistency"""
        if not embeddings or not contents:
            return {"valid": False, "reason": "Empty embeddings or contents"}
        
        try:
            # Check dimensions consistency
            dimensions = [len(emb) for emb in embeddings]
            if len(set(dimensions)) > 1:
                return {"valid": False, "reason": "Inconsistent embedding dimensions"}
            
            # Check for NaN or infinite values
            for i, embedding in enumerate(embeddings):
                if any(not np.isfinite(val) for val in embedding):
                    return {"valid": False, "reason": f"Invalid values in embedding {i}"}
            
            # Check embedding magnitude (should not be zero vectors)
            for i, embedding in enumerate(embeddings):
                magnitude = np.linalg.norm(embedding)
                if magnitude < 1e-6:
                    return {"valid": False, "reason": f"Zero or near-zero embedding {i}"}
            
            # Quality metrics
            embeddings_array = np.array(embeddings)
            
            # Calculate average cosine similarity (diversity check)
            if len(embeddings) > 1:
                similarities = cosine_similarity(embeddings_array)
                # Remove diagonal (self-similarity)
                mask = ~np.eye(similarities.shape[0], dtype=bool)
                avg_similarity = similarities[mask].mean()
                
                # Very high average similarity might indicate poor quality
                if avg_similarity > 0.95:
                    return {
                        "valid": True,
                        "warning": f"High average similarity ({avg_similarity:.3f}) - possible quality issue"
                    }
            
            return {
                "valid": True,
                "dimension": dimensions[0],
                "count": len(embeddings),
                "avg_magnitude": float(np.mean([np.linalg.norm(emb) for emb in embeddings]))
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {e}"}
    
    def generate_single_embedding(self, content: str) -> Embedding:
        """Generate embedding for a single piece of content"""
        if not content or not content.strip():
            raise EmbeddingGenerationError("Empty content provided")
        
        model_name = self.config.embedding.model_name
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(content, model_name)
        if cached_embedding:
            return Embedding(
                chunk_id="",  # Will be set by caller
                vector=cached_embedding,
                model_name=model_name,
                dimension=len(cached_embedding)
            )
        
        # Generate new embedding
        try:
            start_time = time.time()
            
            with self.model_lock:
                if self.model is None:
                    raise EmbeddingGenerationError("Model not initialized")
                
                # Generate embedding
                embedding_tensor = self.model.encode(
                    content,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=1
                )
                
                # Convert to list
                if torch.is_tensor(embedding_tensor):
                    embedding_list = embedding_tensor.cpu().numpy().tolist()
                else:
                    embedding_list = embedding_tensor.tolist()
            
            processing_time = time.time() - start_time
            
            # Validate embedding
            validation = self._validate_embedding_quality([embedding_list], [content])
            if not validation["valid"]:
                raise EmbeddingGenerationError(f"Invalid embedding: {validation['reason']}")
            
            # Cache the embedding
            self._cache_embedding(content, model_name, embedding_list)
            
            # Update stats
            self.stats['total_embeddings_generated'] += 1
            self.stats['total_processing_time'] += processing_time
            
            return Embedding(
                chunk_id="",  # Will be set by caller
                vector=embedding_list,
                model_name=model_name,
                dimension=len(embedding_list)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}")
    
    def generate_batch_embeddings(
        self, 
        contents: List[str], 
        batch_size: Optional[int] = None
    ) -> BatchProcessingResult:
        """Generate embeddings for multiple contents with batch processing"""
        if not contents:
            return BatchProcessingResult(
                embeddings=[],
                processing_time=0.0,
                cache_hits=0,
                cache_misses=0
            )
        
        start_time = time.time()
        batch_size = batch_size or self.config.embedding.batch_size
        model_name = self.config.embedding.model_name
        
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        errors = []
        
        # Process contents in batches
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            batch_embeddings = []
            uncached_contents = []
            uncached_indices = []
            
            # Check cache for each content in batch
            for j, content in enumerate(batch_contents):
                if not content or not content.strip():
                    errors.append(f"Empty content at index {i + j}")
                    batch_embeddings.append(None)
                    continue
                
                cached_embedding = self._get_cached_embedding(content, model_name)
                if cached_embedding:
                    batch_embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)
                    uncached_contents.append(content)
                    uncached_indices.append(j)
                    cache_misses += 1
            
            # Generate embeddings for uncached contents
            if uncached_contents:
                try:
                    with self.model_lock:
                        if self.model is None:
                            raise EmbeddingGenerationError("Model not initialized")
                        
                        # Generate batch embeddings
                        embedding_tensors = self.model.encode(
                            uncached_contents,
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            batch_size=len(uncached_contents)
                        )
                        
                        # Convert to lists
                        if torch.is_tensor(embedding_tensors):
                            embedding_lists = embedding_tensors.cpu().numpy().tolist()
                        else:
                            embedding_lists = embedding_tensors.tolist()
                    
                    # Validate batch embeddings
                    validation = self._validate_embedding_quality(embedding_lists, uncached_contents)
                    if not validation["valid"]:
                        errors.append(f"Batch validation failed: {validation['reason']}")
                        # Still continue with individual validation
                    
                    # Cache and assign embeddings
                    for idx, (content, embedding_list) in zip(uncached_indices, zip(uncached_contents, embedding_lists)):
                        # Individual validation
                        individual_validation = self._validate_embedding_quality([embedding_list], [content])
                        if individual_validation["valid"]:
                            batch_embeddings[idx] = embedding_list
                            self._cache_embedding(content, model_name, embedding_list)
                        else:
                            errors.append(f"Invalid embedding at index {i + idx}: {individual_validation['reason']}")
                
                except Exception as e:
                    error_msg = f"Batch processing failed for batch starting at {i}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Create Embedding objects for valid embeddings
            for j, embedding_list in enumerate(batch_embeddings):
                if embedding_list is not None:
                    embeddings.append(Embedding(
                        chunk_id="",  # Will be set by caller
                        vector=embedding_list,
                        model_name=model_name,
                        dimension=len(embedding_list)
                    ))
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats['total_embeddings_generated'] += len(embeddings)
        self.stats['total_processing_time'] += processing_time
        self.stats['batch_operations'] += 1
        self.stats['average_batch_size'] = (
            (self.stats['average_batch_size'] * (self.stats['batch_operations'] - 1) + len(contents)) /
            self.stats['batch_operations']
        )
        
        self.logger.info(
            f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s "
            f"(cache hits: {cache_hits}, misses: {cache_misses}, errors: {len(errors)})"
        )
        
        return BatchProcessingResult(
            embeddings=embeddings,
            processing_time=processing_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors
        )
    
    def create_embeddings(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """Create embeddings for content chunks (implements IEmbeddingSearchEngine interface)"""
        if not chunks:
            return []
        
        # Extract content from chunks
        contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        result = self.generate_batch_embeddings(contents)
        
        # Assign chunk IDs to embeddings
        embeddings = result.embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if i < len(embeddings):
                embedding.chunk_id = chunk.id
        
        # Log any errors
        if result.errors:
            self.logger.warning(f"Embedding generation errors: {result.errors}")
        
        return embeddings
    
    async def create_embeddings_async(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """Async version of create_embeddings"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, self.create_embeddings, chunks)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['total_embeddings_generated'] > 0:
            stats['average_processing_time_per_embedding'] = (
                stats['total_processing_time'] / stats['total_embeddings_generated']
            )
        else:
            stats['average_processing_time_per_embedding'] = 0.0
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Memory cache info
        with self.cache_lock:
            stats['memory_cache_size'] = len(self.memory_cache)
        
        # Model info
        stats['model_name'] = self.config.embedding.model_name
        stats['model_device'] = self.config.embedding.device
        stats['batch_size'] = self.config.embedding.batch_size
        
        return stats
    
    def clear_cache(self, memory_only: bool = False) -> None:
        """Clear embedding cache"""
        # Clear memory cache
        with self.cache_lock:
            self.memory_cache.clear()
        
        # Clear disk cache if requested
        if not memory_only:
            try:
                for cache_file in self.embeddings_cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Disk cache cleared")
            except Exception as e:
                self.logger.warning(f"Failed to clear disk cache: {e}")
        
        self.logger.info("Memory cache cleared")
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        removed_count = 0
        current_time = datetime.now()
        ttl_seconds = self.config.database.cache_ttl_seconds
        
        # Clean memory cache
        with self.cache_lock:
            expired_keys = []
            for key, cache_entry in self.memory_cache.items():
                if (current_time - cache_entry.created_at).total_seconds() > ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                removed_count += 1
        
        # Clean disk cache
        try:
            for cache_file in self.embeddings_cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cache_entry = pickle.load(f)
                    
                    if isinstance(cache_entry, EmbeddingCache):
                        cache_age = current_time - cache_entry.created_at
                        if cache_age.total_seconds() > ttl_seconds:
                            cache_file.unlink()
                            removed_count += 1
                except Exception:
                    # Remove corrupted cache files
                    cache_file.unlink()
                    removed_count += 1
        except Exception as e:
            self.logger.warning(f"Failed to clean disk cache: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Log final stats
            if hasattr(self, 'logger') and hasattr(self, 'stats'):
                self.logger.info(f"EmbeddingGenerator final stats: {self.get_embedding_stats()}")
        except Exception:
            pass