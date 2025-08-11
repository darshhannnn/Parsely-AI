"""
Stage 3: Embedding Search - Vector-based semantic search using embeddings
"""

from .embedding_generator import (
    EmbeddingGenerator,
    EmbeddingCache,
    BatchProcessingResult
)
from .faiss_search_engine import (
    FAISSSearchEngine,
    FAISSIndexMetadata,
    SearchConfiguration
)
from .pinecone_search_engine import (
    PineconeSearchEngine,
    PineconeIndexMetadata,
    PineconeSearchConfiguration
)
from .unified_search_engine import (
    UnifiedSearchEngine,
    VectorDBType,
    UnifiedSearchConfiguration
)

__all__ = [
    'EmbeddingGenerator',
    'EmbeddingCache',
    'BatchProcessingResult',
    'FAISSSearchEngine',
    'FAISSIndexMetadata',
    'SearchConfiguration',
    'PineconeSearchEngine',
    'PineconeIndexMetadata',
    'PineconeSearchConfiguration',
    'UnifiedSearchEngine',
    'VectorDBType',
    'UnifiedSearchConfiguration'
]