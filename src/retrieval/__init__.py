"""
Context retrieval for question answering over documents.

Provides pluggable retrievers used both by the API (to build per-question
context) and by the offline evaluation harness (eval/run_eval.py) that
measures retrieval hit rate:

- FixedChunkTFIDFRetriever   — fixed-size char windows ranked by TF-IDF (baseline)
- SemanticChunkTFIDFRetriever — clause/paragraph-aware chunks ranked by TF-IDF
- RerankedRetriever          — TF-IDF shortlist re-ranked with dense embeddings

All retrievers share the same interface: retrieve(document_text, question,
budget_chars) -> str, and gracefully fall back to TF-IDF when the optional
embedding model is unavailable.
"""

from .context_retrieval import (
    SemanticChunker,
    FixedChunker,
    TfidfRanker,
    EmbeddingReranker,
    FixedChunkTFIDFRetriever,
    SemanticChunkTFIDFRetriever,
    RerankedRetriever,
    build_retriever,
    normalize_ws,
)

__all__ = [
    "SemanticChunker",
    "FixedChunker",
    "TfidfRanker",
    "EmbeddingReranker",
    "FixedChunkTFIDFRetriever",
    "SemanticChunkTFIDFRetriever",
    "RerankedRetriever",
    "build_retriever",
    "normalize_ws",
]
