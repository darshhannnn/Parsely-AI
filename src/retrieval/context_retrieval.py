"""
Context retrieval implementations shared by the API and the evaluation harness.

A retriever selects, for one question, the excerpt of a document that fits a
character budget. The quality of that selection decides answer accuracy, and
`eval/run_eval.py` measures it as retrieval hit rate against a labelled
question set.

Retrievers available:
- FixedChunkTFIDFRetriever    (baseline: fixed char windows + TF-IDF ranking)
- SemanticChunkTFIDFRetriever (clause/paragraph-aware chunking + TF-IDF ranking)
- RerankedRetriever           (semantic chunks + TF-IDF shortlist re-ranked by
                               dense embeddings; falls back to TF-IDF when the
                               optional sentence-transformers model is missing)
"""

import logging
import re
import threading
from typing import List, Protocol

logger = logging.getLogger(__name__)

_CLAUSE_HEADER = re.compile(
    r"(?:(?<=\n)|\A)\s*(?:Def\.?\s*)?(\d{1,3})\s*[.):]?\s+[A-Z][^.!?]{0,90}?(?:[:-]|—|\n)"
)


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to single spaces (PDF text is ragged)."""
    return re.sub(r"\s+", " ", text or "").strip()


class Chunker(Protocol):
    def chunk(self, text: str) -> List[str]: ...


class FixedChunker:
    """Fixed-size character windows with a small overlap (the baseline)."""

    def __init__(self, chunk_chars: int = 2000):
        self.chunk_chars = chunk_chars

    def chunk(self, text: str) -> List[str]:
        text = text or ""
        if len(text) <= self.chunk_chars:
            return [text] if text.strip() else []
        chunks = []
        step = int(self.chunk_chars * 0.9)  # 10% overlap between windows
        for i in range(0, len(text), step):
            piece = text[i:i + self.chunk_chars]
            if piece.strip():
                chunks.append(piece)
            if i + self.chunk_chars >= len(text):
                break
        return chunks


class SemanticChunker:
    """Clause/paragraph-aware chunking.

    PDF extractions lose most formatting, so chunks are built around the
    policy's own structure: numbered clause headers ("12. Grace Period...",
    "Def. 18. Grace Period...") when detectable, paragraph breaks otherwise,
    then merged into chunks of at most max_chunk_chars. Each chunk stays a
    semantically complete unit instead of a mid-sentence window.
    """

    def __init__(self, max_chunk_chars: int = 2000, min_chunk_chars: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def _split_units(self, text: str) -> List[str]:
        """Split into semantic units at clause headers, paragraphs or sentences."""
        header_positions = [m.start() for m in _CLAUSE_HEADER.finditer(text)]
        if len(header_positions) >= 2:
            units = []
            for i, start in enumerate(header_positions):
                end = header_positions[i + 1] if i + 1 < len(header_positions) else len(text)
                unit = text[start:end].strip()
                if unit:
                    units.append(unit)
            # keep any preamble before the first header
            preamble = text[:header_positions[0]].strip()
            if preamble:
                units.insert(0, preamble)
        else:
            units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Further split oversized units on sentence boundaries
        final_units: List[str] = []
        for unit in units:
            if len(unit) <= self.max_chunk_chars:
                final_units.append(unit)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", unit)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > self.max_chunk_chars and current:
                    final_units.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                final_units.append(current)
        return final_units

    def chunk(self, text: str) -> List[str]:
        units = self._split_units(text or "")
        if not units:
            return []

        chunks: List[str] = []
        current = ""
        for unit in units:
            if len(unit) > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(unit), self.max_chunk_chars):
                    pieces = unit[i:i + self.max_chunk_chars]
                    if pieces.strip():
                        chunks.append(pieces)
                continue
            if len(current) + len(unit) + 2 <= self.max_chunk_chars:
                current = f"{current}\n\n{unit}" if current else unit
            else:
                chunks.append(current)
                current = unit
        if current:
            chunks.append(current)

        # Merge tiny trailing chunks into their neighbour
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_chars:
            chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
            chunks.pop()
        return chunks


class TfidfRanker:
    """Ranks chunks by TF-IDF cosine similarity to the question."""

    def rank(self, chunks: List[str], question: str) -> List[int]:
        if not chunks:
            return []
        if len(chunks) == 1:
            return [0]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectors = TfidfVectorizer(stop_words="english").fit_transform(chunks + [question])
            similarities = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
            return sorted(range(len(chunks)), key=lambda i: similarities[i], reverse=True)
        except Exception as e:
            logger.warning(f"TF-IDF ranking failed, using document order: {e}")
            return list(range(len(chunks)))


class EmbeddingReranker:
    """Re-ranks a TF-IDF shortlist with dense sentence embeddings.

    Whole 2000-char chunks embed poorly: sentence-transformers models
    truncate to ~256 tokens and mean-pooling dilutes the vector across many
    topics, so dense ranking on full chunks loses to TF-IDF (measured in
    eval/RESULTS.md). Instead, each shortlisted chunk is split into small
    sentence windows; every window is embedded and the chunk's score is the
    similarity of its best window (a simplified max-sim re-ranker).

    The sentence-transformers model is heavy and optional: it is loaded once
    (lazily, thread-safely) and any failure disables re-ranking, letting
    callers fall back to the TF-IDF order.
    """

    _model = None
    _model_lock = threading.Lock()
    _load_failed = False

    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    def _get_model(cls):
        if cls._model is not None or cls._load_failed:
            return cls._model
        with cls._model_lock:
            if cls._model is None and not cls._load_failed:
                try:
                    from sentence_transformers import SentenceTransformer

                    cls._model = SentenceTransformer(cls.MODEL_NAME)
                    logger.info(f"EmbeddingReranker loaded model {cls.MODEL_NAME}")
                except Exception as e:
                    logger.warning(f"Embedding re-ranking unavailable, using TF-IDF only: {e}")
                    cls._load_failed = True
        return cls._model

    def __init__(self, shortlist_size: int = 24, window_chars: int = 300,
                 rrf_k: int = 60):
        self.shortlist_size = shortlist_size
        self.window_chars = window_chars
        self.rrf_k = rrf_k

    def _windows(self, text: str) -> List[str]:
        """Split a chunk into overlapping sentence windows."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return [text]
        windows = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.window_chars and current:
                windows.append(current)
                # carry a tail so windows overlap slightly across boundaries
                tail = current[-(self.window_chars // 4):]
                current = f"{tail} {sentence}".strip()
            else:
                current = f"{current} {sentence}".strip()
        if current:
            windows.append(current)
        return windows

    def rerank(self, chunks: List[str], tfidf_order: List[int], question: str) -> List[int]:
        model = self._get_model()
        if model is None:
            return tfidf_order

        shortlist = tfidf_order[: self.shortlist_size]
        window_map: List[tuple] = []  # (position in shortlist, window text)
        for pos, idx in enumerate(shortlist):
            for window in self._windows(chunks[idx]):
                window_map.append((pos, window))
        if not window_map:
            return tfidf_order

        try:
            import numpy as np

            embeddings = model.encode(
                [w for _, w in window_map] + [question], show_progress_bar=False
            )
            q = np.asarray(embeddings[-1])
            q_norm = np.linalg.norm(q) or 1.0
            matrix = np.asarray(embeddings[:-1])
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            sims = matrix @ q / (norms * q_norm)

            scores: dict = {}
            for (pos, _), sim in zip(window_map, sims):
                scores[pos] = max(scores.get(pos, -1.0), float(sim))

            dense_rank = sorted(range(len(shortlist)),
                                key=lambda j: scores.get(j, -1.0), reverse=True)
            dense_position = {j: p for p, j in enumerate(dense_rank)}

            # Reciprocal Rank Fusion: combine the TF-IDF order (the shortlist
            # prefix order) with the dense window score instead of replacing
            # it — each ranker's best guess survives the other's blind spots.
            fused = sorted(
                range(len(shortlist)),
                key=lambda j: 1.0 / (self.rrf_k + j) + 1.0 / (self.rrf_k + dense_position[j]),
                reverse=True,
            )
            reranked = [shortlist[j] for j in fused]
            reranked += [i for i in tfidf_order if i not in set(reranked)]
            return reranked
        except Exception as e:
            logger.warning(f"Re-ranking failed, using TF-IDF order: {e}")
            return tfidf_order


class _BaseRetriever:
    """Shared budget-filling logic: walk ranked chunks, keep document order."""

    def __init__(self, chunker: Chunker, ranker: TfidfRanker):
        self.chunker = chunker
        self.ranker = ranker

    def retrieve(self, document_text: str, question: str, budget_chars: int = 6000) -> str:
        chunks = self.chunker.chunk(document_text)
        if not chunks:
            return ""
        full_text = "\n\n".join(chunks)
        if len(full_text) <= budget_chars:
            return full_text

        ranked = self.ranker.rank(chunks, question)
        if hasattr(self, "reranker"):
            ranked = self.reranker.rerank(chunks, ranked, question)

        selected: List[int] = []
        budget = budget_chars
        for idx in ranked:
            size = len(chunks[idx])
            if size <= budget:
                selected.append(idx)
                budget -= size
            if budget <= 0:
                break
        if not selected:
            return chunks[ranked[0]][:budget_chars]
        selected.sort()
        return "\n\n".join(chunks[i] for i in selected)


class FixedChunkTFIDFRetriever(_BaseRetriever):
    """Baseline: fixed char windows ranked by TF-IDF (the original approach)."""

    def __init__(self, chunk_chars: int = 2000):
        super().__init__(FixedChunker(chunk_chars), TfidfRanker())


class SemanticChunkTFIDFRetriever(_BaseRetriever):
    """Improvement part 1: clause-aware chunks ranked by TF-IDF."""

    def __init__(self, max_chunk_chars: int = 2000):
        super().__init__(SemanticChunker(max_chunk_chars), TfidfRanker())


class RerankedRetriever(_BaseRetriever):
    """Improvement: clause-level semantic chunks (1000 chars) + TF-IDF shortlist
    re-ranked by sentence-window dense embeddings fused with Reciprocal Rank
    Fusion. Falls back to pure TF-IDF when the embedding model is unavailable.
    Measured on the 37-question eval set: hit rate 89.2% (baseline) -> 97.3%."""

    def __init__(self, max_chunk_chars: int = 1000, shortlist_size: int = 24):
        super().__init__(SemanticChunker(max_chunk_chars), TfidfRanker())
        self.reranker = EmbeddingReranker(shortlist_size)


def build_retriever(name: str) -> _BaseRetriever:
    """Build a retriever by name: 'baseline' | 'semantic' | 'reranked'."""
    retrievers = {
        "baseline": FixedChunkTFIDFRetriever,
        "semantic": SemanticChunkTFIDFRetriever,
        "reranked": RerankedRetriever,
    }
    if name not in retrievers:
        raise ValueError(f"Unknown retriever '{name}'. Choose from: {sorted(retrievers)}")
    return retrievers[name]()
