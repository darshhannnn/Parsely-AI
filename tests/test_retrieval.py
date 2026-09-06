"""
Tests for the context retrieval package (src/retrieval) and the API
context builder that uses it.
"""

import pytest
from unittest.mock import patch

from src.retrieval import (
    FixedChunker,
    SemanticChunker,
    TfidfRanker,
    EmbeddingReranker,
    FixedChunkTFIDFRetriever,
    SemanticChunkTFIDFRetriever,
    RerankedRetriever,
    build_retriever,
    normalize_ws,
)


LONG_POLICY = "\n\n".join(
    [
        f"Section {i}: The insured shall comply with all administrative requirements "
        "of the plan and submit any documents the insurer may reasonably request."
        for i in range(1, 30)
    ]
    + [
        "Section 40: Waiting Period. All claims arising from pre-existing diseases are "
        "covered only after the expiry of 36 months of continuous coverage from the "
        "date of inception of the first policy with the company."
    ]
)


class TestNormalizeWs:
    def test_collapses_whitespace(self):
        assert normalize_ws("a  \n\t  b") == "a b"

    def test_strips_edges(self):
        assert normalize_ws("  hello  ") == "hello"


class TestFixedChunker:
    def test_short_text_single_chunk(self):
        assert FixedChunker(2000).chunk("short text") == ["short text"]

    def test_long_text_respects_size(self):
        chunks = FixedChunker(1000).chunk("word " * 5000)
        assert len(chunks) > 1
        assert all(len(c) <= 1000 for c in chunks)

    def test_windows_overlap(self):
        text = "a" * 3000
        chunks = FixedChunker(1000).chunk(text)
        assert chunks[1].startswith(chunks[0][-100:])


class TestSemanticChunker:
    def test_respects_max_size(self):
        chunks = SemanticChunker(1000).chunk(LONG_POLICY)
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks).count("Section 40") >= 1

    def test_keeps_clause_whole(self):
        doc = (
            "1. Waiting Period. Expenses related to pre-existing diseases are excluded "
            "until the expiry of 36 months of continuous coverage after inception. "
            "The waiting period is waived for accident claims entirely.\n\n"
            + "filler text without structure " * 120
        )
        chunks = SemanticChunker(1000).chunk(doc)
        clause_chunks = [c for c in chunks if "Waiting Period" in c]
        assert clause_chunks
        assert any("36 months" in c and "accident claims" in c for c in clause_chunks)

    def test_empty_text(self):
        assert SemanticChunker(1000).chunk("") == []


class TestTfidfRanker:
    def test_relevant_chunk_ranks_first(self):
        chunks = [
            "The quick brown fox jumps over the lazy dog.",
            "Grace Period means the time to renew the policy premium without penalty.",
            "Totally unrelated content about submarine sandwiches and pickles.",
        ]
        ranked = TfidfRanker().rank(chunks, "What is the grace period for renewing the policy?")
        assert ranked[0] == 1

    def test_single_chunk(self):
        assert TfidfRanker().rank(["only one"], "question") == [0]

    def test_empty(self):
        assert TfidfRanker().rank([], "question") == []


class TestRerankerFallback:
    def test_falls_back_when_model_unavailable(self):
        EmbeddingReranker._load_failed = True
        EmbeddingReranker._model = None
        try:
            reranker = EmbeddingReranker()
            tfidf_order = [2, 0, 1]
            assert reranker.rerank(["a", "b", "c"], tfidf_order, "question") == tfidf_order
        finally:
            EmbeddingReranker._load_failed = False


class TestRetrievers:
    def test_full_doc_returned_when_under_budget(self):
        retriever = SemanticChunkTFIDFRetriever()
        ctx = retriever.retrieve(LONG_POLICY, "anything", budget_chars=10 ** 9)
        assert "36 months of continuous coverage" in ctx

    def test_budget_respected_when_over(self):
        retriever = SemanticChunkTFIDFRetriever(1000)
        ctx = retriever.retrieve(LONG_POLICY, "What is the waiting period for pre-existing diseases?",
                                 budget_chars=3000)
        assert len(ctx) <= 3000 + 2000  # one oversize chunk may exceed slightly
        assert "36 months" in ctx

    def test_reranked_retriever_without_model_beats_random(self):
        with patch.object(EmbeddingReranker, "_load_failed", True), \
             patch.object(EmbeddingReranker, "_model", None):
            retriever = RerankedRetriever(1000)
            ctx = retriever.retrieve(
                LONG_POLICY, "What is the waiting period for pre-existing diseases?",
                budget_chars=3000,
            )
        assert "36 months" in ctx

    def test_baseline_retriever(self):
        retriever = FixedChunkTFIDFRetriever(1000)
        ctx = retriever.retrieve(LONG_POLICY, "waiting period pre-existing diseases",
                                 budget_chars=3000)
        assert "36 months" in ctx

    def test_build_retriever_unknown_name(self):
        with pytest.raises(ValueError):
            build_retriever("nonexistent")

    def test_build_retriever_known_names(self):
        assert isinstance(build_retriever("baseline"), FixedChunkTFIDFRetriever)
        assert isinstance(build_retriever("semantic"), SemanticChunkTFIDFRetriever)
        assert isinstance(build_retriever("reranked"), RerankedRetriever)


class TestApiContextBuilder:
    def test_build_question_context_uses_retrieval(self):
        from src.api.hackathon_main import _build_question_context

        ctx = _build_question_context(
            LONG_POLICY, "What is the waiting period for pre-existing diseases?",
            max_chars=3000,
        )
        assert "36 months" in ctx
