import pytest
from src.semantic_search.semantic_retriever import SemanticRetriever
from src.query_parsing.query_parser import QueryParser

@pytest.fixture(scope="module")
def retriever():
    retriever = SemanticRetriever()
    retriever.initialize_index()
    return retriever

@pytest.fixture
def parser():
    return QueryParser()

def test_relevant_clauses_found(retriever, parser):
    query = "knee surgery in Pune, 3-month-old policy"
    parsed = parser.parse_query(query)
    results = retriever.search_relevant_clauses(parsed, top_k=5)
    assert len(results) > 0
    assert any("knee" in c.text.lower() or "surgery" in c.text.lower() for c in results)

def test_exclusion_search(retriever, parser):
    query = "maternity exclusion"
    parsed = parser.parse_query(query)
    results = retriever.search_relevant_clauses(parsed, top_k=5)
    assert len(results) > 0
    assert any("exclusion" in c.clause_type or "not covered" in c.text.lower() for c in results)
