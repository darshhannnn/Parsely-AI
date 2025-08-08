import pytest
from src.decision_engine.claim_evaluator import ClaimEvaluator
from src.semantic_search.semantic_retriever import SemanticRetriever
from src.query_parsing.query_parser import QueryParser

@pytest.fixture(scope="module")
def evaluator():
    retriever = SemanticRetriever()
    retriever.initialize_index()
    return ClaimEvaluator(retriever)

@pytest.fixture
def parser():
    return QueryParser()

def test_approved_claim(evaluator, parser):
    query = "46-year-old male, knee surgery in Pune, 12-month-old insurance policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "approved"
    assert decision.amount is not None
    assert "covered" in decision.justification.lower()

def test_rejected_waiting_period(evaluator, parser):
    query = "35-year-old female, maternity delivery in Mumbai, 3-month-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "rejected"
    assert "waiting period" in decision.justification.lower() or "not satisfied" in decision.justification.lower()

def test_exclusion_claim(evaluator, parser):
    query = "28-year-old male, cosmetic surgery in Delhi, 6-month-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "rejected" or "exclusion" in decision.justification.lower()

def test_sub_limit_claim(evaluator, parser):
    query = "40-year-old female, admitted for ICU, 2-year-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "approved"
    assert "sub-limit" in decision.justification.lower() or "icu" in decision.justification.lower()

def test_co_pay_claim(evaluator, parser):
    query = "65-year-old male, cataract surgery, 5-year-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "approved"
    assert "co-pay" in decision.justification.lower() or "copay" in decision.justification.lower()

def test_preexisting_waiting_period_claim(evaluator, parser):
    query = "50-year-old female, diabetes treatment, 1-year-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    # Should be rejected due to pre-existing waiting period
    assert decision.decision == "rejected"
    assert "pre-existing" in decision.justification.lower() or "waiting period" in decision.justification.lower()

def test_daycare_claim(evaluator, parser):
    query = "30-year-old male, daycare procedure, 3-year-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "approved"
    assert "daycare" in decision.justification.lower()

def test_permanent_exclusion_claim(evaluator, parser):
    query = "35-year-old female, experimental treatment, 4-year-old policy"
    parsed = parser.parse_query(query)
    decision = evaluator.evaluate_claim(parsed)
    assert decision.decision == "rejected"
    assert "permanent exclusion" in decision.justification.lower() or "experimental" in decision.justification.lower()
