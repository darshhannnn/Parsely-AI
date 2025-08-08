import pytest
from src.query_parsing.query_parser import QueryParser

@pytest.fixture
def parser():
    return QueryParser()

def test_basic_entity_extraction(parser):
    query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    result = parser.parse_query(query)
    assert result.age == 46
    assert result.gender == "male"
    assert result.procedure == "knee surgery"
    assert result.location == "Pune"
    assert result.policy_age_months == 3
    assert "Age not specified" not in result.assumptions

def test_missing_info(parser):
    query = "surgery in Mumbai"
    result = parser.parse_query(query)
    assert result.age is None
    assert result.gender is None
    assert result.procedure == "knee surgery" or result.procedure is None
    assert result.location == "Mumbai"
    assert any("Age not specified" in a for a in result.assumptions)

def test_amount_extraction(parser):
    query = "Patient claims Rs 50,000 for accident treatment in Delhi"
    result = parser.parse_query(query)
    assert result.amount_claimed == 50000
    assert result.procedure == "accident"
    assert result.location == "Delhi"

def test_date_extraction(parser):
    query = "Surgery performed on 12/06/2025 in Chennai"
    result = parser.parse_query(query)
    assert result.date == "12/06/2025"
    assert result.location == "Chennai"
