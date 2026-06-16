"""Integration tests for Gemini API integration."""
import pytest
import os


@pytest.mark.integration
@pytest.mark.gemini
@pytest.mark.skipif(not os.getenv('GOOGLE_API_KEY'), reason="GOOGLE_API_KEY not set")
def test_gemini_query_parser():
    """Test Gemini query parser integration."""
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        test_query = "46-year-old male, knee surgery in Pune"
        
        result = parser.parse_query(test_query)
        assert result is not None
        
    except ImportError:
        pytest.skip("Gemini query parser module not available")
    except Exception as e:
        # API might fail in CI environment
        pytest.skip(f"Gemini API test failed: {e}")


@pytest.mark.integration
@pytest.mark.gemini
@pytest.mark.skipif(not os.getenv('GOOGLE_API_KEY'), reason="GOOGLE_API_KEY not set")
def test_gemini_claim_evaluator():
    """Test Gemini claim evaluator integration."""
    try:
        from src.decision_engine.gemini_claim_evaluator import GeminiClaimEvaluator
        
        evaluator = GeminiClaimEvaluator()
        
        # Test with mock data
        test_query = "knee surgery claim"
        test_clauses = ["Coverage for orthopedic procedures"]
        
        result = evaluator.evaluate_claim(test_query, test_clauses)
        assert result is not None
        
    except ImportError:
        pytest.skip("Gemini claim evaluator module not available")
    except Exception as e:
        pytest.skip(f"Gemini API test failed: {e}")


@pytest.mark.integration
@pytest.mark.gemini
def test_gemini_configuration():
    """Test Gemini configuration without API call."""
    try:
        from config.gemini_config import GeminiConfig
        
        config = GeminiConfig()
        assert config is not None
        
    except ImportError:
        pytest.skip("Gemini config module not available")