"""Pytest configuration and shared fixtures."""
import os
import pytest
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def mock_gemini_api():
    """Mock Gemini API for testing."""
    with patch('google.generativeai.configure') as mock_configure:
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_instance = Mock()
            mock_instance.generate_content.return_value.text = "Mock response"
            mock_model.return_value = mock_instance
            yield mock_instance


@pytest.fixture
def sample_policy_text():
    """Sample policy text for testing."""
    return """
    HEALTH INSURANCE POLICY
    
    Section 1: Coverage
    This policy covers medical expenses for the insured person.
    
    Section 2: Orthopedic Procedures
    Knee surgeries and joint replacements are covered up to ₹200,000.
    
    Section 3: Waiting Period
    Pre-existing conditions have a 2-year waiting period.
    """


@pytest.fixture
def sample_claim_query():
    """Sample claim query for testing."""
    return "46-year-old male, knee surgery in Pune, 3-month-old policy"


@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    env_vars = {
        'GOOGLE_API_KEY': 'test_api_key',
        'LLM_PROVIDER': 'google',
        'LLM_MODEL': 'gemini-1.5-pro',
        'LLM_TEMPERATURE': '0.1'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "data" / "policies"
    data_dir.mkdir(parents=True)
    
    # Create sample policy file
    policy_file = data_dir / "sample_policy.txt"
    policy_file.write_text("""
    Sample Health Insurance Policy
    Coverage: Medical expenses up to ₹500,000
    Orthopedic procedures: Covered
    """)
    
    return data_dir