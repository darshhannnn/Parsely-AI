"""
Tests for LLM Integration Layer
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.pipeline.stages.stage2_llm_parser.llm_integration import (
    LLMProvider,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageStats,
    GoogleGeminiProvider,
    OpenAIProvider,
    LLMProviderFactory,
    LLMManager,
    ILLMProvider
)
from src.pipeline.core.exceptions import (
    LLMProcessingError,
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError
)


class TestLLMConfig:
    """Test LLM configuration"""
    
    def test_llm_config_creation(self):
        """Test LLM config creation with defaults"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-key",
            model_name="gemini-pro"
        )
        
        assert config.provider == LLMProvider.GOOGLE_GEMINI
        assert config.api_key == "test-key"
        assert config.model_name == "gemini-pro"
        assert config.max_tokens == 4000
        assert config.temperature == 0.1
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.rate_limit_requests_per_minute == 60
        assert config.enable_circuit_breaker is True
        assert config.additional_params == {}
    
    def test_llm_config_with_custom_params(self):
        """Test LLM config with custom parameters"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI_GPT,
            api_key="test-key",
            model_name="gpt-4",
            max_tokens=2000,
            temperature=0.5,
            timeout_seconds=60,
            additional_params={"top_p": 0.9}
        )
        
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
        assert config.timeout_seconds == 60
        assert config.additional_params == {"top_p": 0.9}


class TestLLMRequest:
    """Test LLM request structure"""
    
    def test_llm_request_creation(self):
        """Test LLM request creation"""
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="System instructions",
            max_tokens=1000,
            temperature=0.3
        )
        
        assert request.prompt == "Test prompt"
        assert request.system_prompt == "System instructions"
        assert request.max_tokens == 1000
        assert request.temperature == 0.3
        assert request.metadata == {}
        assert request.request_id is not None
    
    def test_llm_request_with_metadata(self):
        """Test LLM request with metadata"""
        metadata = {"user_id": "123", "session_id": "abc"}
        request = LLMRequest(
            prompt="Test prompt",
            metadata=metadata
        )
        
        assert request.metadata == metadata


class TestLLMResponse:
    """Test LLM response structure"""
    
    def test_llm_response_creation(self):
        """Test LLM response creation"""
        response = LLMResponse(
            content="Test response",
            provider=LLMProvider.GOOGLE_GEMINI,
            model_name="gemini-pro",
            request_id="test-123",
            response_time_ms=500.0,
            token_usage={"total_tokens": 100},
            finish_reason="completed"
        )
        
        assert response.content == "Test response"
        assert response.provider == LLMProvider.GOOGLE_GEMINI
        assert response.model_name == "gemini-pro"
        assert response.request_id == "test-123"
        assert response.response_time_ms == 500.0
        assert response.token_usage == {"total_tokens": 100}
        assert response.finish_reason == "completed"
        assert isinstance(response.timestamp, datetime)


class TestGoogleGeminiProvider:
    """Test Google Gemini provider"""
    
    @pytest.fixture
    def gemini_config(self):
        """Create Gemini config for testing"""
        return LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-gemini-key",
            model_name="gemini-pro",
            rate_limit_requests_per_minute=10,
            enable_circuit_breaker=False  # Disable for testing
        )
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Mock Gemini client"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            # Mock response
            mock_response = Mock()
            mock_response.text = "Test response from Gemini"
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 20
            mock_response.usage_metadata.total_token_count = 30
            mock_response.safety_ratings = []
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model.return_value = mock_model_instance
            
            yield mock_model_instance
    
    def test_gemini_provider_initialization(self, gemini_config, mock_gemini_client):
        """Test Gemini provider initialization"""
        provider = GoogleGeminiProvider(gemini_config)
        
        assert provider.config == gemini_config
        assert provider.usage_stats.total_requests == 0
        assert provider.rate_limiter is not None
        assert provider.circuit_breaker is None  # Disabled in config
    
    def test_gemini_config_validation(self, gemini_config):
        """Test Gemini config validation"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            
            provider = GoogleGeminiProvider(gemini_config)
            
            # Valid config
            assert provider.validate_config(gemini_config) is True
            
            # Invalid configs
            invalid_config = LLMConfig(
                provider=LLMProvider.GOOGLE_GEMINI,
                api_key="",
                model_name="gemini-pro"
            )
            assert provider.validate_config(invalid_config) is False
            
            invalid_config.api_key = "test-key"
            invalid_config.model_name = ""
            assert provider.validate_config(invalid_config) is False
            
            invalid_config.model_name = "gemini-pro"
            invalid_config.provider = LLMProvider.OPENAI_GPT
            assert provider.validate_config(invalid_config) is False
    
    def test_gemini_generate_success(self, gemini_config, mock_gemini_client):
        """Test successful Gemini generation"""
        provider = GoogleGeminiProvider(gemini_config)
        
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="System prompt",
            max_tokens=1000,
            temperature=0.5
        )
        
        response = provider.generate(request)
        
        assert response.content == "Test response from Gemini"
        assert response.provider == LLMProvider.GOOGLE_GEMINI
        assert response.model_name == "gemini-pro"
        assert response.request_id == request.request_id
        assert response.token_usage["total_tokens"] == 30
        assert response.finish_reason == "STOP"
        
        # Check usage stats
        stats = provider.get_usage_stats()
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_tokens_used == 30
    
    def test_gemini_generate_with_rate_limit(self, gemini_config, mock_gemini_client):
        """Test Gemini generation with rate limiting"""
        # Set very low rate limit
        gemini_config.rate_limit_requests_per_minute = 1
        provider = GoogleGeminiProvider(gemini_config)
        
        request = LLMRequest(prompt="Test prompt")
        
        # First request should succeed
        response1 = provider.generate(request)
        assert response1.content == "Test response from Gemini"
        
        # Second request should hit rate limit
        with pytest.raises(LLMRateLimitError):
            provider.generate(request)
        
        # Check rate limit stats
        stats = provider.get_usage_stats()
        assert stats.rate_limit_hits == 1
    
    def test_gemini_api_error_handling(self, gemini_config):
        """Test Gemini API error handling"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.side_effect = Exception("API Error")
            mock_model.return_value = mock_model_instance
            
            provider = GoogleGeminiProvider(gemini_config)
            request = LLMRequest(prompt="Test prompt")
            
            with pytest.raises(LLMAPIError) as exc_info:
                provider.generate(request)
            
            assert "Google Gemini API error" in str(exc_info.value)
            
            # Check failure stats
            stats = provider.get_usage_stats()
            assert stats.failed_requests == 1
    
    def test_gemini_rate_limit_error(self, gemini_config):
        """Test Gemini rate limit error handling"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.side_effect = Exception("quota exceeded")
            mock_model.return_value = mock_model_instance
            
            provider = GoogleGeminiProvider(gemini_config)
            request = LLMRequest(prompt="Test prompt")
            
            with pytest.raises(LLMRateLimitError):
                provider.generate(request)
    
    def test_gemini_timeout_error(self, gemini_config):
        """Test Gemini timeout error handling"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.side_effect = Exception("timeout")
            mock_model.return_value = mock_model_instance
            
            provider = GoogleGeminiProvider(gemini_config)
            request = LLMRequest(prompt="Test prompt")
            
            with pytest.raises(LLMTimeoutError):
                provider.generate(request)
    
    @pytest.mark.asyncio
    async def test_gemini_generate_async(self, gemini_config, mock_gemini_client):
        """Test async Gemini generation"""
        provider = GoogleGeminiProvider(gemini_config)
        request = LLMRequest(prompt="Test prompt")
        
        response = await provider.generate_async(request)
        
        assert response.content == "Test response from Gemini"
        assert response.provider == LLMProvider.GOOGLE_GEMINI
    
    def test_gemini_cost_estimation(self, gemini_config, mock_gemini_client):
        """Test Gemini cost estimation"""
        provider = GoogleGeminiProvider(gemini_config)
        request = LLMRequest(prompt="Test prompt", max_tokens=1000)
        
        cost = provider.estimate_cost(request)
        
        assert isinstance(cost, float)
        assert cost > 0


class TestOpenAIProvider:
    """Test OpenAI provider"""
    
    @pytest.fixture
    def openai_config(self):
        """Create OpenAI config for testing"""
        return LLMConfig(
            provider=LLMProvider.OPENAI_GPT,
            api_key="test-openai-key",
            model_name="gpt-4",
            rate_limit_requests_per_minute=10,
            enable_circuit_breaker=False
        )
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            
            # Mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response from OpenAI"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_response.model = "gpt-4"
            mock_response.created = int(time.time())
            mock_response.id = "test-response-id"
            
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            yield mock_client
    
    def test_openai_provider_initialization(self, openai_config, mock_openai_client):
        """Test OpenAI provider initialization"""
        provider = OpenAIProvider(openai_config)
        
        assert provider.config == openai_config
        assert provider.usage_stats.total_requests == 0
    
    def test_openai_generate_success(self, openai_config, mock_openai_client):
        """Test successful OpenAI generation"""
        provider = OpenAIProvider(openai_config)
        
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="System prompt",
            max_tokens=1000
        )
        
        response = provider.generate(request)
        
        assert response.content == "Test response from OpenAI"
        assert response.provider == LLMProvider.OPENAI_GPT
        assert response.model_name == "gpt-4"
        assert response.token_usage["total_tokens"] == 30
        assert response.finish_reason == "stop"
        
        # Verify the API call was made correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["max_tokens"] == 1000
        assert len(call_args[1]["messages"]) == 2  # system + user message
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"
    
    def test_openai_cost_estimation(self, openai_config, mock_openai_client):
        """Test OpenAI cost estimation"""
        provider = OpenAIProvider(openai_config)
        request = LLMRequest(prompt="Test prompt", max_tokens=1000)
        
        cost = provider.estimate_cost(request)
        
        assert isinstance(cost, float)
        assert cost > 0


class TestLLMProviderFactory:
    """Test LLM provider factory"""
    
    def test_create_gemini_provider(self):
        """Test creating Gemini provider"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-key",
            model_name="gemini-pro"
        )
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            
            provider = LLMProviderFactory.create_provider(config)
            assert isinstance(provider, GoogleGeminiProvider)
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI_GPT,
            api_key="test-key",
            model_name="gpt-4"
        )
        
        with patch('openai.OpenAI'):
            provider = LLMProviderFactory.create_provider(config)
            assert isinstance(provider, OpenAIProvider)
    
    def test_unsupported_provider(self):
        """Test creating unsupported provider"""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC_CLAUDE,
            api_key="test-key",
            model_name="claude-3"
        )
        
        with pytest.raises(NotImplementedError):
            LLMProviderFactory.create_provider(config)


class TestLLMManager:
    """Test LLM manager with fallback support"""
    
    @pytest.fixture
    def primary_config(self):
        """Primary provider config"""
        return LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-gemini-key",
            model_name="gemini-pro",
            enable_circuit_breaker=False
        )
    
    @pytest.fixture
    def fallback_config(self):
        """Fallback provider config"""
        return LLMConfig(
            provider=LLMProvider.OPENAI_GPT,
            api_key="test-openai-key",
            model_name="gpt-4",
            enable_circuit_breaker=False
        )
    
    def test_llm_manager_initialization(self, primary_config, fallback_config):
        """Test LLM manager initialization"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'), \
             patch('openai.OpenAI'):
            
            manager = LLMManager(
                primary_config=primary_config,
                fallback_configs=[fallback_config]
            )
            
            assert isinstance(manager.primary_provider, GoogleGeminiProvider)
            assert len(manager.fallback_providers) == 1
            assert isinstance(manager.fallback_providers[0][0], OpenAIProvider)
    
    def test_llm_manager_primary_success(self, primary_config):
        """Test LLM manager with successful primary provider"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            # Mock successful response
            mock_response = Mock()
            mock_response.text = "Primary response"
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.total_token_count = 30
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model.return_value = mock_model_instance
            
            manager = LLMManager(primary_config=primary_config)
            request = LLMRequest(prompt="Test prompt")
            
            response = manager.generate(request)
            
            assert response.content == "Primary response"
            assert response.provider == LLMProvider.GOOGLE_GEMINI
    
    def test_llm_manager_fallback_success(self, primary_config, fallback_config):
        """Test LLM manager with fallback when primary fails"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_gemini, \
             patch('openai.OpenAI') as mock_openai:
            
            # Mock primary provider failure
            mock_gemini_instance = Mock()
            mock_gemini_instance.generate_content.side_effect = Exception("Primary failed")
            mock_gemini.return_value = mock_gemini_instance
            
            # Mock fallback provider success
            mock_openai_client = Mock()
            mock_openai_response = Mock()
            mock_openai_response.choices = [Mock()]
            mock_openai_response.choices[0].message.content = "Fallback response"
            mock_openai_response.choices[0].finish_reason = "stop"
            mock_openai_response.usage = Mock()
            mock_openai_response.usage.total_tokens = 25
            mock_openai_response.model = "gpt-4"
            mock_openai_response.created = int(time.time())
            mock_openai_response.id = "fallback-id"
            
            mock_openai_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_openai_client
            
            manager = LLMManager(
                primary_config=primary_config,
                fallback_configs=[fallback_config]
            )
            
            request = LLMRequest(prompt="Test prompt")
            response = manager.generate(request)
            
            assert response.content == "Fallback response"
            assert response.provider == LLMProvider.OPENAI_GPT
    
    def test_llm_manager_all_providers_fail(self, primary_config, fallback_config):
        """Test LLM manager when all providers fail"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_gemini, \
             patch('openai.OpenAI') as mock_openai:
            
            # Mock both providers failing
            mock_gemini_instance = Mock()
            mock_gemini_instance.generate_content.side_effect = Exception("Primary failed")
            mock_gemini.return_value = mock_gemini_instance
            
            mock_openai_client = Mock()
            mock_openai_client.chat.completions.create.side_effect = Exception("Fallback failed")
            mock_openai.return_value = mock_openai_client
            
            manager = LLMManager(
                primary_config=primary_config,
                fallback_configs=[fallback_config]
            )
            
            request = LLMRequest(prompt="Test prompt")
            
            with pytest.raises(Exception) as exc_info:
                manager.generate(request)
            
            assert "Primary failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_llm_manager_async_generation(self, primary_config):
        """Test async generation with LLM manager"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            # Mock successful response
            mock_response = Mock()
            mock_response.text = "Async response"
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.total_token_count = 30
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model.return_value = mock_model_instance
            
            manager = LLMManager(primary_config=primary_config)
            request = LLMRequest(prompt="Test prompt")
            
            response = await manager.generate_async(request)
            
            assert response.content == "Async response"
    
    def test_llm_manager_usage_stats(self, primary_config, fallback_config):
        """Test getting usage stats from all providers"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'), \
             patch('openai.OpenAI'):
            
            manager = LLMManager(
                primary_config=primary_config,
                fallback_configs=[fallback_config]
            )
            
            stats = manager.get_all_usage_stats()
            
            assert "primary_google_gemini" in stats
            assert "fallback_0_openai_gpt" in stats
            assert isinstance(stats["primary_google_gemini"], LLMUsageStats)
    
    def test_llm_manager_health_status(self, primary_config):
        """Test getting health status"""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            
            manager = LLMManager(primary_config=primary_config)
            status = manager.get_health_status()
            
            assert "primary_provider" in status
            assert status["primary_provider"]["provider"] == "google_gemini"
            assert status["primary_provider"]["model"] == "gemini-pro"
            assert "fallback_providers" in status


class TestLLMIntegrationEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_missing_api_key(self):
        """Test behavior with missing API key"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="",
            model_name="gemini-pro"
        )
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel'):
            
            provider = GoogleGeminiProvider(config)
            assert provider.validate_config(config) is False
    
    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-key",
            model_name="gemini-pro",
            enable_circuit_breaker=False
        )
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            # Mock empty response
            mock_response = Mock()
            mock_response.text = ""
            mock_response.candidates = []
            mock_response.usage_metadata = None
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model.return_value = mock_model_instance
            
            provider = GoogleGeminiProvider(config)
            request = LLMRequest(prompt="Test prompt")
            
            response = provider.generate(request)
            
            assert response.content == ""
            assert response.finish_reason == "completed"
    
    def test_large_prompt_handling(self):
        """Test handling of very large prompts"""
        config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key="test-key",
            model_name="gemini-pro",
            enable_circuit_breaker=False
        )
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model:
            
            mock_response = Mock()
            mock_response.text = "Response to large prompt"
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.total_token_count = 1000
            
            mock_model_instance = Mock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model.return_value = mock_model_instance
            
            provider = GoogleGeminiProvider(config)
            
            # Create a very large prompt
            large_prompt = "This is a test prompt. " * 1000
            request = LLMRequest(prompt=large_prompt)
            
            response = provider.generate(request)
            
            assert response.content == "Response to large prompt"
            assert response.token_usage["total_tokens"] == 1000


if __name__ == "__main__":
    pytest.main([__file__])