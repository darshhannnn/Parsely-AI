"""
LLM Integration Layer with support for multiple providers
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator, retry_decorator, RateLimiter, CircuitBreaker
from ...core.exceptions import LLMProcessingError, LLMAPIError, LLMRateLimitError, LLMTimeoutError


class LLMProvider(Enum):
    """Supported LLM providers"""
    GOOGLE_GEMINI = "google_gemini"
    OPENAI_GPT = "openai_gpt"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    AZURE_OPENAI = "azure_openai"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    api_key: str
    model_name: str
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_requests_per_minute: int = 60
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class LLMRequest:
    """Request to LLM provider"""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    provider: LLMProvider
    model_name: str
    request_id: str
    response_time_ms: float
    token_usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMUsageStats:
    """Usage statistics for LLM provider"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    average_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    rate_limit_hits: int = 0
    circuit_breaker_trips: int = 0


class ILLMProvider(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM asynchronously"""
        pass
    
    @abstractmethod
    def validate_config(self, config: LLMConfig) -> bool:
        """Validate provider configuration"""
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> LLMUsageStats:
        """Get usage statistics"""
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for request"""
        pass


class GoogleGeminiProvider(ILLMProvider):
    """Google Gemini LLM provider"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = get_pipeline_logger()
        self.usage_stats = LLMUsageStats()
        
        # Initialize rate limiter and circuit breaker
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests_per_minute,
            window_seconds=60
        )
        
        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout
            )
        else:
            self.circuit_breaker = None
        
        # Initialize Gemini client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.client = genai.GenerativeModel(self.config.model_name)
            self.logger.info(f"Google Gemini client initialized with model {self.config.model_name}")
        except ImportError:
            raise LLMProcessingError("google-generativeai library not installed")
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize Google Gemini client: {e}", "google_gemini", self.config.model_name)
    
    def validate_config(self, config: LLMConfig) -> bool:
        """Validate Google Gemini configuration"""
        if not config.api_key:
            return False
        if not config.model_name:
            return False
        if config.provider != LLMProvider.GOOGLE_GEMINI:
            return False
        return True
    
    @timing_decorator
    @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Google Gemini"""
        start_time = time.time()
        
        # Check rate limit
        if not self.rate_limiter.is_allowed():
            self.usage_stats.rate_limit_hits += 1
            wait_time = self.rate_limiter.time_until_allowed()
            raise LLMRateLimitError("google_gemini", int(wait_time))
        
        # Check circuit breaker
        if self.circuit_breaker:
            try:
                response = self.circuit_breaker.call(self._make_request, request)
            except Exception as e:
                self.usage_stats.circuit_breaker_trips += 1
                raise
        else:
            response = self._make_request(request)
        
        # Update usage stats
        response_time_ms = (time.time() - start_time) * 1000
        self._update_usage_stats(request, response, response_time_ms, success=True)
        
        return response
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Generate response asynchronously"""
        # For now, run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)
    
    def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make actual request to Google Gemini"""
        try:
            # Prepare prompt
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
            
            # Configure generation parameters
            generation_config = {
                'max_output_tokens': request.max_tokens or self.config.max_tokens,
                'temperature': request.temperature or self.config.temperature,
            }
            
            if request.stop_sequences:
                generation_config['stop_sequences'] = request.stop_sequences
            
            # Generate response
            start_time = time.time()
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response content
            content = response.text if response.text else ""
            
            # Extract token usage (if available)
            token_usage = {
                'prompt_tokens': 0,  # Gemini doesn't provide detailed token counts
                'completion_tokens': 0,
                'total_tokens': 0
            }
            
            # Try to get usage metadata if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # Determine finish reason
            finish_reason = "completed"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GOOGLE_GEMINI,
                model_name=self.config.model_name,
                request_id=request.request_id,
                response_time_ms=response_time_ms,
                token_usage=token_usage,
                finish_reason=finish_reason,
                metadata={
                    'generation_config': generation_config,
                    'safety_ratings': getattr(response, 'safety_ratings', [])
                }
            )
            
        except Exception as e:
            self._update_usage_stats(request, None, 0, success=False)
            
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                raise LLMRateLimitError("google_gemini")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(self.config.timeout_seconds, "google_gemini")
            else:
                raise LLMAPIError(f"Google Gemini API error: {e}", "google_gemini", self.config.model_name)
    
    def _update_usage_stats(self, request: LLMRequest, response: Optional[LLMResponse], response_time_ms: float, success: bool):
        """Update usage statistics"""
        self.usage_stats.total_requests += 1
        self.usage_stats.last_request_time = datetime.now()
        
        if success and response:
            self.usage_stats.successful_requests += 1
            self.usage_stats.total_tokens_used += response.token_usage.get('total_tokens', 0)
            
            # Update average response time
            total_time = self.usage_stats.average_response_time_ms * (self.usage_stats.successful_requests - 1)
            self.usage_stats.average_response_time_ms = (total_time + response_time_ms) / self.usage_stats.successful_requests
            
            # Estimate cost (rough estimate for Gemini)
            estimated_cost = self.estimate_cost(request)
            self.usage_stats.total_cost_usd += estimated_cost
        else:
            self.usage_stats.failed_requests += 1
    
    def get_usage_stats(self) -> LLMUsageStats:
        """Get usage statistics"""
        return self.usage_stats
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Google Gemini request"""
        # Rough cost estimation for Gemini (as of 2024)
        # These are approximate rates and should be updated with actual pricing
        prompt_length = len(request.prompt)
        estimated_prompt_tokens = prompt_length // 4  # Rough estimate
        estimated_completion_tokens = (request.max_tokens or self.config.max_tokens) // 2
        
        # Gemini pricing (approximate)
        cost_per_1k_input_tokens = 0.00025  # $0.00025 per 1K input tokens
        cost_per_1k_output_tokens = 0.0005   # $0.0005 per 1K output tokens
        
        input_cost = (estimated_prompt_tokens / 1000) * cost_per_1k_input_tokens
        output_cost = (estimated_completion_tokens / 1000) * cost_per_1k_output_tokens
        
        return input_cost + output_cost


class OpenAIProvider(ILLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = get_pipeline_logger()
        self.usage_stats = LLMUsageStats()
        
        # Initialize rate limiter and circuit breaker
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests_per_minute,
            window_seconds=60
        )
        
        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout
            )
        else:
            self.circuit_breaker = None
        
        # Initialize OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key)
            self.logger.info(f"OpenAI client initialized with model {self.config.model_name}")
        except ImportError:
            raise LLMProcessingError("openai library not installed")
        except Exception as e:
            raise LLMAPIError(f"Failed to initialize OpenAI client: {e}", "openai", self.config.model_name)
    
    def validate_config(self, config: LLMConfig) -> bool:
        """Validate OpenAI configuration"""
        if not config.api_key:
            return False
        if not config.model_name:
            return False
        if config.provider != LLMProvider.OPENAI_GPT:
            return False
        return True
    
    @timing_decorator
    @retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI GPT"""
        start_time = time.time()
        
        # Check rate limit
        if not self.rate_limiter.is_allowed():
            self.usage_stats.rate_limit_hits += 1
            wait_time = self.rate_limiter.time_until_allowed()
            raise LLMRateLimitError("openai", int(wait_time))
        
        # Check circuit breaker
        if self.circuit_breaker:
            try:
                response = self.circuit_breaker.call(self._make_request, request)
            except Exception as e:
                self.usage_stats.circuit_breaker_trips += 1
                raise
        else:
            response = self._make_request(request)
        
        # Update usage stats
        response_time_ms = (time.time() - start_time) * 1000
        self._update_usage_stats(request, response, response_time_ms, success=True)
        
        return response
    
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """Generate response asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)
    
    def _make_request(self, request: LLMRequest) -> LLMResponse:
        """Make actual request to OpenAI"""
        try:
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Make request
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                stop=request.stop_sequences,
                timeout=self.config.timeout_seconds
            )
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response content
            content = response.choices[0].message.content if response.choices else ""
            
            # Extract token usage
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            }
            
            # Get finish reason
            finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI_GPT,
                model_name=self.config.model_name,
                request_id=request.request_id,
                response_time_ms=response_time_ms,
                token_usage=token_usage,
                finish_reason=finish_reason,
                metadata={
                    'model': response.model,
                    'created': response.created,
                    'id': response.id
                }
            )
            
        except Exception as e:
            self._update_usage_stats(request, None, 0, success=False)
            
            if "rate_limit" in str(e).lower():
                raise LLMRateLimitError("openai")
            elif "timeout" in str(e).lower():
                raise LLMTimeoutError(self.config.timeout_seconds, "openai")
            else:
                raise LLMAPIError(f"OpenAI API error: {e}", "openai", self.config.model_name)
    
    def _update_usage_stats(self, request: LLMRequest, response: Optional[LLMResponse], response_time_ms: float, success: bool):
        """Update usage statistics"""
        self.usage_stats.total_requests += 1
        self.usage_stats.last_request_time = datetime.now()
        
        if success and response:
            self.usage_stats.successful_requests += 1
            self.usage_stats.total_tokens_used += response.token_usage.get('total_tokens', 0)
            
            # Update average response time
            total_time = self.usage_stats.average_response_time_ms * (self.usage_stats.successful_requests - 1)
            self.usage_stats.average_response_time_ms = (total_time + response_time_ms) / self.usage_stats.successful_requests
            
            # Estimate cost
            estimated_cost = self.estimate_cost(request)
            self.usage_stats.total_cost_usd += estimated_cost
        else:
            self.usage_stats.failed_requests += 1
    
    def get_usage_stats(self) -> LLMUsageStats:
        """Get usage statistics"""
        return self.usage_stats
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for OpenAI request"""
        # OpenAI pricing (approximate, as of 2024)
        model_pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        
        # Get pricing for model (default to GPT-4 if not found)
        pricing = model_pricing.get(self.config.model_name, model_pricing['gpt-4'])
        
        # Estimate tokens
        prompt_length = len(request.prompt)
        if request.system_prompt:
            prompt_length += len(request.system_prompt)
        
        estimated_prompt_tokens = prompt_length // 4  # Rough estimate
        estimated_completion_tokens = (request.max_tokens or self.config.max_tokens) // 2
        
        input_cost = (estimated_prompt_tokens / 1000) * pricing['input']
        output_cost = (estimated_completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(config: LLMConfig) -> ILLMProvider:
        """Create LLM provider based on configuration"""
        
        if config.provider == LLMProvider.GOOGLE_GEMINI:
            return GoogleGeminiProvider(config)
        elif config.provider == LLMProvider.OPENAI_GPT:
            return OpenAIProvider(config)
        elif config.provider == LLMProvider.ANTHROPIC_CLAUDE:
            raise NotImplementedError("Anthropic Claude provider not yet implemented")
        elif config.provider == LLMProvider.AZURE_OPENAI:
            raise NotImplementedError("Azure OpenAI provider not yet implemented")
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


class LLMManager:
    """Manager for multiple LLM providers with fallback support"""
    
    def __init__(self, primary_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.logger = get_pipeline_logger()
        
        # Initialize primary provider
        self.primary_provider = LLMProviderFactory.create_provider(primary_config)
        self.primary_config = primary_config
        
        # Initialize fallback providers
        self.fallback_providers = []
        if fallback_configs:
            for config in fallback_configs:
                try:
                    provider = LLMProviderFactory.create_provider(config)
                    self.fallback_providers.append((provider, config))
                except Exception as e:
                    self.logger.warning(f"Failed to initialize fallback provider {config.provider}: {e}")
        
        self.logger.info(
            f"LLMManager initialized with primary provider {primary_config.provider} "
            f"and {len(self.fallback_providers)} fallback providers"
        )
    
    @timing_decorator
    def generate(self, request: LLMRequest, use_fallback: bool = True) -> LLMResponse:
        """Generate response with fallback support"""
        
        # Try primary provider first
        try:
            self.logger.debug(f"Attempting generation with primary provider {self.primary_config.provider}")
            return self.primary_provider.generate(request)
        
        except Exception as e:
            self.logger.warning(f"Primary provider {self.primary_config.provider} failed: {e}")
            
            if not use_fallback or not self.fallback_providers:
                raise
            
            # Try fallback providers
            for i, (provider, config) in enumerate(self.fallback_providers):
                try:
                    self.logger.info(f"Attempting generation with fallback provider {config.provider}")
                    response = provider.generate(request)
                    self.logger.info(f"Fallback provider {config.provider} succeeded")
                    return response
                
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback provider {config.provider} failed: {fallback_error}")
                    
                    # If this is the last fallback, raise the original error
                    if i == len(self.fallback_providers) - 1:
                        raise e
            
            # If we get here, all providers failed
            raise e
    
    async def generate_async(self, request: LLMRequest, use_fallback: bool = True) -> LLMResponse:
        """Generate response asynchronously with fallback support"""
        
        # Try primary provider first
        try:
            self.logger.debug(f"Attempting async generation with primary provider {self.primary_config.provider}")
            return await self.primary_provider.generate_async(request)
        
        except Exception as e:
            self.logger.warning(f"Primary provider {self.primary_config.provider} failed: {e}")
            
            if not use_fallback or not self.fallback_providers:
                raise
            
            # Try fallback providers
            for i, (provider, config) in enumerate(self.fallback_providers):
                try:
                    self.logger.info(f"Attempting async generation with fallback provider {config.provider}")
                    response = await provider.generate_async(request)
                    self.logger.info(f"Fallback provider {config.provider} succeeded")
                    return response
                
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback provider {config.provider} failed: {fallback_error}")
                    
                    # If this is the last fallback, raise the original error
                    if i == len(self.fallback_providers) - 1:
                        raise e
            
            # If we get here, all providers failed
            raise e
    
    def get_all_usage_stats(self) -> Dict[str, LLMUsageStats]:
        """Get usage statistics for all providers"""
        stats = {
            f"primary_{self.primary_config.provider.value}": self.primary_provider.get_usage_stats()
        }
        
        for i, (provider, config) in enumerate(self.fallback_providers):
            stats[f"fallback_{i}_{config.provider.value}"] = provider.get_usage_stats()
        
        return stats
    
    def get_total_cost(self) -> float:
        """Get total cost across all providers"""
        total_cost = self.primary_provider.get_usage_stats().total_cost_usd
        
        for provider, _ in self.fallback_providers:
            total_cost += provider.get_usage_stats().total_cost_usd
        
        return total_cost
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers"""
        status = {
            'primary_provider': {
                'provider': self.primary_config.provider.value,
                'model': self.primary_config.model_name,
                'healthy': True,  # Would implement actual health check
                'last_request': self.primary_provider.get_usage_stats().last_request_time
            },
            'fallback_providers': []
        }
        
        for i, (provider, config) in enumerate(self.fallback_providers):
            status['fallback_providers'].append({
                'provider': config.provider.value,
                'model': config.model_name,
                'healthy': True,  # Would implement actual health check
                'last_request': provider.get_usage_stats().last_request_time
            })
        
        return status