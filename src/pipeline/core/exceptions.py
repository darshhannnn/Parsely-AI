"""
Custom exceptions for the document processing pipeline
"""

from typing import Optional, Dict, Any


class PipelineException(Exception):
    """Base exception for pipeline errors"""
    
    def __init__(
        self, 
        message: str, 
        stage: str = "", 
        error_code: str = "", 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.error_code = error_code
        self.details = details or {}


class DocumentProcessingError(PipelineException):
    """Stage 1: Document processing errors"""
    
    def __init__(self, message: str, error_code: str = "DOCUMENT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "input_documents", error_code, details)


class DocumentDownloadError(DocumentProcessingError):
    """Document download specific errors"""
    
    def __init__(self, message: str, url: str = "", status_code: Optional[int] = None):
        details = {"url": url}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "DOWNLOAD_ERROR", details)


class UnsupportedFormatError(DocumentProcessingError):
    """Unsupported document format errors"""
    
    def __init__(self, format_type: str, supported_formats: list):
        message = f"Unsupported document format: {format_type}. Supported formats: {', '.join(supported_formats)}"
        details = {"format": format_type, "supported_formats": supported_formats}
        super().__init__(message, "UNSUPPORTED_FORMAT", details)


class DocumentSizeError(DocumentProcessingError):
    """Document size limit errors"""
    
    def __init__(self, size_mb: float, max_size_mb: float):
        message = f"Document size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        details = {"size_mb": size_mb, "max_size_mb": max_size_mb}
        super().__init__(message, "SIZE_LIMIT_EXCEEDED", details)


class ContentExtractionError(DocumentProcessingError):
    """Content extraction errors"""
    
    def __init__(self, message: str, document_type: str = ""):
        details = {"document_type": document_type}
        super().__init__(message, "EXTRACTION_ERROR", details)


class LLMProcessingError(PipelineException):
    """Stage 2: LLM processing errors"""
    
    def __init__(self, message: str, error_code: str = "LLM_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "llm_parser", error_code, details)


class LLMAPIError(LLMProcessingError):
    """LLM API specific errors"""
    
    def __init__(self, message: str, provider: str = "", model: str = "", status_code: Optional[int] = None):
        details = {"provider": provider, "model": model}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "LLM_API_ERROR", details)


class LLMRateLimitError(LLMProcessingError):
    """LLM rate limit errors"""
    
    def __init__(self, provider: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {provider}"
        details = {"provider": provider}
        if retry_after:
            details["retry_after_seconds"] = retry_after
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class LLMTimeoutError(LLMProcessingError):
    """LLM timeout errors"""
    
    def __init__(self, timeout_seconds: int, provider: str = ""):
        message = f"LLM request timed out after {timeout_seconds} seconds"
        details = {"timeout_seconds": timeout_seconds, "provider": provider}
        super().__init__(message, "LLM_TIMEOUT", details)


class ChunkingError(LLMProcessingError):
    """Content chunking errors"""
    
    def __init__(self, message: str, document_id: str = ""):
        details = {"document_id": document_id}
        super().__init__(message, "CHUNKING_ERROR", details)


class EmbeddingError(PipelineException):
    """Stage 3: Embedding processing errors"""
    
    def __init__(self, message: str, error_code: str = "EMBEDDING_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "embedding_search", error_code, details)


class EmbeddingGenerationError(EmbeddingError):
    """Embedding generation specific errors"""
    
    def __init__(self, message: str, model_name: str = "", content_length: int = 0):
        details = {"model_name": model_name, "content_length": content_length}
        super().__init__(message, "EMBEDDING_GENERATION_ERROR", details)


class EmbeddingModelError(EmbeddingError):
    """Embedding model errors"""
    
    def __init__(self, message: str, model_name: str = ""):
        details = {"model_name": model_name}
        super().__init__(message, "EMBEDDING_MODEL_ERROR", details)


class VectorIndexError(EmbeddingError):
    """Vector index errors"""
    
    def __init__(self, message: str, index_type: str = "", operation: str = ""):
        details = {"index_type": index_type, "operation": operation}
        super().__init__(message, "VECTOR_INDEX_ERROR", details)


class FAISSError(VectorIndexError):
    """FAISS specific errors"""
    
    def __init__(self, message: str, operation: str = ""):
        super().__init__(message, "faiss", operation)


class PineconeError(VectorIndexError):
    """Pinecone specific errors"""
    
    def __init__(self, message: str, operation: str = "", status_code: Optional[int] = None):
        details = {"operation": operation}
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, "pinecone", operation)


class SearchError(EmbeddingError):
    """Search operation errors"""
    
    def __init__(self, message: str, query: str = "", top_k: int = 0):
        details = {"query": query[:100], "top_k": top_k}  # Truncate query for logging
        super().__init__(message, "SEARCH_ERROR", details)


class ClauseMatchingError(PipelineException):
    """Stage 4: Clause matching errors"""
    
    def __init__(self, message: str, error_code: str = "CLAUSE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "clause_matching", error_code, details)


class ClauseExtractionError(ClauseMatchingError):
    """Clause extraction errors"""
    
    def __init__(self, message: str, document_id: str = ""):
        details = {"document_id": document_id}
        super().__init__(message, "CLAUSE_EXTRACTION_ERROR", details)


class ClauseCategorizationError(ClauseMatchingError):
    """Clause categorization errors"""
    
    def __init__(self, message: str, clause_count: int = 0):
        details = {"clause_count": clause_count}
        super().__init__(message, "CLAUSE_CATEGORIZATION_ERROR", details)


class LogicEvaluationError(PipelineException):
    """Stage 5: Logic evaluation errors"""
    
    def __init__(self, message: str, error_code: str = "LOGIC_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "logic_evaluation", error_code, details)


class ReasoningError(LogicEvaluationError):
    """Reasoning process errors"""
    
    def __init__(self, message: str, query: str = "", context_chunks: int = 0):
        details = {"query": query[:100], "context_chunks": context_chunks}
        super().__init__(message, "REASONING_ERROR", details)


class ConflictResolutionError(LogicEvaluationError):
    """Conflict resolution errors"""
    
    def __init__(self, message: str, conflict_count: int = 0):
        details = {"conflict_count": conflict_count}
        super().__init__(message, "CONFLICT_RESOLUTION_ERROR", details)


class ConfidenceCalculationError(LogicEvaluationError):
    """Confidence calculation errors"""
    
    def __init__(self, message: str, evidence_count: int = 0):
        details = {"evidence_count": evidence_count}
        super().__init__(message, "CONFIDENCE_ERROR", details)


class ResponseFormattingError(PipelineException):
    """Stage 6: Response formatting errors"""
    
    def __init__(self, message: str, error_code: str = "RESPONSE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "json_output", error_code, details)


class JSONSchemaError(ResponseFormattingError):
    """JSON schema validation errors"""
    
    def __init__(self, message: str, schema_errors: list = None):
        details = {"schema_errors": schema_errors or []}
        super().__init__(message, "SCHEMA_VALIDATION_ERROR", details)


class ResponseOptimizationError(ResponseFormattingError):
    """Response optimization errors"""
    
    def __init__(self, message: str, response_size: int = 0):
        details = {"response_size_bytes": response_size}
        super().__init__(message, "OPTIMIZATION_ERROR", details)


class CacheError(PipelineException):
    """Cache operation errors"""
    
    def __init__(self, message: str, error_code: str = "CACHE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "cache", error_code, details)


class CacheConnectionError(CacheError):
    """Cache connection errors"""
    
    def __init__(self, message: str, cache_type: str = ""):
        details = {"cache_type": cache_type}
        super().__init__(message, "CACHE_CONNECTION_ERROR", details)


class ConfigurationError(PipelineException):
    """Configuration errors"""
    
    def __init__(self, message: str, config_key: str = "", details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["config_key"] = config_key
        super().__init__(message, "configuration", "CONFIG_ERROR", details)


class AuthenticationError(PipelineException):
    """Authentication errors"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "authentication", "AUTH_ERROR")


class AuthorizationError(PipelineException):
    """Authorization errors"""
    
    def __init__(self, message: str = "Authorization failed"):
        super().__init__(message, "authorization", "AUTHZ_ERROR")


class RateLimitError(PipelineException):
    """Rate limiting errors"""
    
    def __init__(self, message: str, limit: int = 0, window: str = ""):
        details = {"limit": limit, "window": window}
        super().__init__(message, "rate_limiting", "RATE_LIMIT_ERROR", details)


class TimeoutError(PipelineException):
    """Timeout errors"""
    
    def __init__(self, message: str, timeout_seconds: int = 0, operation: str = ""):
        details = {"timeout_seconds": timeout_seconds, "operation": operation}
        super().__init__(message, "timeout", "TIMEOUT_ERROR", details)


class ValidationError(PipelineException):
    """Input validation errors"""
    
    def __init__(self, message: str, field: str = "", value: Any = None):
        details = {"field": field, "value": str(value) if value is not None else None}
        super().__init__(message, "validation", "VALIDATION_ERROR", details)