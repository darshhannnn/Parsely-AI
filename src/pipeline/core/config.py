"""
Configuration management for the document processing pipeline
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # Vector Database
    vector_db_type: str = "faiss"  # "faiss" or "pinecone"
    faiss_index_path: str = "./data/faiss_indexes"
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "document-processing"
    
    # Cache Database
    cache_type: str = "memory"  # "memory", "redis"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600


@dataclass
class LLMConfig:
    """LLM configuration"""
    # Primary LLM
    primary_provider: str = "google"  # "google", "openai", "anthropic"
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Model settings
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 30
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_day: int = 1000


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "all-MiniLM-L6-v2"
    model_cache_dir: str = "./models/embeddings"
    batch_size: int = 32
    max_sequence_length: int = 512
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    # Document limits
    max_document_size_mb: int = 50
    max_pages: int = 1000
    supported_formats: list = field(default_factory=lambda: ["pdf", "docx", "email"])
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    
    # Search settings
    default_top_k: int = 5
    similarity_threshold: float = 0.1
    max_search_results: int = 20
    
    # Processing timeouts
    download_timeout_seconds: int = 30
    processing_timeout_seconds: int = 300
    
    # Temporary storage
    temp_dir: str = "./temp"
    cleanup_temp_files: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    # API Security
    api_key_required: bool = True
    api_key: str = ""
    allowed_origins: list = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_requests_per_hour: int = 1000
    
    # Data privacy
    log_document_content: bool = False
    data_retention_days: int = 30
    anonymize_logs: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "./logs/pipeline.log"
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Health checks
    health_check_interval_seconds: int = 30
    
    # Alerting
    enable_alerting: bool = False
    alert_webhook_url: str = ""


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Component configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Pipeline settings
    pipeline_version: str = "2.0.0"
    enable_caching: bool = True
    enable_async_processing: bool = True
    max_concurrent_requests: int = 10
    
    # Environment
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = False


class ConfigManager:
    """Configuration manager with environment variable support"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config: Optional[PipelineConfig] = None
    
    def load_config(self) -> PipelineConfig:
        """Load configuration from environment variables and config file"""
        if self._config is None:
            self._config = self._create_config_from_env()
        return self._config
    
    def _create_config_from_env(self) -> PipelineConfig:
        """Create configuration from environment variables"""
        
        # Database config
        database_config = DatabaseConfig(
            vector_db_type=os.getenv("VECTOR_DB_TYPE", "faiss"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_indexes"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "document-processing"),
            cache_type=os.getenv("CACHE_TYPE", "memory"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        )
        
        # LLM config
        llm_config = LLMConfig(
            primary_provider=os.getenv("LLM_PROVIDER", "google"),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model_name=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT", "30")),
            requests_per_minute=int(os.getenv("LLM_RPM", "60")),
            requests_per_day=int(os.getenv("LLM_RPD", "1000"))
        )
        
        # Embedding config
        embedding_config = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            model_cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "./models/embeddings"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_sequence_length=int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
            device=os.getenv("EMBEDDING_DEVICE", "cpu")
        )
        
        # Processing config
        processing_config = ProcessingConfig(
            max_document_size_mb=int(os.getenv("MAX_DOC_SIZE_MB", "50")),
            max_pages=int(os.getenv("MAX_PAGES", "1000")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "50")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.1")),
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "20")),
            download_timeout_seconds=int(os.getenv("DOWNLOAD_TIMEOUT", "30")),
            processing_timeout_seconds=int(os.getenv("PROCESSING_TIMEOUT", "300")),
            temp_dir=os.getenv("TEMP_DIR", "./temp"),
            cleanup_temp_files=os.getenv("CLEANUP_TEMP", "true").lower() == "true"
        )
        
        # Security config
        security_config = SecurityConfig(
            api_key_required=os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
            api_key=os.getenv("API_KEY", ""),
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            rate_limit_requests_per_hour=int(os.getenv("RATE_LIMIT_RPH", "1000")),
            log_document_content=os.getenv("LOG_CONTENT", "false").lower() == "true",
            data_retention_days=int(os.getenv("DATA_RETENTION_DAYS", "30")),
            anonymize_logs=os.getenv("ANONYMIZE_LOGS", "true").lower() == "true"
        )
        
        # Monitoring config
        monitoring_config = MonitoringConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            log_file=os.getenv("LOG_FILE", "./logs/pipeline.log"),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            health_check_interval_seconds=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            enable_alerting=os.getenv("ENABLE_ALERTING", "false").lower() == "true",
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", "")
        )
        
        # Main pipeline config
        config = PipelineConfig(
            database=database_config,
            llm=llm_config,
            embedding=embedding_config,
            processing=processing_config,
            security=security_config,
            monitoring=monitoring_config,
            pipeline_version=os.getenv("PIPELINE_VERSION", "2.0.0"),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            enable_async_processing=os.getenv("ENABLE_ASYNC", "true").lower() == "true",
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT", "10")),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
        
        return config
    
    def validate_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        issues = []
        warnings = []
        
        # Check required API keys
        if config.llm.primary_provider == "google" and not config.llm.google_api_key:
            issues.append("Google API key is required when using Google as LLM provider")
        
        if config.llm.primary_provider == "openai" and not config.llm.openai_api_key:
            issues.append("OpenAI API key is required when using OpenAI as LLM provider")
        
        if config.database.vector_db_type == "pinecone" and not config.database.pinecone_api_key:
            issues.append("Pinecone API key is required when using Pinecone as vector database")
        
        # Check directories
        temp_dir = Path(config.processing.temp_dir)
        if not temp_dir.exists():
            try:
                temp_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create temp directory: {e}")
        
        # Check resource limits
        if config.processing.max_document_size_mb > 100:
            warnings.append("Large document size limit may cause memory issues")
        
        if config.processing.processing_timeout_seconds < 60:
            warnings.append("Short processing timeout may cause failures for large documents")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def get_config_summary(self, config: PipelineConfig) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            "pipeline_version": config.pipeline_version,
            "environment": config.environment,
            "llm_provider": config.llm.primary_provider,
            "llm_model": config.llm.model_name,
            "vector_db": config.database.vector_db_type,
            "embedding_model": config.embedding.model_name,
            "caching_enabled": config.enable_caching,
            "async_processing": config.enable_async_processing,
            "max_concurrent": config.max_concurrent_requests,
            "debug_mode": config.debug
        }


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> PipelineConfig:
    """Get the current pipeline configuration"""
    return config_manager.load_config()


def setup_logging(config: PipelineConfig) -> None:
    """Setup logging based on configuration"""
    log_level = getattr(logging, config.monitoring.log_level.upper())
    
    # Create logs directory if it doesn't exist
    log_file_path = Path(config.monitoring.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.monitoring.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set third-party library log levels
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)