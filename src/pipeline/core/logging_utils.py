"""
Logging utilities for the document processing pipeline
"""

import logging
import json
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from .utils import sanitize_for_logging, generate_correlation_id


class StructuredLogger:
    """Structured logger with correlation ID support"""
    
    def __init__(self, name: str, correlation_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id or generate_correlation_id()
    
    def _log_structured(self, level: int, message: str, **kwargs):
        """Log structured message with metadata"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": self.correlation_id,
            "message": message,
            "level": logging.getLevelName(level),
            **sanitize_for_logging(kwargs)
        }
        
        # Log as JSON for structured logging
        self.logger.log(level, json.dumps(log_data))
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_structured(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_structured(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_structured(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_structured(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_structured(logging.CRITICAL, message, **kwargs)
    
    def log_stage_start(self, stage: str, **kwargs):
        """Log pipeline stage start"""
        self.info(f"Stage {stage} started", stage=stage, event="stage_start", **kwargs)
    
    def log_stage_complete(self, stage: str, duration_ms: float, **kwargs):
        """Log pipeline stage completion"""
        self.info(
            f"Stage {stage} completed", 
            stage=stage, 
            event="stage_complete", 
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_stage_error(self, stage: str, error: Exception, **kwargs):
        """Log pipeline stage error"""
        self.error(
            f"Stage {stage} failed", 
            stage=stage, 
            event="stage_error", 
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )
    
    def log_processing_start(self, document_url: str, queries: list, **kwargs):
        """Log document processing start"""
        self.info(
            "Document processing started",
            event="processing_start",
            document_url=document_url,
            query_count=len(queries),
            **kwargs
        )
    
    def log_processing_complete(self, duration_ms: float, success: bool, **kwargs):
        """Log document processing completion"""
        self.info(
            "Document processing completed",
            event="processing_complete",
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )


class PipelineLoggerFactory:
    """Factory for creating pipeline loggers"""
    
    @staticmethod
    def get_logger(name: str, correlation_id: Optional[str] = None) -> StructuredLogger:
        """Get structured logger instance"""
        return StructuredLogger(name, correlation_id)
    
    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: str = "json"
    ):
        """Setup logging configuration"""
        
        # Create logs directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        if log_format == "json":
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Set third-party library log levels
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# Global logger instances
def get_pipeline_logger(correlation_id: Optional[str] = None) -> StructuredLogger:
    """Get pipeline logger with correlation ID"""
    return PipelineLoggerFactory.get_logger("pipeline", correlation_id)


def get_stage_logger(stage: str, correlation_id: Optional[str] = None) -> StructuredLogger:
    """Get stage-specific logger"""
    return PipelineLoggerFactory.get_logger(f"pipeline.{stage}", correlation_id)


def get_logger(name: str) -> logging.Logger:
    """Get standard logger instance"""
    return logging.getLogger(name)