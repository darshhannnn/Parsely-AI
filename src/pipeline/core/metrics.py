"""
Metrics and monitoring for the document processing pipeline
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsService:
    """Service for tracking system metrics"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsService, cls).__new__(cls)
                cls._instance._init_metrics()
            return cls._instance
            
    def _init_metrics(self):
        """Initialize Prometheus metrics if available"""
        if not PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = {}
            return
            
        self.prometheus_metrics = {
            'processing_total': Counter(
                'doc_processing_total', 
                'Total number of documents processed',
                ['status', 'document_type']
            ),
            'processing_duration': Histogram(
                'doc_processing_duration_seconds',
                'Time spent processing documents',
                ['stage']
            ),
            'llm_token_usage': Counter(
                'llm_token_usage_total',
                'Total tokens used by LLM providers',
                ['provider', 'model', 'type']
            ),
            'cache_hits': Counter(
                'cache_hits_total',
                'Total number of cache hits/misses',
                ['type', 'result']
            ),
            'active_requests': Gauge(
                'active_processing_requests',
                'Number of currently active processing requests'
            )
        }
        
    def record_processing_start(self):
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['active_requests'].inc()
            
    def record_processing_complete(self, status: str, doc_type: str):
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['active_requests'].dec()
            self.prometheus_metrics['processing_total'].labels(status=status, document_type=doc_type).inc()
            
    def record_stage_duration(self, stage: str, duration_seconds: float):
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['processing_duration'].labels(stage=stage).observe(duration_seconds)
            
    def record_llm_usage(self, provider: str, model: str, token_type: str, count: int):
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics['llm_token_usage'].labels(
                provider=provider, model=model, type=token_type
            ).inc(count)


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance"""
    return MetricsService()
