"""
Caching system for document processing pipeline
"""

import json
import hashlib
from typing import Any, Optional, Dict
from datetime import timedelta

from .interfaces import ICacheManager
from .logging_utils import get_pipeline_logger
from .config import get_config

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class MemoryCacheManager(ICacheManager):
    """Simple in-memory cache manager"""
    
    def __init__(self):
        self._cache = {}
        self.logger = get_pipeline_logger()
        
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._cache[key] = value
        
    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            
    def clear(self) -> None:
        self._cache.clear()


class RedisCacheManager(ICacheManager):
    """Redis-based cache manager"""
    
    def __init__(self, redis_url: str):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis client not available. Install with: pip install redis")
            
        self.redis = redis.from_url(redis_url)
        self.logger = get_pipeline_logger()
        
    def get(self, key: str) -> Optional[Any]:
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        data = json.dumps(value)
        self.redis.set(key, data, ex=ttl)
        
    def delete(self, key: str) -> None:
        self.redis.delete(key)
        
    def clear(self) -> None:
        self.redis.flushdb()


def get_cache_manager() -> ICacheManager:
    """Factory function to get the configured cache manager"""
    config = get_config()
    
    if config.database.cache_type == "redis" and REDIS_AVAILABLE:
        try:
            return RedisCacheManager(config.database.redis_url)
        except Exception as e:
            get_pipeline_logger().warning(f"Failed to initialize Redis cache, falling back to memory: {e}")
            
    return MemoryCacheManager()
