from pydantic import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    LLM_PROVIDER: str = "google"
    LLM_MODEL: str = "gemini-1.5-pro"
    LLM_TEMPERATURE: float = 0.1
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    POLICY_DOCUMENTS_PATH: str = "data/policies"
    EMBEDDINGS_PATH: str = "data/embeddings"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    TOP_K_RESULTS: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_CONCURRENT_REQUESTS: int = 10
    CACHE_TTL_SECONDS: int = 3600
    ENABLE_QUERY_CACHING: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()