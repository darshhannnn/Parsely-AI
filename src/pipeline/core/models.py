"""
Core data models for the 6-stage document processing pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


@dataclass
class DocumentContent:
    """Raw document content with metadata"""
    url: str
    content_type: str
    raw_content: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.raw_content)


@dataclass
class ExtractedContent:
    """Extracted and processed document content"""
    document_id: str
    document_type: str
    text_content: str
    pages: Optional[List[str]] = None
    sections: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())


@dataclass
class ParsedContent:
    """LLM-parsed and structured content"""
    document_id: str
    structured_content: Dict[str, Any]
    identified_sections: List[Dict[str, Any]]
    document_summary: str
    key_topics: List[str]
    parsing_confidence: float
    parsing_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContentChunk:
    """Individual content chunk with embeddings"""
    id: str
    content: str
    document_id: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_type: str = "paragraph"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Clause:
    """Legal/policy clause with analysis"""
    id: str
    content: str
    clause_type: str
    obligations: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    document_id: str = ""
    page_number: Optional[int] = None
    section: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Embedding:
    """Vector embedding with metadata"""
    chunk_id: str
    vector: List[float]
    model_name: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.vector)


@dataclass
class VectorIndex:
    """Vector index for similarity search"""
    index_id: str
    index_type: str  # "faiss" or "pinecone"
    dimension: int
    total_vectors: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.index_id:
            self.index_id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """Search result with similarity score"""
    chunk_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class ClauseMatch:
    """Clause matching result"""
    clause: Clause
    similarity_score: float
    match_type: str  # "exact", "semantic", "partial"
    explanation: str = ""
    rank: int = 0


@dataclass
class Evidence:
    """Evidence supporting a conclusion"""
    source_chunk: ContentChunk
    relevance_score: float
    evidence_type: str  # "direct", "supporting", "contextual"
    explanation: str = ""


@dataclass
class Evaluation:
    """Logic evaluation result"""
    query: str
    answer: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    evaluation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Explanation:
    """Detailed explanation for an evaluation"""
    reasoning_chain: List[str]
    evidence_summary: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]
    alternative_interpretations: List[str]
    limitations: List[str]
    sources_cited: List[str]


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline"""
    correlation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    stage_durations: Dict[str, float] = field(default_factory=dict)
    document_info: Dict[str, Any] = field(default_factory=dict)
    pipeline_version: str = "2.0.0"
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class JSONResponse:
    """Structured JSON response"""
    success: bool
    processing_id: str
    timestamp: datetime
    document_info: Dict[str, Any]
    results: List[Dict[str, Any]]
    metadata: ProcessingMetadata
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.processing_id:
            self.processing_id = str(uuid.uuid4())


@dataclass
class ErrorInfo:
    """Error information with context"""
    code: str
    message: str
    stage: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = ""
    retry_possible: bool = False


@dataclass
class ProcessingOptions:
    """Options for document processing"""
    include_explanations: bool = True
    max_chunks: int = 100
    similarity_threshold: float = 0.1
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_processing_time_seconds: int = 300
    vector_db_type: str = "faiss"  # "faiss" or "pinecone"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-1.5-flash"
    chunk_size: int = 500
    chunk_overlap: int = 100


@dataclass
class SystemHealth:
    """System health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, bool]
    last_check: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    version: str = "2.0.0"
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    requests_per_minute: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)