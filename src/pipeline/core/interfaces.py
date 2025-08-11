"""
Core interfaces for the 6-stage document processing pipeline
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import datetime

from .models import (
    DocumentContent, ExtractedContent, ParsedContent, ContentChunk,
    Clause, Embedding, VectorIndex, SearchResult, ClauseMatch,
    Evaluation, Explanation, JSONResponse, ProcessingMetadata
)


class DocumentType(Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"


class ChunkType(Enum):
    """Types of content chunks"""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    CLAUSE = "clause"
    HEADER = "header"
    FOOTER = "footer"
    TABLE = "table"


class ClauseType(Enum):
    """Types of clauses in legal/policy documents"""
    TERMS = "terms"
    CONDITIONS = "conditions"
    OBLIGATIONS = "obligations"
    RIGHTS = "rights"
    DEFINITIONS = "definitions"
    PENALTIES = "penalties"


# Stage 1: Input Documents Interface
class IDocumentProcessor(ABC):
    """Interface for Stage 1: Input Documents processing"""
    
    @abstractmethod
    def download_document(self, url: str) -> DocumentContent:
        """Download document from URL with validation"""
        pass
    
    @abstractmethod
    def detect_format(self, content: bytes) -> DocumentType:
        """Detect document format from content"""
        pass
    
    @abstractmethod
    def extract_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content based on document type"""
        pass
    
    @abstractmethod
    def validate_document(self, content: ExtractedContent) -> bool:
        """Validate extracted content"""
        pass


# Stage 2: LLM Parser Interface
class ILLMParser(ABC):
    """Interface for Stage 2: LLM Parser"""
    
    @abstractmethod
    def parse_content(self, content: ExtractedContent) -> ParsedContent:
        """Parse content using LLM for structure understanding"""
        pass
    
    @abstractmethod
    def create_chunks(self, parsed_content: ParsedContent) -> List[ContentChunk]:
        """Create semantic chunks from parsed content"""
        pass
    
    @abstractmethod
    def extract_structure(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract document structure (headings, sections, etc.)"""
        pass
    
    @abstractmethod
    def identify_clauses(self, content: ExtractedContent) -> List[Clause]:
        """Identify clauses in legal/policy documents"""
        pass


# Stage 3: Embedding Search Interface
class IEmbeddingSearchEngine(ABC):
    """Interface for Stage 3: Embedding Search"""
    
    @abstractmethod
    def create_embeddings(self, chunks: List[ContentChunk]) -> List[Embedding]:
        """Create vector embeddings for content chunks"""
        pass
    
    @abstractmethod
    def build_index(self, embeddings: List[Embedding]) -> VectorIndex:
        """Build vector index for similarity search"""
        pass
    
    @abstractmethod
    def search_similar(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar content using semantic similarity"""
        pass
    
    @abstractmethod
    def update_index(self, new_embeddings: List[Embedding]) -> None:
        """Update existing index with new embeddings"""
        pass


# Stage 4: Clause Matching Interface
class IClauseMatcher(ABC):
    """Interface for Stage 4: Clause Matching"""
    
    @abstractmethod
    def match_clauses(self, query: str, clauses: List[Clause]) -> List[ClauseMatch]:
        """Match query against clauses using semantic similarity"""
        pass
    
    @abstractmethod
    def find_related_clauses(self, clause: Clause) -> List[Clause]:
        """Find clauses related to the given clause"""
        pass
    
    @abstractmethod
    def categorize_clauses(self, clauses: List[Clause]) -> Dict[ClauseType, List[Clause]]:
        """Categorize clauses by type"""
        pass
    
    @abstractmethod
    def extract_obligations(self, clauses: List[Clause]) -> List[str]:
        """Extract obligations from clauses"""
        pass


# Stage 5: Logic Evaluation Interface
class ILogicEvaluator(ABC):
    """Interface for Stage 5: Logic Evaluation"""
    
    @abstractmethod
    def evaluate_query(self, query: str, context: List[ContentChunk]) -> Evaluation:
        """Evaluate query against context with reasoning"""
        pass
    
    @abstractmethod
    def generate_explanation(self, evaluation: Evaluation) -> Explanation:
        """Generate detailed explanation for the evaluation"""
        pass
    
    @abstractmethod
    def resolve_conflicts(self, conflicting_info: List[ContentChunk]) -> Dict[str, Any]:
        """Resolve conflicts in information"""
        pass
    
    @abstractmethod
    def calculate_confidence(self, evidence: List[ContentChunk]) -> float:
        """Calculate confidence score based on evidence"""
        pass


# Stage 6: JSON Output Interface
class IResponseFormatter(ABC):
    """Interface for Stage 6: JSON Output"""
    
    @abstractmethod
    def format_response(self, evaluation: Evaluation) -> JSONResponse:
        """Format evaluation into structured JSON response"""
        pass
    
    @abstractmethod
    def include_metadata(self, response: JSONResponse, metadata: ProcessingMetadata) -> JSONResponse:
        """Include processing metadata in response"""
        pass
    
    @abstractmethod
    def validate_schema(self, response: JSONResponse) -> bool:
        """Validate response against JSON schema"""
        pass
    
    @abstractmethod
    def optimize_response(self, response: JSONResponse) -> JSONResponse:
        """Optimize response for size and readability"""
        pass


# Main Pipeline Interface
class IDocumentProcessingPipeline(ABC):
    """Main interface for the complete 6-stage pipeline"""
    
    @abstractmethod
    def process_document(
        self, 
        document_url: str, 
        queries: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Process document through complete 6-stage pipeline"""
        pass
    
    @abstractmethod
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[DocumentType]:
        """Get list of supported document formats"""
        pass


# Cache Interface
class ICacheManager(ABC):
    """Interface for caching system"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete cached value"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values"""
        pass


# Monitoring Interface
class IMonitoringService(ABC):
    """Interface for monitoring and observability"""
    
    @abstractmethod
    def log_processing_start(self, document_url: str, queries: List[str]) -> str:
        """Log processing start and return correlation ID"""
        pass
    
    @abstractmethod
    def log_stage_completion(self, correlation_id: str, stage: str, duration: float) -> None:
        """Log completion of pipeline stage"""
        pass
    
    @abstractmethod
    def log_error(self, correlation_id: str, error: Exception, stage: str) -> None:
        """Log error with context"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        pass