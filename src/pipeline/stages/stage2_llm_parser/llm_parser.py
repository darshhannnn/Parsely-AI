"""
Stage 2: LLM Parser - Intelligent content parsing and structuring
"""

import time
from typing import List, Dict, Any, Optional
import asyncio

from ...core.interfaces import ILLMParser, DocumentType
from ...core.models import ExtractedContent, ParsedContent, ContentChunk, Clause, ProcessingMetadata
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator, generate_correlation_id
from .llm_integration import LLMManager, LLMRequest
from .content_chunker import IntelligentContentChunker, ChunkingConfig
from .clause_identifier import ClauseStructureIdentifier, IdentifiedClause, DocumentStructure
from .prompt_templates import PromptTemplateManager, PromptType
from .response_parser import LLMResponseParser, ResponseFormat


class Stage2LLMParser(ILLMParser):
    """
    Orchestrates the LLM parsing stage of the pipeline.
    
    Responsibilities:
    1. Document structure analysis
    2. Semantic content chunking
    3. Legal/Policy clause identification
    4. Relationship mapping between clauses
    """
    
    def __init__(self, llm_manager: LLMManager, chunking_config: Optional[ChunkingConfig] = None):
        self.llm_manager = llm_manager
        self.logger = get_pipeline_logger()
        self.chunker = IntelligentContentChunker(chunking_config)
        self.clause_identifier = ClauseStructureIdentifier(llm_manager)
        self.prompt_manager = PromptTemplateManager()
        self.response_parser = LLMResponseParser()

    @timing_decorator
    def parse_content(self, content: ExtractedContent) -> ParsedContent:
        """Parse content using LLM for structure understanding"""
        self.logger.info(f"Starting LLM parsing for document {content.document_id}")
        start_time = time.time()
        
        # Step 1: Analyze document structure
        structure_info = self.extract_structure(content)
        
        # Step 2: Identify clauses
        clauses = self.identify_clauses(content)
        
        # Step 3: Create semantic chunks
        chunks = self.create_chunks(content)
        
        processing_time = (time.time() - start_time) * 1000
        
        metadata = ProcessingMetadata(
            stage="Stage 2: LLM Parser",
            duration_ms=processing_time,
            success=True,
            additional_info={
                "clause_count": len(clauses),
                "chunk_count": len(chunks),
                "structure_elements": len(structure_info.get('sections', []))
            }
        )
        
        return ParsedContent(
            document_id=content.document_id,
            structured_data=structure_info,
            chunks=chunks,
            clauses=clauses,
            metadata=metadata
        )

    def create_chunks(self, content: ExtractedContent) -> List[ContentChunk]:
        """Create semantic chunks from content"""
        # For simple cases, we use the IntelligentContentChunker
        # In advanced cases, we could use LLM to refine chunks
        return self.chunker.chunk_content(content)

    def extract_structure(self, content: ExtractedContent) -> Dict[str, Any]:
        """Extract document structure (headings, sections, etc.) using LLM and pattern matching"""
        
        # Use ClauseStructureIdentifier for initial structure extraction
        doc_structure = self.clause_identifier.identify_clauses_and_structure(
            content.text_content, content.document_id
        )
        
        return {
            "title": doc_structure.title,
            "sections": [
                {"title": s.title, "level": s.level, "start_index": s.start_index} 
                for s in doc_structure.sections
            ],
            "metadata": doc_structure.metadata
        }

    def identify_clauses(self, content: ExtractedContent) -> List[Clause]:
        """Identify clauses in legal/policy documents"""
        
        # Use ClauseStructureIdentifier for clause identification
        doc_structure = self.clause_identifier.identify_clauses_and_structure(
            content.text_content, content.document_id
        )
        
        # Convert IdentifiedClause to core Clause model
        clauses = []
        for id_clause in doc_structure.clauses:
            clause = Clause(
                clause_id=id_clause.clause_id,
                text=id_clause.text,
                clause_type=id_clause.clause_type.value,
                section_title=id_clause.metadata.get('section_title', ''),
                metadata=id_clause.metadata
            )
            clauses.append(clause)
            
        return clauses

    async def parse_content_async(self, content: ExtractedContent) -> ParsedContent:
        """Asynchronous version of parse_content"""
        # In a real implementation, this would use async LLM calls
        return await asyncio.to_thread(self.parse_content, content)
