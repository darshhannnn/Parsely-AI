"""
Main Document Processing Pipeline Orchestrator
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .core.interfaces import IDocumentProcessingPipeline, DocumentType
from .core.models import JSONResponse, ProcessingMetadata, ProcessingOptions, ErrorInfo
from .core.logging_utils import get_pipeline_logger
from .core.utils import timing_decorator, generate_correlation_id
from .stages import (
    DocumentDownloader,
    Stage2LLMParser,
    UnifiedSearchEngine,
    ClauseMatcher,
    DecisionEngine,
    ResponseFormatter
)
from .stages.stage2_llm_parser import LLMManager, LLMProviderFactory


class DocumentProcessingPipeline(IDocumentProcessingPipeline):
    """
    Main orchestrator for the 6-stage document processing pipeline.
    """
    
    def __init__(self, options: Optional[ProcessingOptions] = None):
        self.options = options or ProcessingOptions()
        self.logger = get_pipeline_logger()
        
        # Initialize components
        from .stages.stage2_llm_parser import LLMConfig, LLMProvider
        
        # Create default LLM config
        from .core.config import get_config
        app_config = get_config()
        
        llm_config = LLMConfig(
            provider=LLMProvider.GOOGLE_GEMINI,
            api_key=app_config.llm.google_api_key or "test-key",
            model_name=app_config.llm.model_name or "gemini-1.5-flash"
        )
        
        self.llm_manager = LLMManager(primary_config=llm_config)
        
        self.downloader = DocumentDownloader()
        self.llm_parser = Stage2LLMParser(self.llm_manager)
        self.search_engine = UnifiedSearchEngine()
        self.clause_matcher = ClauseMatcher(self.search_engine)
        self.decision_engine = DecisionEngine(self.llm_manager)
        self.formatter = ResponseFormatter()

    @timing_decorator
    def process_document(
        self, 
        document_url: str, 
        queries: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Process document through complete 6-stage pipeline"""
        
        correlation_id = generate_correlation_id()
        start_time = datetime.now()
        stage_durations = {}
        
        self.logger.info(f"Starting pipeline processing", correlation_id=correlation_id, url=document_url)
        
        try:
            # Stage 1: Input Documents
            s1_start = time.time()
            doc_content = self.downloader.download_document(document_url)
            extracted_content = self.downloader.extract_content(doc_content)
            stage_durations["Stage 1: Input Documents"] = (time.time() - s1_start) * 1000
            
            # Stage 2: LLM Parser
            s2_start = time.time()
            parsed_content = self.llm_parser.parse_content(extracted_content)
            stage_durations["Stage 2: LLM Parser"] = (time.time() - s2_start) * 1000
            
            # Stage 3: Embedding Search
            s3_start = time.time()
            embeddings = self.search_engine.create_embeddings(parsed_content.chunks)
            self.search_engine.build_index(embeddings)
            stage_durations["Stage 3: Embedding Search"] = (time.time() - s3_start) * 1000
            
            # Stage 4 & 5: Matching and Evaluation for each query
            results = []
            for query in queries:
                # Stage 4: Clause Matching
                s4_start = time.time()
                matches = self.clause_matcher.match_clauses(query, parsed_content.clauses)
                stage_durations["Stage 4: Clause Matching"] = (time.time() - s4_start) * 1000
                
                # Stage 5: Logic Evaluation
                s5_start = time.time()
                # Use matches as context if available, otherwise use chunks
                context = [m.clause for m in matches] if matches else parsed_content.chunks
                evaluation = self.decision_engine.evaluate_query(query, context)
                stage_durations["Stage 5: Logic Evaluation"] = (time.time() - s5_start) * 1000
                
                # Stage 6: JSON Output
                s6_start = time.time()
                formatted_result = self.formatter.format_response(evaluation)
                results.append(formatted_result.results[0])
                stage_durations["Stage 6: JSON Output"] = (time.time() - s6_start) * 1000
            
            end_time = datetime.now()
            metadata = ProcessingMetadata(
                correlation_id=correlation_id,
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=(end_time - start_time).total_seconds() * 1000,
                stage_durations=stage_durations,
                document_info={
                    "url": document_url,
                    "type": extracted_content.document_type,
                    "id": extracted_content.document_id
                }
            )
            
            return JSONResponse(
                success=True,
                processing_id=correlation_id,
                timestamp=end_time,
                document_info=metadata.document_info,
                results=results,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}", correlation_id=correlation_id)
            # Return error response
            return JSONResponse(
                success=False,
                processing_id=correlation_id,
                timestamp=datetime.now(),
                document_info={"url": document_url},
                results=[],
                metadata=ProcessingMetadata(correlation_id=correlation_id, start_time=start_time),
                error_message=str(e)
            )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health"""
        return {
            "status": "healthy",
            "stages": [
                "Input Documents",
                "LLM Parser",
                "Embedding Search",
                "Clause Matching",
                "Logic Evaluation",
                "JSON Output"
            ],
            "version": "2.0.0"
        }

    def get_supported_formats(self) -> List[DocumentType]:
        """Get list of supported document formats"""
        return [DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL]
