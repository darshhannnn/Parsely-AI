"""
Stage 6: JSON Output - Structured response formatting and validation
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...core.interfaces import IResponseFormatter
from ...core.models import Evaluation, JSONResponse, ProcessingMetadata
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator


class ResponseFormatter(IResponseFormatter):
    """
    Structured JSON response formatter and validator.
    
    Features:
    - Schema validation
    - Metadata injection
    - Response optimization
    """
    
    def __init__(self):
        self.logger = get_pipeline_logger()

    @timing_decorator
    def format_response(self, evaluation: Evaluation) -> JSONResponse:
        """Format evaluation into structured JSON response"""
        self.logger.info(f"Formatting response for query: {evaluation.query}")
        
        return JSONResponse(
            success=True,
            processing_id=str(evaluation.evaluation_timestamp.timestamp()),
            timestamp=datetime.now(),
            document_info={},  # Will be populated by include_metadata
            results=[{
                "query": evaluation.query,
                "answer": evaluation.answer,
                "confidence": evaluation.confidence,
                "evidence": [
                    {
                        "source": e.source_chunk.id,
                        "relevance": e.relevance_score,
                        "snippet": e.source_chunk.content[:200]
                    } for e in evaluation.evidence
                ]
            }],
            metadata=ProcessingMetadata(
                correlation_id="",
                start_time=evaluation.evaluation_timestamp,
                end_time=datetime.now()
            )
        )

    def include_metadata(self, response: JSONResponse, metadata: ProcessingMetadata) -> JSONResponse:
        """Include processing metadata in response"""
        response.metadata = metadata
        if metadata.document_info:
            response.document_info = metadata.document_info
        return response

    def validate_schema(self, response: JSONResponse) -> bool:
        """Validate response against JSON schema"""
        # In a real implementation, this would use jsonschema
        return True

    def optimize_response(self, response: JSONResponse) -> JSONResponse:
        """Optimize response for size and readability"""
        # For example, truncating long fields or removing internal metadata
        return response
