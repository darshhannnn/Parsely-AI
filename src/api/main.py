"""
Main FastAPI Application for Document Processing Pipeline
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..pipeline.pipeline_orchestrator import DocumentProcessingPipeline
from ..pipeline.core.models import ProcessingOptions, JSONResponse as PipelineResponse

logger = logging.getLogger(__name__)

# Optional rate limiting - gracefully handle if slowapi is not available
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    logger.warning("slowapi not available - rate limiting disabled")
    RATE_LIMITING_AVAILABLE = False

app = FastAPI(
    title="Parsely AI - Document Processing API",
    description="Advanced 6-stage document analysis pipeline powered by LLMs",
    version="2.0.0"
)

if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (could be moved to a dependency)
pipeline = DocumentProcessingPipeline()

class ProcessRequest(BaseModel):
    """Request model for document processing"""
    url: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer based on the document")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional processing options")

@app.get("/health")
def health_check():
    """Service health check"""
    return pipeline.get_pipeline_status()

@app.post("/process", response_model=PipelineResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document through the 6-stage pipeline.
    """
    try:
        response = pipeline.process_document(
            document_url=request.url,
            queries=request.questions,
            options=request.options
        )
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error_message or "Unknown processing error"
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/formats")
def get_formats():
    """Get list of supported document formats"""
    return {"supported_formats": [f.value for f in pipeline.get_supported_formats()]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
