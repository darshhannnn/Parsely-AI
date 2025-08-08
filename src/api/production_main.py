"""
Production FastAPI application with security, monitoring, and performance optimizations
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import your components
from ..insurance_claim_processor import InsuranceClaimProcessor

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global processor
    
    # Startup
    logger.info("Starting LLM Document Processing System...")
    try:
        processor = InsuranceClaimProcessor()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Document Processing System...")

# Create FastAPI app
app = FastAPI(
    title="Parsely AI API",
    description="Production-ready intelligent document processing platform - Fresh take on document analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for your domain in production
)

# Request models
class ClaimRequest(BaseModel):
    query: str
    return_json: Optional[bool] = True

class DocumentUploadRequest(BaseModel):
    content: str
    filename: str
    document_type: str

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    expected_token = os.getenv("API_KEY_SECRET", "your_production_api_secret_here")
    
    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    start_time = time.time()
    
    # Add rate limiting logic here if needed
    # For production, consider using Redis-based rate limiting
    
    response = await call_next(request)
    
    # Add processing time header
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Health check endpoint (no auth required)
@app.get("/health", tags=["System"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "llm_provider": "google_gemini",
        "model": os.getenv("LLM_MODEL", "gemini-1.5-flash")
    }

# Metrics endpoint (no auth required)
@app.get("/metrics", tags=["System"])
async def get_metrics():
    """System metrics for monitoring"""
    return {
        "requests_processed": 0,  # Implement counter
        "average_response_time": 0,  # Implement tracking
        "error_rate": 0,  # Implement tracking
        "system_status": "operational"
    }

# Main API endpoints (with authentication)
@app.post("/process_claim", tags=["Claims"], dependencies=[Depends(verify_token)])
async def process_claim(request: ClaimRequest):
    """Process an insurance claim query"""
    try:
        logger.info(f"Processing claim: {request.query[:50]}...")
        
        result = processor.process_claim(request.query, return_json=request.return_json)
        
        logger.info("Claim processed successfully")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Claim processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/analyze_query", tags=["Analysis"], dependencies=[Depends(verify_token)])
async def analyze_query(query: str):
    """Analyze and extract entities from a query"""
    try:
        logger.info(f"Analyzing query: {query[:50]}...")
        
        result = processor.analyze_query_only(query)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/search_clauses", tags=["Search"], dependencies=[Depends(verify_token)])
async def search_clauses(query: str, top_k: int = 5):
    """Semantic search for relevant policy clauses"""
    try:
        if not 1 <= top_k <= 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 20"
            )
        
        logger.info(f"Searching clauses for: {query[:50]}...")
        
        result = processor.search_clauses_only(query, top_k=top_k)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Clause search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/policy_summary", tags=["System"], dependencies=[Depends(verify_token)])
async def policy_summary():
    """Get summary of loaded policies and clause counts"""
    try:
        result = processor.get_policy_summary()
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Policy summary failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary failed: {str(e)}"
        )

@app.post("/rebuild_index", tags=["System"], dependencies=[Depends(verify_token)])
async def rebuild_index():
    """Rebuild semantic search index from scratch"""
    try:
        logger.info("Rebuilding search index...")
        
        result = processor.rebuild_index()
        
        logger.info("Index rebuilt successfully")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rebuild failed: {str(e)}"
        )

# Enhanced Gemini endpoints
@app.get("/classify_intent", tags=["AI"], dependencies=[Depends(verify_token)])
async def classify_intent(query: str):
    """Classify query intent using Gemini"""
    try:
        intent = processor.query_parser.classify_query_intent(query)
        return JSONResponse(content={"query": query, "intent": intent})
        
    except Exception as e:
        logger.error(f"Intent classification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )

@app.get("/expand_query", tags=["AI"], dependencies=[Depends(verify_token)])
async def expand_query(query: str):
    """Generate search variations using Gemini"""
    try:
        variations = processor.query_parser.expand_query_for_search(query)
        return JSONResponse(content={"original_query": query, "variations": variations})
        
    except Exception as e:
        logger.error(f"Query expansion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expansion failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    uvicorn.run(
        "src.api.production_main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )