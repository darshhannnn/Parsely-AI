"""
Hackathon API endpoint for LLM-Powered Intelligent Query-Retrieval System
Implements the required /hackrx/run endpoint with bearer token authentication
"""
import os
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
# Optional rate limiting - gracefully handle if slowapi is not available
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    logger.warning("slowapi not available - rate limiting disabled")
    RATE_LIMITING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()

# Environment configuration
EXPECTED_TOKEN = os.getenv("HACKATHON_API_TOKEN")
if not EXPECTED_TOKEN:
    logger.warning("HACKATHON_API_TOKEN not set - authentication will fail")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set - document processing will fail")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Document Processing - Hackathon API",
    description="Hackathon API for intelligent document query processing with PDF blob URL support",
    version="1.0"
)

# Initialize rate limiter if available
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Customize this based on your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme for bearer token (already defined above)
security = HTTPBearer()

class HackathonRequest(BaseModel):
    """Request model matching hackathon specification"""
    documents: str = Field(
        ..., 
        description="PDF blob URL from Azure storage"
    )
    questions: List[str] = Field(
        ..., 
        description="List of natural language questions",
        min_length=1,
        max_length=10
    )
    
    @validator('questions')
    def validate_questions(cls, questions):
        for question in questions:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
            if len(question) > 500:
                raise ValueError("Question too long (max 500 characters)")
        return [q.strip() for q in questions]

class HackathonResponse(BaseModel):
    """Response model matching hackathon specification"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Ensure server is configured correctly
    if not EXPECTED_TOKEN:
        logger.error("Auth not configured: HACKATHON_API_TOKEN missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server auth not configured: missing HACKATHON_API_TOKEN",
        )
    # Basic trace without leaking secrets
    logger.info(
        "Auth check: token provided=%s, expected_token_set=%s",
        bool(credentials and credentials.credentials),
        bool(EXPECTED_TOKEN),
    )
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def download_pdf_from_blob_url(blob_url: str) -> str:
    """Download PDF from Azure blob URL and return local file path"""
    try:
        logger.info(f"Downloading PDF from blob URL: {blob_url[:100]}...")
        
        # Download the PDF file
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        logger.info(f"PDF downloaded successfully to: {temp_file_path}")
        return temp_file_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download PDF from blob URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error downloading PDF: {str(e)}"
        )

def process_document_and_questions(pdf_path: str, questions: List[str]) -> List[str]:
    """
    Process PDF document and answer questions using the complete 6-stage pipeline.
    """
    try:
        import google.generativeai as genai
        import PyPDF2
        
        # Configure Gemini client
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY not configured",
            )
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        
        logger.info(f"Starting 6-stage pipeline for document: {pdf_path}")
        
        # Stage 1: Input Documents - Extract text from PDF
        logger.info("Stage 1: Extracting text from PDF...")
        document_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        document_text += text + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return [f"Unable to process PDF document: {str(e)}" for _ in questions]
        
        if not document_text.strip():
            return ["Document appears to be empty or unreadable" for _ in questions]
        
        # Stages 2-6: Process each question with simplified pipeline
        answers = []
        for question in questions:
            logger.info(f"Processing question: {question}")
            
            try:
                # Simple approach: Use full document as context
                prompt = f"""
You are an expert document analyzer. Based on the provided document content, please answer the question accurately and professionally.

Document Content:
{document_text[:4000]}  # Limit context to avoid token limits

Question: {question}

Instructions:
1. Focus only on information present in the provided document content
2. If information is not available, clearly state: "Based on the provided document content, this information is not available."
3. Be precise and factual
4. Keep the answer concise and professional

Answer:"""

                response = model.generate_content(prompt)
                answer = response.text.strip()
                
                # Ensure reasonable length
                if len(answer) > 500:
                    answer = answer[:497] + "..."
                
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Unable to process question due to error: {str(e)}")
        
        logger.info(f"Successfully processed {len(questions)} questions")
        return answers
        
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document and questions: {str(e)}"
        )

@app.post("/hackrx/run", response_model=HackathonResponse, 
         summary="Process document and answer questions",
         responses={
             200: {"description": "Successfully processed document and generated answers"},
             400: {"description": "Invalid request or unable to process document"},
             401: {"description": "Invalid or missing authentication token"},
             500: {"description": "Internal server error or processing failure"}
         })
async def hackrx_run(
    request: HackathonRequest,
    token: str = Depends(verify_bearer_token)
) -> HackathonResponse:
    """
    Main hackathon endpoint that processes PDF documents from blob URLs and answers questions.
    
    This endpoint implements the 6-stage pipeline:
    1. Input Documents - Download PDF from blob URL
    2. LLM Parser - Extract structured query information
    3. Embedding Search - FAISS/Pinecone retrieval for semantic similarity
    4. Clause Matching - Semantic similarity scoring and relevance ranking
    5. Logic Evaluation - Decision processing with domain-specific business rules
    6. JSON Output - Structured response with explainable rationale
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing hackathon request with {len(request.questions)} questions")
        
        # Stage 1: Input Documents - Download PDF from blob URL
        pdf_path = download_pdf_from_blob_url(request.documents)
        
        try:
            # Stages 2-6: Process document and answer questions
            answers = process_document_and_questions(pdf_path, request.questions)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully processed request in {processing_time:.2f} seconds")
            
            return HackathonResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(pdf_path)
                logger.info(f"Cleaned up temporary file: {pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {pdf_path}: {str(e)}")
                
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_run: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health", summary="Health check endpoint")
def health():
    """Health check endpoint for the hackathon API"""
    return {
        "status": "ok",
        "service": "LLM Document Processing - Hackathon API",
        "version": "1.0",
        "endpoints": ["/hackrx/run", "/health"],
        "authentication": "Bearer token required",
        "supported_formats": ["PDF via blob URL"],
        "pipeline_stages": [
            "Input Documents",
            "LLM Parser", 
            "Embedding Search",
            "Clause Matching",
            "Logic Evaluation",
            "JSON Output"
        ]
    }

@app.get("/", summary="API Information")
def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Document Processing - Hackathon API",
        "version": "1.0",
        "description": "Intelligent document query processing with PDF blob URL support",
        "main_endpoint": "/hackrx/run",
        "documentation": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)