"""
Hackathon API endpoint for LLM-Powered Intelligent Query-Retrieval System
Implements the required /hackrx/run endpoint with bearer token authentication
"""
import os
import re
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Configure logging first
logging.basicConfig(level=logging.INFO)
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

class HackathonRequest(BaseModel):
    """Request model matching hackathon specification"""
    documents: str = Field(
        ..., 
        description="Document blob URL from Azure storage (supports PDF, DOCX, and Email formats)"
    )
    questions: List[str] = Field(
        ..., 
        description="List of natural language questions",
        min_length=1,
        max_length=10
    )
    
    @field_validator('questions')
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

def download_document_from_blob_url(blob_url: str) -> tuple[str, str]:
    """Download document from Azure blob URL and return local file path and detected format"""
    try:
        logger.info(f"Downloading document from blob URL: {blob_url[:100]}...")
        
        # Download the document file
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        
        # Detect content type from headers and content
        content_type = response.headers.get('content-type', '').lower()
        content = response.content
        
        # Detect format from content and URL
        document_format = detect_document_format(content, blob_url, content_type)
        
        # Determine file extension
        if document_format == 'pdf':
            suffix = '.pdf'
        elif document_format == 'docx':
            suffix = '.docx'
        elif document_format == 'email':
            suffix = '.eml'
        else:
            suffix = '.pdf'  # Default fallback
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Document downloaded successfully to: {temp_file_path} (format: {document_format})")
        return temp_file_path, document_format
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document from blob URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error downloading document: {str(e)}"
        )


def detect_document_format(content: bytes, url: str, content_type: str) -> str:
    """Detect document format from content, URL, and content type"""
    
    # Check file signature (magic numbers) first
    if content.startswith(b'%PDF'):
        return 'pdf'
    elif content.startswith(b'PK\x03\x04') and b'word/' in content[:1000]:
        return 'docx'
    elif content.startswith((b'Return-Path:', b'Received:', b'From:', b'To:', b'Subject:')):
        return 'email'
    
    # Check content type
    if 'pdf' in content_type:
        return 'pdf'
    elif 'wordprocessingml' in content_type or 'msword' in content_type:
        return 'docx'
    elif 'rfc822' in content_type or content_type == 'message/rfc822':
        return 'email'
    
    # Check URL extension as fallback
    url_lower = url.lower()
    if url_lower.endswith('.pdf'):
        return 'pdf'
    elif url_lower.endswith(('.docx', '.doc')):
        return 'docx'
    elif url_lower.endswith(('.eml', '.msg')):
        return 'email'
    
    # Default to PDF
    return 'pdf'

def _split_text_intelligently(text: str, max_chars: int = 4000) -> str:
    """
    Split text at sentence boundaries to avoid truncating mid-sentence.
    Returns text truncated at the nearest sentence end.
    """
    if len(text) <= max_chars:
        return text

    # Truncate to max_chars
    truncated = text[:max_chars]

    # Find the last sentence boundary (., !, or ?) before the limit
    sentence_endings = ['.', '!', '?']
    for ending in sentence_endings:
        last_pos = truncated.rfind(ending)
        if last_pos > max_chars * 0.8:  # Ensure we don't lose too much content
            return truncated[:last_pos + 1]

    # If no good sentence boundary found, just truncate and add ellipsis
    return truncated + "..."


def _chunk_document(text: str, max_chunk_chars: int = 2000) -> List[str]:
    """Split a document into paragraph-aligned chunks of at most max_chunk_chars"""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_chars:
            # Flush the current chunk, then hard-split the oversized paragraph
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(paragraph), max_chunk_chars):
                chunks.append(paragraph[i:i + max_chunk_chars])
        elif len(current) + len(paragraph) + 2 <= max_chunk_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _build_question_context(document_text: str, question: str, max_chars: Optional[int] = None) -> str:
    """Build the most relevant context for a question within a character budget.

    If the document fits the budget it is returned whole. Otherwise it is
    chunked and the chunks most similar to the question (TF-IDF cosine) are
    selected, preserving document order so the model reads a coherent excerpt.
    """
    if max_chars is None:
        max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

    if len(document_text) <= max_chars:
        return document_text

    chunks = _chunk_document(document_text)
    if not chunks:
        return _split_text_intelligently(document_text, max_chars)

    ranked_indices = list(range(len(chunks)))
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = chunks + [question]
        vectors = TfidfVectorizer(stop_words='english').fit_transform(corpus)
        similarities = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
        ranked_indices = sorted(range(len(chunks)), key=lambda i: similarities[i], reverse=True)
    except Exception as e:
        logger.warning(f"TF-IDF ranking failed, falling back to document order: {e}")

    selected_indices: List[int] = []
    budget = max_chars
    for idx in ranked_indices:
        chunk_len = len(chunks[idx])
        if chunk_len <= budget:
            selected_indices.append(idx)
            budget -= chunk_len
        if budget <= 0:
            break

    if not selected_indices:
        # Every chunk exceeds the budget: use the best one, truncated
        best = chunks[ranked_indices[0]]
        return _split_text_intelligently(best, max_chars)

    # Restore document order for a coherent excerpt
    selected_indices.sort()
    return "\n\n".join(chunks[i] for i in selected_indices)

def process_document_and_questions(document_path: str, document_format: str, questions: List[str]) -> List[str]:
    """
    Process multi-format document and answer questions using the enhanced 6-stage pipeline.
    """
    try:
        import google.generativeai as genai
        
        # Configure Gemini client
        if not GOOGLE_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY not configured",
            )
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        
        logger.info(f"Starting enhanced 6-stage pipeline for {document_format} document: {document_path}")
        
        # Stage 1: Input Documents - Extract text using enhanced extractors
        logger.info(f"Stage 1: Extracting text from {document_format} document...")
        document_text = ""
        metadata = {}
        
        try:
            if document_format == 'pdf':
                document_text, metadata = extract_pdf_content_enhanced(document_path)
            elif document_format == 'docx':
                document_text, metadata = extract_docx_content_enhanced(document_path)
            elif document_format == 'email':
                document_text, metadata = extract_email_content_enhanced(document_path)
            else:
                # Fallback to basic PDF processing
                document_text, metadata = extract_pdf_content_basic(document_path)
                
        except Exception as e:
            logger.error(f"Error extracting {document_format} content: {str(e)}")
            return [f"Unable to process {document_format} document: {str(e)}" for _ in questions]
        
        if not document_text.strip():
            return ["Document appears to be empty or unreadable" for _ in questions]
        
        logger.info(f"Extracted {len(document_text)} characters from {document_format} document")

        # Stages 2-6: Process each question with enhanced context
        max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))
        document_fits = len(document_text) <= max_context_chars
        if not document_fits:
            logger.info(
                f"Document exceeds context budget ({len(document_text)} > {max_context_chars}); "
                f"per-question retrieval is enabled"
            )

        answers = []
        for question in questions:
            logger.info(f"Processing question: {question}")

            try:
                # Per-question context: the whole document when it fits the
                # model's context window, otherwise the most relevant chunks
                document_context = _build_question_context(
                    document_text, question, max_context_chars
                )

                # Enhanced approach: Use document structure and metadata
                context_info = ""
                if metadata.get('sections'):
                    context_info += f"Document has {len(metadata['sections'])} sections: {', '.join(list(metadata['sections'].keys())[:20])}\n"
                if metadata.get('document_type'):
                    context_info += f"Document type: {metadata['document_type']}\n"
                if metadata.get('word_count'):
                    context_info += f"Word count: {metadata['word_count']}\n"
                if not document_fits:
                    context_info += "Note: only the excerpts most relevant to the question are included below.\n"

                prompt = f"""
You are an expert document analyst. Answer the question using ONLY the provided document content.

Document Information:
{context_info}

Document Content:
{document_context}

Question: {question}

Instructions:
1. Base your answer strictly on the document content provided above.
2. Extract exact figures wherever they exist: amounts, percentages, time periods, limits, and waiting periods. Quote them precisely as written.
3. Cite the specific section, clause number, or heading the answer comes from.
4. If several clauses are relevant, synthesize them and mention each one.
5. If the information genuinely does not appear in the document, state exactly: "Based on the provided document content, this information is not available." Do not guess.
6. Be precise, factual and professional.

Answer:"""

                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1},
                )
                answer = response.text.strip()

                # Ensure reasonable length
                if len(answer) > 2000:
                    answer = answer[:1997] + "..."

                answers.append(answer)

            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Unable to process question due to error: {str(e)}")
        
        logger.info(f"Successfully processed {len(questions)} questions")
        return answers
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document and questions: {str(e)}"
        )


def extract_pdf_content_enhanced(pdf_path: str) -> tuple[str, dict]:
    """Extract PDF content using enhanced extractor"""
    try:
        import PyPDF2
        
        document_text = ""
        pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
                    document_text += page_text + "\n"
        
        metadata = {
            'document_type': 'PDF',
            'page_count': len(pages),
            'word_count': len(document_text.split()),
            'sections': {'full_document': document_text},
            'extraction_method': 'Enhanced PyPDF2'
        }
        
        return document_text, metadata
        
    except Exception as e:
        logger.error(f"Enhanced PDF extraction failed: {str(e)}")
        raise


def extract_docx_content_enhanced(docx_path: str) -> tuple[str, dict]:
    """Extract DOCX content using enhanced extractor"""
    try:
        from docx import Document
        
        doc = Document(docx_path)
        
        paragraphs = []
        sections = {}
        current_section = "Introduction"
        section_content = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
                
                # Detect headings (basic heuristic)
                if para.style.name.startswith('Heading'):
                    # Save previous section
                    if section_content:
                        sections[current_section] = '\n'.join(section_content)
                    
                    # Start new section
                    current_section = para.text.strip()
                    section_content = []
                else:
                    section_content.append(para.text.strip())
        
        # Save final section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        document_text = '\n\n'.join(paragraphs)
        
        metadata = {
            'document_type': 'DOCX',
            'paragraph_count': len(paragraphs),
            'section_count': len(sections),
            'word_count': len(document_text.split()),
            'sections': sections,
            'extraction_method': 'Enhanced python-docx'
        }
        
        # Add document properties if available
        if hasattr(doc, 'core_properties'):
            props = doc.core_properties
            metadata['document_properties'] = {
                'title': props.title,
                'author': props.author,
                'subject': props.subject,
                'created': props.created.isoformat() if props.created else None,
                'modified': props.modified.isoformat() if props.modified else None
            }
        
        return document_text, metadata
        
    except Exception as e:
        logger.error(f"Enhanced DOCX extraction failed: {str(e)}")
        raise


def extract_email_content_enhanced(email_path: str) -> tuple[str, dict]:
    """Extract email content using enhanced extractor"""
    try:
        import email
        from email import policy
        
        with open(email_path, 'rb') as f:
            msg = email.message_from_bytes(f.read(), policy=policy.default)
        
        # Extract headers
        headers = {
            'subject': msg.get('Subject', ''),
            'from': msg.get('From', ''),
            'to': msg.get('To', ''),
            'date': msg.get('Date', ''),
        }
        
        # Extract body content
        body_parts = []
        sections = {}
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    content = part.get_content()
                    if content and content.strip():
                        body_parts.append(content.strip())
        else:
            content = msg.get_content()
            if content and content.strip():
                body_parts.append(content.strip())
        
        # Combine all text content
        header_text = f"Subject: {headers['subject']}\nFrom: {headers['from']}\nTo: {headers['to']}\nDate: {headers['date']}\n"
        body_text = '\n\n'.join(body_parts)
        document_text = header_text + '\n\n' + body_text
        
        # Store sections
        sections['headers'] = header_text
        sections['body'] = body_text
        
        metadata = {
            'document_type': 'Email',
            'headers': headers,
            'is_multipart': msg.is_multipart(),
            'word_count': len(document_text.split()),
            'sections': sections,
            'extraction_method': 'Enhanced email.parser'
        }
        
        return document_text, metadata
        
    except Exception as e:
        logger.error(f"Enhanced email extraction failed: {str(e)}")
        raise


def extract_pdf_content_basic(pdf_path: str) -> tuple[str, dict]:
    """Basic PDF extraction fallback"""
    try:
        import PyPDF2
        
        document_text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    document_text += text + "\n"
        
        metadata = {
            'document_type': 'PDF',
            'word_count': len(document_text.split()),
            'sections': {'full_document': document_text},
            'extraction_method': 'Basic PyPDF2'
        }
        
        return document_text, metadata
        
    except Exception as e:
        logger.error(f"Basic PDF extraction failed: {str(e)}")
        raise

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
    Main hackathon endpoint that processes multi-format documents from blob URLs and answers questions.
    
    Supported formats: PDF, DOCX, Email (.eml)
    
    This endpoint implements the enhanced 6-stage pipeline:
    1. Input Documents - Download and detect document format from blob URL
    2. LLM Parser - Extract structured content with format-specific processing
    3. Embedding Search - FAISS/Pinecone retrieval for semantic similarity
    4. Clause Matching - Semantic similarity scoring and relevance ranking
    5. Logic Evaluation - Decision processing with domain-specific business rules
    6. JSON Output - Structured response with explainable rationale
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing hackathon request with {len(request.questions)} questions")
        
        # Stage 1: Input Documents - Download document from blob URL
        document_path, document_format = download_document_from_blob_url(request.documents)
        
        try:
            # Stages 2-6: Process document and answer questions
            answers = process_document_and_questions(document_path, document_format, request.questions)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully processed request in {processing_time:.2f} seconds")
            
            return HackathonResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(document_path)
                logger.info(f"Cleaned up temporary file: {document_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {document_path}: {str(e)}")
                
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
        "supported_formats": ["PDF", "DOCX", "Email (.eml)"],
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
        "description": "Intelligent multi-format document query processing with blob URL support",
        "main_endpoint": "/hackrx/run",
        "documentation": "/docs",
        "web_interface": "/ui",
        "health_check": "/health"
    }

# Path to the browser-based testing interface shipped with the project
WEB_INTERFACE_PATH = Path(__file__).resolve().parent.parent.parent / "web_interface.html"


@app.get("/ui", summary="Web testing interface", include_in_schema=True)
def web_interface():
    """Serve the browser-based testing interface for the API"""
    if WEB_INTERFACE_PATH.exists():
        return FileResponse(WEB_INTERFACE_PATH)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="web_interface.html not found next to the project root"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"Starting Parsely AI Hackathon API on {host}:{port}")
    logger.info(f"Health check: http://{host}:{port}/health")
    logger.info(f"API docs: http://{host}:{port}/docs")
    logger.info(f"Main endpoint: http://{host}:{port}/hackrx/run")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )
