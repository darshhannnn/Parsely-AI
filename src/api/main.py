from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..insurance_claim_processor import InsuranceClaimProcessor
from .auth import verify_api_key

app = FastAPI(
    title="Parsely AI API", 
    description="Fresh take on document processing - Gemini-powered intelligent document analysis and decision making.", 
    version="2.0"
)

# Initialize the processor with Gemini components
processor = InsuranceClaimProcessor()

class ClaimRequest(BaseModel):
    query: str
    return_json: Optional[bool] = True

@app.post("/process_claim", summary="Process an insurance claim query")
def process_claim(request: ClaimRequest, api_key: str = Depends(verify_api_key)):
    result = processor.process_claim(request.query, return_json=request.return_json)
    return JSONResponse(content=result)

@app.get("/analyze_query", summary="Analyze/extract entities from a query")
def analyze_query(query: str = Query(..., description="Natural language claim query"), api_key: str = Depends(verify_api_key)):
    result = processor.analyze_query_only(query)
    return JSONResponse(content=result)

@app.get("/search_clauses", summary="Semantic search for relevant policy clauses")
def search_clauses(query: str = Query(...), top_k: int = Query(5, ge=1, le=20), api_key: str = Depends(verify_api_key)):
    result = processor.search_clauses_only(query, top_k=top_k)
    return JSONResponse(content=result)

@app.get("/policy_summary", summary="Get summary of loaded policies and clause counts")
def policy_summary(api_key: str = Depends(verify_api_key)):
    result = processor.get_policy_summary()
    return JSONResponse(content=result)

@app.post("/rebuild_index", summary="Rebuild semantic search index from scratch")
def rebuild_index(api_key: str = Depends(verify_api_key)):
    result = processor.rebuild_index()
    return JSONResponse(content=result)

@app.get("/classify_intent", summary="Classify query intent using Gemini")
def classify_intent(query: str = Query(..., description="Natural language query"), api_key: str = Depends(verify_api_key)):
    try:
        intent = processor.query_parser.classify_query_intent(query)
        return JSONResponse(content={"query": query, "intent": intent})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/expand_query", summary="Generate search variations using Gemini")
def expand_query(query: str = Query(..., description="Natural language query"), api_key: str = Depends(verify_api_key)):
    try:
        variations = processor.query_parser.expand_query_for_search(query)
        return JSONResponse(content={"original_query": query, "variations": variations})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/health", summary="Health check endpoint")
def health():
    return {
        "status": "ok",
        "service": "Parsely AI", 
        "llm_provider": "google_gemini",
        "model": "gemini-1.5-flash",
        "version": "2.0",
        "tagline": "Fresh take on document processing"
    }
