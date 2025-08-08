"""
Simplified API that works without document processing
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="LLM Document Processing API", 
    description="Gemini-powered intelligent query processing API.", 
    version="2.0"
)

class ClaimRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse, summary="API Home Page")
def home():
    """Home page with API information"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Document Processing API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { font-weight: bold; color: #27ae60; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .status { background: #d5f4e6; color: #27ae60; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 LLM Document Processing API</h1>
            <p><span class="status">✅ ONLINE</span> - Powered by Google Gemini 2.0 Flash</p>
            
            <h2>📚 Available Endpoints</h2>
            
            <div class="endpoint">
                <div class="method">GET /health</div>
                <p>Check API health and configuration status</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /analyze_query</div>
                <p>Extract entities from insurance queries using AI</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /classify_intent</div>
                <p>Classify the intent of a query (coverage check, claim evaluation, etc.)</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /expand_query</div>
                <p>Generate search variations for better document matching</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /process_simple_claim</div>
                <p>Process insurance claims with AI-powered decision making</p>
            </div>
            
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="/docs">📖 Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">📋 Alternative Documentation (ReDoc)</a></li>
                <li><a href="/health">💚 Health Check</a></li>
            </ul>
            
            <h2>🧪 Example Usage</h2>
            <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Test query analysis
curl -X POST "http://localhost:8000/analyze_query" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "46-year-old male, knee surgery in Pune, 3-month policy"}'

# Check system health  
curl http://localhost:8000/health
            </pre>
            
            <footer style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; text-align: center;">
                <p>LLM Document Processing System v2.0 | Powered by Gemini AI</p>
            </footer>
        </div>
    </body>
    </html>
    """)

@app.get("/favicon.ico")
def favicon():
    """Return a simple favicon"""
    # Simple 16x16 pixel favicon as base64 encoded ICO
    favicon_data = """
    AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAABILAAASCwAAAAAAAAAAAAD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///8A////AP///wD///wAAAAA
    """
    import base64
    favicon_bytes = base64.b64decode(favicon_data.strip())
    return Response(content=favicon_bytes, media_type="image/x-icon")

@app.get("/health", summary="Health check endpoint")
def health():
    api_key = os.getenv("GOOGLE_API_KEY")
    api_key_status = "configured" if api_key else "missing"
    api_key_preview = f"{api_key[:10]}..." if api_key else "not_set"
    
    return {
        "status": "ok", 
        "llm_provider": "google_gemini",
        "model": os.getenv("LLM_MODEL", "gemini-2.0-flash-exp"),
        "version": "2.0",
        "api_key_status": api_key_status,
        "api_key_preview": api_key_preview,
        "endpoints": [
            "/", "/health", "/docs", "/analyze_query", 
            "/classify_intent", "/expand_query", "/process_simple_claim"
        ]
    }

@app.post("/analyze_query", summary="Analyze/extract entities from a query using Gemini")
def analyze_query(request: ClaimRequest):
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        result = parser.parse_query(request.query)
        
        return JSONResponse(content={
            "raw_query": request.query,
            "extracted_entities": {
                "age": result.age,
                "gender": result.gender,
                "procedure": result.procedure,
                "location": result.location,
                "policy_age_months": result.policy_age_months,
                "hospital": result.hospital,
                "amount_claimed": result.amount_claimed,
                "date": result.date
            },
            "assumptions": result.assumptions,
            "confidence": "high"
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/classify_intent", summary="Classify query intent using Gemini")
def classify_intent(query: str = Query(..., description="Natural language query")):
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        intent = parser.classify_query_intent(query)
        
        return JSONResponse(content={"query": query, "intent": intent})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/expand_query", summary="Generate search variations using Gemini")
def expand_query(query: str = Query(..., description="Natural language query")):
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        variations = parser.expand_query_for_search(query)
        
        return JSONResponse(content={"original_query": query, "variations": variations})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/process_simple_claim", summary="Simple claim processing without document search")
def process_simple_claim(request: ClaimRequest):
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        import google.generativeai as genai
        
        # Parse the query
        parser = GeminiQueryParser()
        parsed_query = parser.parse_query(request.query)
        
        # Use Gemini to make a simple decision
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(os.getenv("LLM_MODEL", "gemini-2.0-flash-exp"))
        
        decision_prompt = f"""
        Based on this insurance claim query, provide a simple evaluation:
        
        Query: {request.query}
        
        Extracted details:
        - Age: {parsed_query.age}
        - Gender: {parsed_query.gender}
        - Procedure: {parsed_query.procedure}
        - Location: {parsed_query.location}
        - Policy Age: {parsed_query.policy_age_months} months
        
        Provide a JSON response with:
        {{
            "decision": "approved" or "rejected" or "needs_review",
            "estimated_amount": number or null,
            "reasoning": "brief explanation",
            "confidence": 0.0-1.0
        }}
        
        Only return the JSON, no other text.
        """
        
        response = model.generate_content(decision_prompt)
        
        # Try to parse the JSON response
        import json
        try:
            decision_data = json.loads(response.text.strip())
        except:
            decision_data = {
                "decision": "needs_review",
                "estimated_amount": None,
                "reasoning": "Unable to process automatically",
                "confidence": 0.5
            }
        
        return JSONResponse(content={
            "query_analysis": {
                "raw_query": request.query,
                "extracted_entities": {
                    "age": parsed_query.age,
                    "gender": parsed_query.gender,
                    "procedure": parsed_query.procedure,
                    "location": parsed_query.location,
                    "policy_age_months": parsed_query.policy_age_months
                }
            },
            "decision": decision_data.get("decision", "needs_review"),
            "estimated_amount": decision_data.get("estimated_amount"),
            "reasoning": decision_data.get("reasoning", "AI-based evaluation"),
            "confidence": decision_data.get("confidence", 0.5),
            "note": "This is a simplified evaluation without full document processing"
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/robots.txt", response_class=Response)
def robots():
    """Robots.txt file"""
    return Response(content="User-agent: *\nDisallow: /", media_type="text/plain")

@app.get("/sitemap.xml", response_class=Response)  
def sitemap():
    """Simple sitemap"""
    sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>http://localhost:8000/</loc>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>http://localhost:8000/docs</loc>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>http://localhost:8000/health</loc>
        <changefreq>daily</changefreq>
        <priority>0.6</priority>
    </url>
</urlset>"""
    return Response(content=sitemap_content, media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)