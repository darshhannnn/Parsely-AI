#!/usr/bin/env python3
"""
Railway deployment entry point
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    print(f"HACKATHON_API_TOKEN set: {bool(os.getenv('HACKATHON_API_TOKEN'))}")
    print(f"GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
    
    uvicorn.run(
        "src.api.hackathon_main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )