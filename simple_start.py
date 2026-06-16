#!/usr/bin/env python3
"""
Simple server startup script
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Start the server"""
    print("🚀 Starting Parsely AI Server")
    print("=" * 40)
    
    # Check environment
    token = os.getenv("HACKATHON_API_TOKEN")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    print(f"🔑 API Token: {'✅ Set' if token else '❌ Missing'}")
    print(f"🤖 Google API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🎯 Main Endpoint: http://localhost:8000/hackrx/run")
    print("=" * 40)
    
    if not token or not api_key:
        print("❌ Missing required environment variables!")
        return
    
    try:
        # Import and start
        from src.api.hackathon_main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()