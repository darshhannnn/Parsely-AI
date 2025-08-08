"""
Startup script for the hackathon API
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        "GOOGLE_API_KEY",
        "HACKATHON_API_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def main():
    """Start the hackathon API server"""
    print("🚀 Starting Hackathon API Server")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Display configuration
    print(f"🔑 API Token: {os.getenv('HACKATHON_API_TOKEN', 'not_set')[:20]}...")
    print(f"🤖 Google API Key: {os.getenv('GOOGLE_API_KEY', 'not_set')[:20]}...")
    print(f"🌐 Server will start on: http://0.0.0.0:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🏥 Health Check: http://localhost:8000/health")
    print(f"🎯 Main Endpoint: http://localhost:8000/hackrx/run")
    
    print("\n" + "=" * 50)
    print("Starting server...")
    
    # Start the server
    try:
        uvicorn.run(
            "src.api.hackathon_main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()