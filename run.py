#!/usr/bin/env python3
"""
Railway startup script with proper PORT handling
"""
import os
import sys

def main():
    # Get port from environment with proper validation
    port_str = os.getenv("PORT", "8000")
    
    try:
        port = int(port_str)
    except ValueError:
        print(f"Invalid PORT value: {port_str}, using default 8000")
        port = 8000
    
    print(f"🚀 Starting Hackathon API server on port {port}")
    print(f"📊 Environment check:")
    print(f"  PORT: {port}")
    print(f"  HACKATHON_API_TOKEN: {'✅ SET' if os.getenv('HACKATHON_API_TOKEN') else '❌ NOT SET'}")
    print(f"  GOOGLE_API_KEY: {'✅ SET' if os.getenv('GOOGLE_API_KEY') else '❌ NOT SET'}")
    
    # Import and run uvicorn directly
    try:
        import uvicorn
        print(f"🔄 Starting uvicorn server...")
        
        uvicorn.run(
            "src.api.hackathon_main:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()