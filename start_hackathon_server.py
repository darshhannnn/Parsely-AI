#!/usr/bin/env python3
"""
Start Parsely AI Hackathon Server
"""

import os
import subprocess
import sys
import time
import requests
from threading import Thread

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting Parsely AI Hackathon API Server...")
    
    # Set environment variables
    os.environ["HACKATHON_API_TOKEN"] = "hackrx_2024_parsely_ai_token"
    os.environ["PORT"] = "8001"
    
    # Check if GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not set. Please set it in your .env file")
    
    # Start the server
    try:
        subprocess.run([sys.executable, "src/api/hackathon_main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def test_server():
    """Test if server is running"""
    time.sleep(3)  # Wait for server to start
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running successfully!")
            print("📋 Server Info:")
            print(f"   Health Check: http://localhost:8001/health")
            print(f"   API Docs: http://localhost:8001/docs")
            print(f"   Main Endpoint: http://localhost:8001/hackrx/run")
            print("\n🌐 Now run ngrok in another terminal:")
            print("   ngrok http 8001")
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")

def main():
    """Main function"""
    print("🌿 Parsely AI - Hackathon Server Startup")
    print("=" * 50)
    
    # Start test in background
    test_thread = Thread(target=test_server, daemon=True)
    test_thread.start()
    
    # Start API server (blocking)
    start_api_server()

if __name__ == "__main__":
    main()