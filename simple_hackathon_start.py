#!/usr/bin/env python3
"""
Simple Hackathon Server Start
Starts the server and provides instructions for ngrok
"""

import os
import sys
import time
import subprocess
import requests
import webbrowser
import json

def setup_and_start():
    """Setup environment and start server"""
    print("🌿 Parsely AI - Hackathon Server")
    print("="*40)
    
    # Setup environment
    print("🔧 Setting up environment...")
    os.environ["HACKATHON_API_TOKEN"] = "hackrx_2024_parsely_ai_token"
    os.environ["PORT"] = "8001"
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Checking for GOOGLE_API_KEY...")
        
        # Try to load from .env file
        try:
            with open(".env", 'r') as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        os.environ["GOOGLE_API_KEY"] = key
                        print("✅ Loaded GOOGLE_API_KEY from .env file")
                        break
        except FileNotFoundError:
            pass
        
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ GOOGLE_API_KEY not found!")
            print("   Please add it to your .env file:")
            print("   GOOGLE_API_KEY=your_key_here")
            return False
    
    print("✅ Environment ready")
    
    # Start server
    print("🚀 Starting API server on port 8001...")
    print("   Health Check: http://localhost:8001/health")
    print("   API Docs: http://localhost:8001/docs")
    print("   Main Endpoint: http://localhost:8001/hackrx/run")
    
    # Open browser to docs
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8001/docs")
    except:
        pass
    
    print("\n🌐 To get your public webhook URL:")
    print("   1. Install ngrok from https://ngrok.com")
    print("   2. Run: ngrok http 8001")
    print("   3. Copy the https URL from ngrok output")
    print("   4. Your webhook URL: https://[ngrok-url]/hackrx/run")
    
    print("\n🎯 Starting server... (Press Ctrl+C to stop)")
    
    # Start the server
    try:
        subprocess.run([sys.executable, "src/api/hackathon_main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    setup_and_start()