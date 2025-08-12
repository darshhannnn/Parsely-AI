#!/usr/bin/env python3
"""
Final Hackathon Setup - Parsely AI
Automatically starts server with correct configuration
"""

import os
import sys
import subprocess
import time
import webbrowser
import json

def main():
    """Main setup function"""
    print("🌿 Parsely AI - Final Hackathon Setup")
    print("="*45)
    
    # Set environment variables
    print("🔧 Setting up environment...")
    os.environ["HACKATHON_API_TOKEN"] = "hackrx_2024_parsely_ai_token"
    os.environ["PORT"] = "8002"
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Checking for GOOGLE_API_KEY...")
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
    
    print("✅ Environment configured")
    print(f"   Port: 8002")
    print(f"   Token: hackrx_2024_parsely_ai_token")
    
    # Create webhook info
    webhook_info = {
        "local_server": "http://localhost:8002",
        "health_check": "http://localhost:8002/health",
        "api_docs": "http://localhost:8002/docs",
        "main_endpoint": "http://localhost:8002/hackrx/run",
        "auth_token": "hackrx_2024_parsely_ai_token",
        "ngrok_command": "ngrok http 8002",
        "webhook_url_format": "https://[ngrok-url]/hackrx/run"
    }
    
    # Save webhook info
    with open("webhook_info.json", "w") as f:
        json.dump(webhook_info, f, indent=2)
    
    print("\n📋 Server Information:")
    print(f"   Health Check: http://localhost:8002/health")
    print(f"   API Docs: http://localhost:8002/docs")
    print(f"   Main Endpoint: http://localhost:8002/hackrx/run")
    
    print("\n🌐 To get your public webhook URL:")
    print("   1. Install ngrok: https://ngrok.com")
    print("   2. Run in another terminal: ngrok http 8002")
    print("   3. Copy the https URL from ngrok")
    print("   4. Your webhook URL: https://[ngrok-url]/hackrx/run")
    
    print("\n💾 Webhook info saved to: webhook_info.json")
    
    # Open browser
    print("\n🌐 Opening API documentation...")
    try:
        time.sleep(2)
        webbrowser.open("http://localhost:8002/docs")
    except:
        pass
    
    print("\n🚀 Starting server on port 8002...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        # Start the server
        subprocess.run([sys.executable, "src/api/hackathon_main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure port 8002 is free")
        print("   - Check that GOOGLE_API_KEY is set in .env file")
        print("   - Try running: python src/api/hackathon_main.py directly")

if __name__ == "__main__":
    main()