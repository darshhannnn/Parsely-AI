#!/usr/bin/env python3
"""
Simple Railway startup script
"""
import os
import subprocess
import sys

def main():
    # Get port from environment
    port = os.getenv("PORT", "8000")
    
    print(f"Starting server on port {port}")
    print(f"Environment variables:")
    print(f"  PORT: {port}")
    print(f"  HACKATHON_API_TOKEN: {'SET' if os.getenv('HACKATHON_API_TOKEN') else 'NOT SET'}")
    print(f"  GOOGLE_API_KEY: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    
    # Start uvicorn with the correct module path
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.api.hackathon_main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()