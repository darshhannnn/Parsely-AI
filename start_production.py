#!/usr/bin/env python3
"""
Production startup script for Parsely AI
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def check_environment():
    """Check if production environment is properly configured"""
    
    print("🔍 Checking production environment...")
    
    required_vars = [
        'GOOGLE_API_KEY',
        'LLM_MODEL',
        'API_KEY_SECRET'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    
    print("✅ Environment check passed")
    return True

def check_directories():
    """Ensure required directories exist"""
    
    print("📁 Checking directories...")
    
    directories = [
        'data/policies',
        'data/embeddings', 
        'logs',
        'ssl'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories ready")

def test_api_connection():
    """Test if the API is responding"""
    
    print("🧪 Testing API connection...")
    
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                print("✅ API is responding")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("❌ API is not responding")
    return False

def start_production_server():
    """Start the production server"""
    
    print("🚀 Starting production server...")
    
    # Use production main with security
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.production_main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "4",
        "--log-level", "info",
        "--access-log"
    ]
    
    try:
        process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        if test_api_connection():
            print("🎉 Production server started successfully!")
            print("=" * 50)
            print("🌐 API URL: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("🔍 Health Check: http://localhost:8000/health")
            print("📊 Metrics: http://localhost:8000/metrics")
            print("=" * 50)
            print("🛑 Press Ctrl+C to stop the server")
            
            # Keep the script running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                print("✅ Server stopped")
        else:
            print("❌ Server failed to start properly")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def show_production_info():
    """Show production deployment information"""
    
    print("\n📋 Production Deployment Information")
    print("=" * 40)
    print("🔑 Authentication: Bearer token required")
    print("🛡️ Security: HTTPS recommended for production")
    print("📊 Monitoring: Prometheus metrics available")
    print("🗄️ Caching: Redis integration available")
    print("📝 Logging: Structured logging to files")
    print("🔄 Load Balancing: Nginx reverse proxy ready")
    print("\n🚀 Your system is production-ready!")

def main():
    """Main production startup function"""
    
    print("🌿 Parsely AI - Production Mode")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Pre-flight checks
    if not check_environment():
        sys.exit(1)
    
    check_directories()
    
    # Show deployment info
    show_production_info()
    
    # Ask user for confirmation
    print("\n🎯 Ready to start production server?")
    choice = input("Enter 'yes' to continue: ").lower().strip()
    
    if choice != 'yes':
        print("👋 Deployment cancelled")
        sys.exit(0)
    
    # Start server
    if start_production_server():
        print("🎉 Production deployment completed successfully!")
    else:
        print("💥 Production deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()