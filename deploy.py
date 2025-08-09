#!/usr/bin/env python3
"""
Deployment helper script for hackathon API
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "requirements.txt",
        "src/api/hackathon_main.py",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def check_environment():
    """Check environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["GOOGLE_API_KEY", "HACKATHON_API_TOKEN"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    
    print("✅ Environment variables configured")
    return True

def deploy_railway():
    """Deploy to Railway"""
    print("\n🚂 Deploying to Railway...")
    print("1. Install Railway CLI: npm install -g @railway/cli")
    print("2. Login: railway login")
    print("3. Create project: railway new")
    print("4. Set environment variables:")
    print(f"   railway variables set GOOGLE_API_KEY={os.getenv('GOOGLE_API_KEY')}")
    print(f"   railway variables set HACKATHON_API_TOKEN={os.getenv('HACKATHON_API_TOKEN')}")
    print("5. Deploy: railway up")
    print("\n📝 Your webhook URL will be: https://your-app.railway.app/hackrx/run")

def deploy_render():
    """Deploy to Render"""
    print("\n🎨 Deploying to Render...")
    print("1. Go to https://render.com")
    print("2. Connect your GitHub repository")
    print("3. Create a new Web Service")
    print("4. Use these settings:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: python -m uvicorn src.api.hackathon_main:app --host 0.0.0.0 --port $PORT")
    print("5. Add environment variables:")
    print(f"   - GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")
    print(f"   - HACKATHON_API_TOKEN: {os.getenv('HACKATHON_API_TOKEN')}")
    print("   - PYTHONPATH: /opt/render/project/src")
    print("\n📝 Your webhook URL will be: https://your-app.onrender.com/hackrx/run")

def deploy_fly():
    """Deploy to Fly.io"""
    print("\n🪰 Deploying to Fly.io...")
    print("1. Install Fly CLI: https://fly.io/docs/getting-started/installing-flyctl/")
    print("2. Login: fly auth login")
    print("3. Launch app: fly launch")
    print("4. Set secrets:")
    print(f"   fly secrets set GOOGLE_API_KEY={os.getenv('GOOGLE_API_KEY')}")
    print(f"   fly secrets set HACKATHON_API_TOKEN={os.getenv('HACKATHON_API_TOKEN')}")
    print("5. Deploy: fly deploy")
    print("\n📝 Your webhook URL will be: https://your-app.fly.dev/hackrx/run")

def deploy_vercel():
    """Deploy to Vercel"""
    print("\n▲ Deploying to Vercel...")
    print("1. Install Vercel CLI: npm install -g vercel")
    print("2. Login: vercel login")
    print("3. Add secrets:")
    print(f"   vercel secrets add google_api_key {os.getenv('GOOGLE_API_KEY')}")
    print(f"   vercel secrets add hackathon_api_token {os.getenv('HACKATHON_API_TOKEN')}")
    print("4. Deploy: vercel --prod")
    print("\n📝 Your webhook URL will be: https://your-app.vercel.app/hackrx/run")

def main():
    """Main deployment function"""
    print("🚀 Hackathon API Deployment Helper")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    print("\n🎯 Choose deployment platform:")
    print("1. Railway (Recommended - Easy & Fast)")
    print("2. Render (Free tier available)")
    print("3. Fly.io (Good performance)")
    print("4. Vercel (Serverless)")
    print("5. Show all instructions")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        deploy_railway()
    elif choice == "2":
        deploy_render()
    elif choice == "3":
        deploy_fly()
    elif choice == "4":
        deploy_vercel()
    elif choice == "5":
        deploy_railway()
        deploy_render()
        deploy_fly()
        deploy_vercel()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Deployment instructions provided!")
    print("📋 After deployment, test your API with:")
    print("   python validate_hackathon_requirements.py")
    print("   (Update BASE_URL in the script to your deployed URL)")

if __name__ == "__main__":
    main()