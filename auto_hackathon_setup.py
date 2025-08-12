#!/usr/bin/env python3
"""
Automated Hackathon Setup - Parsely AI
Automatically starts server and ngrok tunnel
"""

import os
import sys
import time
import json
import subprocess
import threading
import requests
import webbrowser
from pathlib import Path

class HackathonSetup:
    def __init__(self):
        self.server_process = None
        self.ngrok_process = None
        self.webhook_url = None
        self.port = 8001
        
    def setup_environment(self):
        """Setup environment variables"""
        print("🔧 Setting up environment...")
        
        # Set API token
        os.environ["HACKATHON_API_TOKEN"] = "hackrx_2024_parsely_ai_token"
        os.environ["PORT"] = str(self.port)
        
        # Check for Google API key
        if not os.getenv("GOOGLE_API_KEY"):
            print("⚠️  GOOGLE_API_KEY not found in environment")
            
            # Try to load from .env file
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith("GOOGLE_API_KEY="):
                            key = line.split("=", 1)[1].strip()
                            os.environ["GOOGLE_API_KEY"] = key
                            print("✅ Loaded GOOGLE_API_KEY from .env file")
                            break
            
            if not os.getenv("GOOGLE_API_KEY"):
                print("❌ GOOGLE_API_KEY is required!")
                print("   Please set it in your .env file or environment")
                return False
        
        print("✅ Environment setup complete")
        return True
    
    def check_ngrok(self):
        """Check if ngrok is available"""
        try:
            result = subprocess.run(["ngrok", "version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Ngrok found")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("❌ Ngrok not found!")
        print("   Please install ngrok from https://ngrok.com")
        print("   Or run: choco install ngrok (if you have Chocolatey)")
        return False
    
    def start_server(self):
        """Start the FastAPI server"""
        print(f"🚀 Starting API server on port {self.port}...")
        
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, "src/api/hackathon_main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Check if server is running
            if self.server_process.poll() is None:
                # Test server
                try:
                    response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                    if response.status_code == 200:
                        print("✅ API server started successfully")
                        return True
                except:
                    pass
            
            print("❌ Failed to start API server")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def start_ngrok(self):
        """Start ngrok tunnel"""
        print("🌐 Starting ngrok tunnel...")
        
        try:
            self.ngrok_process = subprocess.Popen(
                ["ngrok", "http", str(self.port), "--log=stdout"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for ngrok to start and get URL
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    # Check ngrok API for tunnel info
                    response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("tunnels"):
                            tunnel = data["tunnels"][0]
                            self.webhook_url = tunnel["public_url"]
                            print(f"✅ Ngrok tunnel created: {self.webhook_url}")
                            return True
                except:
                    pass
                
                time.sleep(1)
                print(f"   Waiting for ngrok... ({attempt + 1}/{max_attempts})")
            
            print("❌ Failed to get ngrok URL")
            return False
            
        except Exception as e:
            print(f"❌ Error starting ngrok: {e}")
            return False
    
    def test_webhook(self):
        """Test the webhook endpoint"""
        if not self.webhook_url:
            return False
        
        print("🧪 Testing webhook endpoint...")
        
        try:
            # Test health endpoint
            health_url = f"{self.webhook_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                print("✅ Health check passed")
                
                # Test main endpoint (should fail without proper request)
                webhook_endpoint = f"{self.webhook_url}/hackrx/run"
                test_response = requests.post(webhook_endpoint, timeout=5)
                
                if test_response.status_code in [401, 422]:  # Expected auth/validation errors
                    print("✅ Main endpoint responding correctly")
                    return True
                
        except Exception as e:
            print(f"⚠️  Webhook test failed: {e}")
        
        return False
    
    def display_results(self):
        """Display final results"""
        print("\n" + "="*60)
        print("🎉 HACKATHON SETUP COMPLETE!")
        print("="*60)
        
        if self.webhook_url:
            webhook_endpoint = f"{self.webhook_url}/hackrx/run"
            print(f"🌐 Webhook URL: {webhook_endpoint}")
            print(f"📋 Health Check: {self.webhook_url}/health")
            print(f"📚 API Docs: {self.webhook_url}/docs")
            print(f"🔑 Auth Token: hackrx_2024_parsely_ai_token")
            
            print("\n📋 HACKATHON SUBMISSION INFO:")
            print(f"   Project: Parsely AI - LLM Document Processing")
            print(f"   Webhook URL: {webhook_endpoint}")
            print(f"   Method: POST")
            print(f"   Authentication: Bearer Token")
            print(f"   Content-Type: application/json")
            
            print("\n🧪 TEST COMMAND:")
            print(f'''curl -X POST "{webhook_endpoint}" \\
  -H "Authorization: Bearer hackrx_2024_parsely_ai_token" \\
  -H "Content-Type: application/json" \\
  -d '{{"documents": "https://example.com/test.pdf", "questions": ["What is this about?"]}}\'''')
            
            # Open browser to API docs
            try:
                webbrowser.open(f"{self.webhook_url}/docs")
                print(f"\n🌐 Opening API documentation in browser...")
            except:
                pass
            
            # Save webhook info to file
            webhook_info = {
                "webhook_url": webhook_endpoint,
                "health_check": f"{self.webhook_url}/health",
                "api_docs": f"{self.webhook_url}/docs",
                "auth_token": "hackrx_2024_parsely_ai_token",
                "method": "POST",
                "content_type": "application/json"
            }
            
            with open("hackathon_webhook_info.json", "w") as f:
                json.dump(webhook_info, f, indent=2)
            
            print(f"\n💾 Webhook info saved to: hackathon_webhook_info.json")
            
        else:
            print("❌ Setup failed - no webhook URL available")
        
        print("\n⚠️  IMPORTANT: Keep this terminal open during the hackathon!")
        print("   Closing this will stop your webhook URL from working.")
    
    def cleanup(self):
        """Cleanup processes"""
        print("\n🛑 Shutting down...")
        
        if self.ngrok_process:
            self.ngrok_process.terminate()
            print("   Stopped ngrok")
        
        if self.server_process:
            self.server_process.terminate()
            print("   Stopped API server")
    
    def run(self):
        """Run the complete setup"""
        print("🌿 Parsely AI - Automated Hackathon Setup")
        print("="*50)
        
        try:
            # Setup steps
            if not self.setup_environment():
                return False
            
            if not self.check_ngrok():
                return False
            
            if not self.start_server():
                return False
            
            if not self.start_ngrok():
                return False
            
            if not self.test_webhook():
                print("⚠️  Webhook test failed, but URL should still work")
            
            self.display_results()
            
            # Keep running
            print("\n⏳ Webhook is live! Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
        finally:
            self.cleanup()

def main():
    """Main function"""
    setup = HackathonSetup()
    setup.run()

if __name__ == "__main__":
    main()