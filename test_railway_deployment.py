#!/usr/bin/env python3
"""
Test script for Railway deployment
"""

import os
import requests
import json
import time
from typing import Dict, Any

def test_railway_deployment(base_url: str, token: str) -> None:
    """Test the Railway deployed API"""
    
    print("🚂 Testing Railway Deployment")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Service: {health_data.get('service', 'Unknown')}")
            print(f"   Version: {health_data.get('version', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint accessible")
        else:
            print(f"⚠️ Root endpoint status: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 3: API Documentation
    print("\n3. Testing API Documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            print(f"   Docs URL: {base_url}/docs")
        else:
            print(f"⚠️ API docs status: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")
    
    # Test 4: Authentication Test (should fail without token)
    print("\n4. Testing Authentication...")
    try:
        test_payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["Test question?"]
        }
        
        # Test without token (should fail)
        response = requests.post(
            f"{base_url}/hackrx/run",
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 401:
            print("✅ Authentication properly enforced (401 without token)")
        else:
            print(f"⚠️ Unexpected auth response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Auth test error: {e}")
    
    # Test 5: Valid Request Test (with token)
    print("\n5. Testing Valid Request...")
    if not token or token == "your_token_here":
        print("⚠️ Skipping valid request test - no token provided")
        print("   Set HACKATHON_API_TOKEN environment variable to test")
    else:
        try:
            test_payload = {
                "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "questions": [
                    "What is this document about?",
                    "What type of document is this?"
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            print("   Sending test request...")
            response = requests.post(
                f"{base_url}/hackrx/run",
                json=test_payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✅ Valid request successful")
                result = response.json()
                print(f"   Answers received: {len(result.get('answers', []))}")
                for i, answer in enumerate(result.get('answers', [])[:2]):
                    print(f"   Answer {i+1}: {answer[:100]}...")
            else:
                print(f"❌ Valid request failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Valid request error: {e}")
    
    # Test 6: Performance Test
    print("\n6. Testing Performance...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=10)
        response_time = time.time() - start_time
        
        print(f"✅ Response time: {response_time:.3f}s")
        if response_time < 2.0:
            print("   🚀 Good performance!")
        elif response_time < 5.0:
            print("   ⚠️ Acceptable performance")
        else:
            print("   🐌 Slow response time")
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
    
    print("\n" + "="*50)
    print("🎯 HACKATHON SUBMISSION INFO")
    print("="*50)
    print(f"Webhook URL: {base_url}/hackrx/run")
    print(f"Health Check: {base_url}/health")
    print(f"API Docs: {base_url}/docs")
    print("Method: POST")
    print("Authentication: Bearer Token")
    print("Content-Type: application/json")
    print("\nExample curl command:")
    print(f'''curl -X POST "{base_url}/hackrx/run" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"documents": "PDF_BLOB_URL", "questions": ["Your question?"]}}\'''')

def main():
    """Main test function"""
    
    # Get configuration from environment or user input
    base_url = os.getenv("RAILWAY_URL")
    token = os.getenv("HACKATHON_API_TOKEN", "your_token_here")
    
    if not base_url:
        print("Enter your Railway URL (e.g., https://your-app.up.railway.app):")
        base_url = input().strip()
        
        if not base_url:
            print("❌ No URL provided")
            return
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    # Run tests
    test_railway_deployment(base_url, token)

if __name__ == "__main__":
    main()