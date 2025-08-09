"""
Test script for deployed hackathon API
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deployed_api(base_url):
    """Test the deployed API"""
    print(f"🧪 Testing deployed API at: {base_url}")
    print("=" * 60)
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Health endpoint working")
            health_data = response.json()
            print(f"   📊 Service: {health_data.get('service', 'Unknown')}")
        else:
            print(f"   ❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health endpoint error: {str(e)}")
        return False
    
    # Test main endpoint
    try:
        print("\n2. Testing main endpoint...")
        headers = {
            "Authorization": f"Bearer {os.getenv('HACKATHON_API_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        test_request = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": ["What is this document about?"]
        }
        
        response = requests.post(
            f"{base_url}/hackrx/run", 
            headers=headers, 
            json=test_request, 
            timeout=60
        )
        
        if response.status_code == 200:
            print("   ✅ Main endpoint working")
            data = response.json()
            if 'answers' in data and len(data['answers']) > 0:
                print(f"   📝 Answer: {data['answers'][0][:100]}...")
                print("   🎉 API is fully functional!")
                return True
            else:
                print("   ❌ Invalid response format")
                return False
        else:
            print(f"   ❌ Main endpoint failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Main endpoint error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Deployed API Tester")
    print("=" * 30)
    
    # Get API URL from user
    api_url = input("Enter your deployed API URL (e.g., https://your-app.railway.app): ").strip()
    
    if not api_url:
        print("❌ No URL provided")
        return
    
    if not api_url.startswith('http'):
        api_url = 'https://' + api_url
    
    # Test the API
    success = test_deployed_api(api_url)
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 HACKATHON SUBMISSION READY!")
        print(f"📝 Webhook URL: {api_url}/hackrx/run")
        print("✅ Your API is ready for hackathon submission!")
    else:
        print("\n" + "=" * 60)
        print("❌ API needs fixes before submission")
        print("🔧 Check the deployment logs and fix any issues")

if __name__ == "__main__":
    main()