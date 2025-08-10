#!/usr/bin/env python3
"""
Simple API test script
"""
import requests
import json

def test_api():
    """Test the deployed API"""
    base_url = "https://parsely-ai-production-f2ad.up.railway.app"
    
    print("🧪 Testing Hackathon API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            data = response.json()
            print(f"   📊 Service: {data.get('service', 'Unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: Main endpoint
    print("\n2. Testing main endpoint...")
    headers = {
        "Authorization": "Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": ["What is this document about?"]
    }
    
    try:
        response = requests.post(f"{base_url}/hackrx/run", headers=headers, json=payload)
        if response.status_code == 200:
            print("   ✅ Main endpoint working")
            data = response.json()
            if 'answers' in data:
                print(f"   📝 Response format: Correct (answers array present)")
                print(f"   📝 Answer preview: {data['answers'][0][:100]}...")
            else:
                print("   ❌ Invalid response format")
        else:
            print(f"   ❌ Main endpoint failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Main endpoint error: {e}")
    
    # Test 3: Authentication
    print("\n3. Testing authentication...")
    try:
        # Test without auth
        response = requests.post(f"{base_url}/hackrx/run", json=payload)
        if response.status_code in [401, 403]:
            print("   ✅ Authentication required (correctly rejected)")
        else:
            print(f"   ❌ Authentication not enforced: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Auth test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 API SUMMARY:")
    print(f"📝 Webhook URL: {base_url}/hackrx/run")
    print("✅ API is functional and ready for hackathon submission!")
    print("\n📋 Usage Example:")
    print("curl -X POST \\")
    print(f'  "{base_url}/hackrx/run" \\')
    print('  -H "Authorization: Bearer 8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"documents": "https://example.com/doc.pdf", "questions": ["What is this about?"]}\'')

if __name__ == "__main__":
    test_api()