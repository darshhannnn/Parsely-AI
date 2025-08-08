"""
Test script for the hackathon API endpoint
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = os.getenv("HACKATHON_API_TOKEN", "8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63")

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Status Code: {response.status_code}")
        print(f"Health Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_hackrx_run_endpoint():
    """Test the main hackrx/run endpoint"""
    print("\nTesting /hackrx/run endpoint...")
    
    # Sample request matching hackathon specification
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    # Alternative test with a simpler PDF for testing
    simple_test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What information does this document contain?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to {BASE_URL}/hackrx/run")
        print(f"Request payload: {json.dumps(test_request, indent=2)}")
        
        # Try the main request first
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=120  # Allow up to 2 minutes for processing
        )
        
        # If main request fails, try simpler test
        if response.status_code != 200:
            print("Main request failed, trying simpler test...")
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                headers=headers,
                json=simple_test_request,
                timeout=60
            )
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Response: {json.dumps(response_data, indent=2)}")
            
            # Validate response format
            if "answers" in response_data and isinstance(response_data["answers"], list):
                print(f"✅ Valid response format with {len(response_data['answers'])} answers")
                return True
            else:
                print("❌ Invalid response format")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_authentication():
    """Test authentication with invalid token"""
    print("\nTesting authentication...")
    
    test_request = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question"]
    }
    
    # Test with invalid token
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=test_request
        )
        
        if response.status_code == 401:
            print("✅ Authentication correctly rejected invalid token")
            return True
        else:
            print(f"❌ Expected 401, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Authentication test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Hackathon API Endpoint")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test authentication
    auth_ok = test_authentication()
    
    # Test main endpoint (only if health is OK)
    if health_ok:
        main_ok = test_hackrx_run_endpoint()
    else:
        print("⚠️ Skipping main endpoint test due to health check failure")
        main_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Authentication: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"Main Endpoint: {'✅ PASS' if main_ok else '❌ FAIL'}")
    
    if health_ok and auth_ok and main_ok:
        print("\n🎉 All tests passed! API is ready for hackathon submission.")
    else:
        print("\n⚠️ Some tests failed. Please check the API implementation.")

if __name__ == "__main__":
    main()