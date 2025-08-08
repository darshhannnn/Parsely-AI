#!/usr/bin/env python3
"""
Test Gemini 2.0 Flash model
"""

import requests
import json
import os
from dotenv import load_dotenv

def test_gemini_2_api():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Test with the exact endpoint format you mentioned
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': api_key
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Extract entities from this insurance query: '46-year-old male, knee surgery in Pune, 3-month-old policy'"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        print(f"🔍 Testing Gemini 2.0 Flash...")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Gemini 2.0 Flash is working!")
            print(f"Response: {result['candidates'][0]['content']['parts'][0]['text']}")
            return True
        else:
            print(f"❌ API Error: {response.text}")
            # Try the experimental version
            print("\n🔄 Trying gemini-2.0-flash-exp...")
            return test_experimental_version(api_key)
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_experimental_version(api_key):
    """Test the experimental version"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': api_key
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Hello from Gemini 2.0!"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Gemini 2.0 Flash Experimental is working!")
            print(f"Response: {result['candidates'][0]['content']['parts'][0]['text']}")
            return True
        else:
            print(f"❌ Experimental API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Experimental request failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_2_api()