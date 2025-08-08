#!/usr/bin/env python3
"""
Direct API test to verify Gemini setup
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_api_direct():
    """Test Gemini API directly"""
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    model_name = os.getenv('LLM_MODEL', 'gemini-1.5-flash')
    
    print(f"🔑 API Key: {api_key[:10]}..." if api_key else "❌ No API Key")
    print(f"🤖 Model: {model_name}")
    
    if not api_key:
        print("❌ No API key found!")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel(model_name)
        
        # Test simple query
        print("\n🧪 Testing simple query...")
        response = model.generate_content("Say hello in 5 words")
        print(f"✅ Response: {response.text}")
        
        # Test entity extraction
        print("\n🧪 Testing entity extraction...")
        test_query = "46-year-old male, knee surgery in Pune, 3-month-old policy"
        response = model.generate_content(f"Extract age, gender, procedure, and location from: '{test_query}'")
        print(f"✅ Extraction: {response.text[:100]}...")
        
        print("\n🎉 API is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_direct()