#!/usr/bin/env python3
"""
Quick test to verify Gemini API key works
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def quick_test():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    model_name = os.getenv('LLM_MODEL', 'gemini-1.5-flash')
    
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"🤖 Model: {model_name}")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    try:
        # Test with Flash model specifically
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say hello in 5 words")
        
        print("✅ API Key is working!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    quick_test()