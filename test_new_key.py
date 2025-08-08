"""
Test a new Google API key before updating .env file
"""

import google.generativeai as genai
from getpass import getpass
import re

def test_api_key(api_key):
    """Test if an API key works with Gemini"""
    try:
        # Configure Gemini with the test key
        genai.configure(api_key=api_key)
        
        # Try to create a model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test basic generation
        response = model.generate_content("Say 'Hello, this API key works!'")
        
        if response and response.text:
            return True, response.text.strip()
        else:
            return False, "No response received"
            
    except Exception as e:
        return False, str(e)

def main():
    print("🧪 API Key Tester")
    print("=" * 25)
    print("Test a Google API key before updating your .env file")
    print()
    
    # Get API key to test
    print("Enter the API key to test:")
    print("(Input will be hidden for security)")
    test_key = getpass("API Key: ").strip()
    
    # Validate format
    if not re.match(r'^AIza[0-9A-Za-z-_]{35}$', test_key):
        print("❌ Invalid API key format")
        print("Google API keys should start with 'AIza' and be 39 characters long")
        return False
    
    # Check if it's the old exposed key
    old_exposed_key = "AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y"
    if test_key == old_exposed_key:
        print("🚨 This is the old EXPOSED key! Don't use it!")
        return False
    
    print("🧪 Testing API key...")
    
    # Test the key
    works, message = test_api_key(test_key)
    
    if works:
        masked_key = test_key[:8] + "..." + test_key[-4:]
        print(f"✅ API key works! ({masked_key})")
        print(f"Response: {message}")
        print()
        print("This key is ready to use!")
        print("Update your .env file: python update_api_key.py")
        return True
    else:
        print(f"❌ API key failed: {message}")
        print()
        print("Please check:")
        print("1. The key is correct and not expired")
        print("2. Generative Language API is enabled in your project")
        print("3. The key has proper permissions")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)