"""
Quick security check for the current API key setup
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("🔍 Quick Security Check")
    print("=" * 30)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No GOOGLE_API_KEY found in environment")
        return False
    
    if api_key == "your_new_api_key_here":
        print("⚠️  API key is still the placeholder value")
        print("📝 Please update your .env file with a real Google API key")
        print("💡 Run: python update_api_key.py")
        return False
    
    # Check if it's the old exposed key
    old_exposed_key = "AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y"
    if api_key == old_exposed_key:
        print("🚨 CRITICAL: Still using the EXPOSED API key!")
        print("🔥 IMMEDIATE ACTION REQUIRED:")
        print("   1. Go to Google Cloud Console")
        print("   2. REVOKE this key immediately")
        print("   3. Create a new restricted API key")
        print("   4. Update your .env file")
        return False
    
    # Check format
    if re.match(r'^AIza[0-9A-Za-z-_]{35}$', api_key):
        masked_key = api_key[:8] + "..." + api_key[-4:]
        print(f"✅ Valid Google API key format detected: {masked_key}")
        print("🧪 Run full security test: python test_api_security.py")
        return True
    else:
        print("❌ Invalid API key format")
        print("💡 Google API keys should start with 'AIza' and be 39 characters long")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)