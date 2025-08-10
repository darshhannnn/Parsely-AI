#!/usr/bin/env python3
"""
Script to update Google API key in both local and Railway environments
"""
import os
import subprocess
import sys

def update_local_env(api_key):
    """Update the local .env file"""
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the API key
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('GOOGLE_API_KEY='):
                lines[i] = f'GOOGLE_API_KEY={api_key}'
                break
        
        # Write back to file
        with open('.env', 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ Local .env file updated successfully")
        return True
    except Exception as e:
        print(f"❌ Error updating local .env: {e}")
        return False

def update_railway_env(api_key):
    """Update Railway environment variable"""
    try:
        cmd = ['npx', '@railway/cli', 'variables', '--set', f'GOOGLE_API_KEY={api_key}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Railway environment variable updated successfully")
            print("🔄 Railway will automatically redeploy with the new key")
            return True
        else:
            print(f"❌ Error updating Railway: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running Railway CLI: {e}")
        return False

def validate_api_key(api_key):
    """Basic validation of API key format"""
    if not api_key:
        return False, "API key is empty"
    
    if not api_key.startswith('AIzaSy'):
        return False, "Google API keys should start with 'AIzaSy'"
    
    if len(api_key) < 35:
        return False, "API key seems too short"
    
    return True, "API key format looks valid"

def test_api_key(api_key):
    """Test the API key with a simple request"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test
        response = model.generate_content("Say 'API key is working'")
        if response and response.text:
            print("✅ API key test successful!")
            return True
        else:
            print("❌ API key test failed - no response")
            return False
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    print("🔑 Google API Key Updater")
    print("=" * 40)
    
    # Get API key from user
    api_key = input("Enter your new Google API key: ").strip()
    
    # Validate API key
    is_valid, message = validate_api_key(api_key)
    if not is_valid:
        print(f"❌ {message}")
        sys.exit(1)
    
    print(f"✅ {message}")
    
    # Test API key
    print("\n🧪 Testing API key...")
    if not test_api_key(api_key):
        response = input("API key test failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Update local environment
    print("\n📝 Updating local environment...")
    if not update_local_env(api_key):
        sys.exit(1)
    
    # Update Railway environment
    print("\n☁️ Updating Railway environment...")
    if not update_railway_env(api_key):
        print("⚠️ Railway update failed, but local environment is updated")
        print("You can manually update Railway with:")
        print(f"npx @railway/cli variables --set \"GOOGLE_API_KEY={api_key}\"")
    
    print("\n🎉 API key update complete!")
    print("\n📋 Next steps:")
    print("1. Wait for Railway to redeploy (1-2 minutes)")
    print("2. Test your API with: python test_api_simple.py")
    print("3. Or open web_interface.html to test in browser")

if __name__ == "__main__":
    main()