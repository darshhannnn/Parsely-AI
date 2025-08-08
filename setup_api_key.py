#!/usr/bin/env python3
"""
Interactive script to help set up Gemini API key
"""

import os
import re

def setup_gemini_api_key():
    """Interactive setup for Gemini API key"""
    
    print("🔑 Gemini API Key Setup")
    print("=" * 30)
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ .env file not found!")
        print("Please make sure you have a .env file in your project root.")
        return False
    
    # Read current .env file
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    # Check if API key is already set
    current_key = None
    if 'GOOGLE_API_KEY=' in env_content:
        match = re.search(r'GOOGLE_API_KEY=(.+)', env_content)
        if match:
            current_key = match.group(1).strip()
    
    if current_key and current_key != 'your_gemini_api_key_here':
        print(f"✅ API key already configured: {current_key[:10]}...")
        
        update = input("Do you want to update it? (y/N): ").lower().strip()
        if update != 'y':
            print("Keeping existing API key.")
            return True
    
    print("\n📝 Please enter your Gemini API key:")
    print("You can get it from: https://makersuite.google.com/app/apikey")
    print()
    
    while True:
        api_key = input("Gemini API Key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty!")
            continue
        
        if len(api_key) < 20:
            print("❌ API key seems too short. Please check and try again.")
            continue
        
        if not api_key.startswith('AIza'):
            print("⚠️ Gemini API keys usually start with 'AIza'. Are you sure this is correct?")
            confirm = input("Continue anyway? (y/N): ").lower().strip()
            if confirm != 'y':
                continue
        
        break
    
    # Update .env file
    if current_key:
        # Replace existing key
        new_content = env_content.replace(f'GOOGLE_API_KEY={current_key}', f'GOOGLE_API_KEY={api_key}')
    else:
        # Add new key
        new_content = env_content.replace('GOOGLE_API_KEY=your_gemini_api_key_here', f'GOOGLE_API_KEY={api_key}')
    
    # Write updated .env file
    with open(env_file, 'w') as f:
        f.write(new_content)
    
    print(f"\n✅ API key saved to {env_file}")
    print(f"Key preview: {api_key[:10]}...")
    
    # Test the API key
    print("\n🧪 Testing API key...")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello, this is a test.")
        
        print("✅ API key is working!")
        print("🎉 Setup complete! You can now run: python test_gemini_setup.py")
        
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        print("Please check your API key and try again.")
        return False

def show_next_steps():
    """Show next steps after setup"""
    
    print("\n🚀 Next Steps:")
    print("-" * 20)
    print("1. Test your setup: python test_gemini_setup.py")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Add policy documents to data/policies/ directory")
    print("4. Run the system:")
    print("   - API: uvicorn src.api.main:app --reload")
    print("   - UI: streamlit run src/ui/streamlit_app.py")
    print("   - Direct: python -m src.insurance_claim_processor")

if __name__ == "__main__":
    if setup_gemini_api_key():
        show_next_steps()
    else:
        print("\n💥 Setup failed. Please try again.")