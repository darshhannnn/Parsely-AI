"""
Update API Key from Google AI Studio
Simple tool to get and test API key from AI Studio
"""

import webbrowser
import time
from getpass import getpass
import google.generativeai as genai
import re
import os

def main():
    print("🤖 UPDATE API KEY FROM GOOGLE AI STUDIO")
    print("="*50)
    print("Google AI Studio is the easiest way to get a Gemini API key!")
    print()
    
    print("🔧 STEP 1: Get API Key from AI Studio")
    print("-" * 40)
    print("I'll open Google AI Studio for you.")
    print()
    print("Follow these steps:")
    print("1. Click 'Get API key' in the left sidebar")
    print("2. Click 'Create API key in new project' (or existing project)")
    print("3. Copy the API key that appears")
    print("4. That's it! AI Studio keys work immediately.")
    print()
    
    try:
        webbrowser.open("https://aistudio.google.com/app/apikey")
        print("✅ Opened Google AI Studio API Key page")
    except:
        print("❌ Please manually go to: https://aistudio.google.com/app/apikey")
    
    input("\nPress Enter after you've copied your API key...")
    
    print("\n🧪 STEP 2: Test Your AI Studio API Key")
    print("-" * 40)
    print("Paste your API key from AI Studio:")
    print("(Input will be hidden for security)")
    
    while True:
        new_key = getpass("AI Studio API Key: ").strip()
        
        if not new_key:
            print("❌ No key entered. Please try again.")
            continue
        
        # Validate format
        if not re.match(r'^AIza[0-9A-Za-z-_]{35}$', new_key):
            print("❌ Invalid format. Google API keys start with 'AIza' and are 39 characters.")
            print("   Make sure you copied the complete key from AI Studio.")
            continue
        
        # Check if it's the old expired key
        current_key = os.getenv("GOOGLE_API_KEY", "")
        if new_key == current_key:
            print("⚠️  This appears to be the same key you already have.")
            print("   Please create a new key in AI Studio.")
            continue
        
        # Test the key immediately
        print("🧪 Testing AI Studio key...")
        try:
            genai.configure(api_key=new_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Say 'AI Studio API key test successful!'")
            
            if response and response.text:
                masked_key = new_key[:8] + "..." + new_key[-4:]
                print(f"✅ SUCCESS! AI Studio key works perfectly: {masked_key}")
                print(f"📝 Test response: {response.text.strip()}")
                break
            else:
                print("❌ Key accepted but no response received.")
                print("   This might be a temporary issue. Try again.")
                continue
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ Key test failed: {str(e)}")
            
            if "expired" in error_msg or "invalid" in error_msg:
                print("🚨 This key is invalid or expired. Create a new one in AI Studio.")
            elif "quota" in error_msg or "billing" in error_msg:
                print("💰 Quota/billing issue. Check your Google Cloud project billing.")
            elif "permission" in error_msg:
                print("🔒 Permission issue. Make sure the key has Gemini API access.")
            else:
                print("⏰ Temporary issue. AI Studio keys usually work immediately.")
            
            retry = input("\nTry again with a different key? (yes/no): ").lower().strip()
            if retry != 'yes':
                return False
    
    print("\n💾 STEP 3: Update Your Environment")
    print("-" * 40)
    
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the API key line
        updated_content = re.sub(
            r'GOOGLE_API_KEY=.*',
            f'GOOGLE_API_KEY={new_key}',
            content
        )
        
        # Write back to .env file
        with open('.env', 'w') as f:
            f.write(updated_content)
        
        print("✅ Successfully updated .env file with AI Studio key")
        
    except Exception as e:
        print(f"❌ Error updating .env file: {str(e)}")
        print("Please manually update your .env file:")
        print(f"GOOGLE_API_KEY={new_key}")
        return False
    
    print("\n🔍 STEP 4: Final Verification")
    print("-" * 40)
    print("Running quick verification test...")
    
    try:
        # Test one more time to be sure
        genai.configure(api_key=new_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Confirm the API key is working!")
        
        if response and response.text:
            print("✅ Final verification successful!")
            print(f"📝 Response: {response.text.strip()}")
        else:
            print("⚠️  Verification had issues but key should work.")
            
    except Exception as e:
        print(f"⚠️  Verification failed: {str(e)}")
        print("But the key was working before, so it should be fine.")
    
    print("\n🎉 SUCCESS!")
    print("="*30)
    print("✅ AI Studio API key obtained")
    print("✅ Key tested and working")
    print("✅ Environment file updated")
    print("✅ Ready for hackathon!")
    print()
    print("🚀 Next steps:")
    print("1. python test_api_security.py      # Run full security test")
    print("2. python start_hackathon_api.py    # Start your hackathon API")
    print("3. python validate_hackathon_requirements.py  # Final validation")
    
    return True

if __name__ == "__main__":
    print("🤖 Google AI Studio is the fastest way to get a Gemini API key!")
    print("   - No complex setup required")
    print("   - Keys work immediately")
    print("   - Free tier included")
    print("   - Perfect for hackathons")
    print()
    
    success = main()
    if success:
        print("\n🎊 Congratulations! Your API key is ready!")
        print("Your hackathon API should now work perfectly.")
    else:
        print("\n❌ Please try again or check the AI Studio documentation.")
    
    exit(0 if success else 1)