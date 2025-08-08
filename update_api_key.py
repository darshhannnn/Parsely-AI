"""
Script to securely update the Google API key in .env file
"""

import os
import re
from getpass import getpass

def validate_google_api_key(api_key):
    """Validate Google API key format"""
    if not api_key:
        return False, "API key is empty"
    
    # Google API keys start with AIza and are 39 characters total
    if not re.match(r'^AIza[0-9A-Za-z-_]{35}$', api_key):
        return False, "Invalid Google API key format. Should start with 'AIza' and be 39 characters long"
    
    # Check if it's not the old exposed key
    old_exposed_key = "AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y"
    if api_key == old_exposed_key:
        return False, "🚨 This is the old EXPOSED key! Please create a new one."
    
    return True, "Valid API key format"

def update_env_file(new_api_key):
    """Update the .env file with the new API key"""
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the API key line
        updated_content = re.sub(
            r'GOOGLE_API_KEY=.*',
            f'GOOGLE_API_KEY={new_api_key}',
            content
        )
        
        # Write back to .env file
        with open('.env', 'w') as f:
            f.write(updated_content)
        
        return True, "Successfully updated .env file"
        
    except Exception as e:
        return False, f"Error updating .env file: {str(e)}"

def main():
    """Main function to update API key"""
    print("🔑 Google API Key Update Tool")
    print("=" * 40)
    print()
    print("This tool will help you securely update your Google API key.")
    print("⚠️  Make sure you have:")
    print("   1. Created a new API key in Google Cloud Console")
    print("   2. Restricted it to only Gemini API access")
    print("   3. Revoked the old exposed key")
    print()
    
    # Get the new API key (hidden input for security)
    print("Please enter your new Google API key:")
    print("(Input will be hidden for security)")
    new_api_key = getpass("API Key: ").strip()
    
    # Validate the API key
    is_valid, message = validate_google_api_key(new_api_key)
    
    if not is_valid:
        print(f"❌ {message}")
        print()
        print("Please check your API key and try again.")
        print("Google API keys should:")
        print("  - Start with 'AIza'")
        print("  - Be exactly 39 characters long")
        print("  - Contain only letters, numbers, hyphens, and underscores")
        return False
    
    print(f"✅ {message}")
    
    # Show masked key for confirmation
    masked_key = new_api_key[:8] + "..." + new_api_key[-4:]
    print(f"Masked key: {masked_key}")
    
    # Confirm update
    confirm = input("\nUpdate .env file with this key? (yes/no): ").lower().strip()
    
    if confirm != 'yes':
        print("Update cancelled.")
        return False
    
    # Update the .env file
    success, message = update_env_file(new_api_key)
    
    if success:
        print(f"✅ {message}")
        print()
        print("Next steps:")
        print("1. Test the API key: python test_api_security.py")
        print("2. Start the hackathon API: python start_hackathon_api.py")
        print("3. Run validation: python validate_hackathon_requirements.py")
        return True
    else:
        print(f"❌ {message}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)