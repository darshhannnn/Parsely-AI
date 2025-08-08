#!/usr/bin/env python3
"""
Simple test to run the system with working API key
"""

import os
import sys
import google.generativeai as genai

# Set the API key directly
API_KEY = "AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y"
MODEL_NAME = "gemini-1.5-flash"

def test_basic_functionality():
    """Test basic Gemini functionality"""
    
    print("🌿 Testing Parsely AI")
    print("=" * 50)
    
    try:
        # Configure Gemini
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        
        print(f"✅ Using model: {MODEL_NAME}")
        print(f"✅ API Key: {API_KEY[:10]}...")
        
        # Test 1: Basic response
        print("\n🧪 Test 1: Basic Response")
        response = model.generate_content("Say hello")
        print(f"Response: {response.text}")
        
        # Test 2: Entity extraction
        print("\n🧪 Test 2: Entity Extraction")
        query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        prompt = f"""
        Extract structured information from this insurance query: "{query}"
        
        Return in this format:
        Age: [age]
        Gender: [gender]  
        Procedure: [procedure]
        Location: [location]
        Policy Age: [months]
        """
        
        response = model.generate_content(prompt)
        print(f"Extraction Result:\n{response.text}")
        
        # Test 3: Decision making
        print("\n🧪 Test 3: Simple Decision Making")
        decision_prompt = f"""
        Based on this insurance claim: "{query}"
        
        Assume:
        - Knee surgery is covered
        - No waiting period for accidents
        - Policy is valid
        
        Should this claim be approved or rejected? Give a brief reason.
        """
        
        response = model.generate_content(decision_prompt)
        print(f"Decision: {response.text}")
        
        print("\n🎉 All tests passed! Your system is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def run_sample_claim():
    """Run a complete sample claim processing"""
    
    print("\n" + "="*60)
    print("🏥 SAMPLE INSURANCE CLAIM PROCESSING")
    print("="*60)
    
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Sample claim
        claim_query = "35-year-old female, maternity delivery in Mumbai, 12-month policy"
        
        print(f"📋 Claim: {claim_query}")
        
        # Process the claim
        processing_prompt = f"""
        Process this insurance claim: "{claim_query}"
        
        Analyze:
        1. Extract key information (age, gender, procedure, location, policy age)
        2. Check coverage (maternity is typically covered after 10 months waiting period)
        3. Make a decision (approved/rejected)
        4. Provide justification
        
        Format your response as:
        
        CLAIM ANALYSIS:
        - Age: [age]
        - Gender: [gender]
        - Procedure: [procedure]
        - Location: [location]
        - Policy Age: [months]
        
        DECISION: [APPROVED/REJECTED]
        
        JUSTIFICATION: [detailed reason]
        
        AMOUNT: [if approved, estimated coverage amount in INR]
        """
        
        response = model.generate_content(processing_prompt)
        print("\n" + response.text)
        
        print("\n✅ Sample claim processed successfully!")
        
    except Exception as e:
        print(f"❌ Sample processing failed: {str(e)}")

if __name__ == "__main__":
    if test_basic_functionality():
        run_sample_claim()
        
        print("\n🚀 Next Steps:")
        print("1. Your Gemini API is working perfectly!")
        print("2. You can now run the full system")
        print("3. Try: python -m src.insurance_claim_processor")
        print("4. Or start the API: uvicorn src.api.main:app --reload")
    else:
        print("\n💥 Setup incomplete. Please check the errors above.")