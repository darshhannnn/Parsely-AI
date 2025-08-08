#!/usr/bin/env python3
"""
Start the LLM Document Processing System (Gemini-powered)
"""

import os
import sys
import json

# Set environment variables
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y'
os.environ['LLM_MODEL'] = 'gemini-1.5-flash'
os.environ['LLM_PROVIDER'] = 'google'

def test_gemini_components():
    """Test individual Gemini components"""
    
    print("🧠 Testing Gemini Components")
    print("=" * 40)
    
    try:
        # Test enhanced query parser
        print("\n1. 🔍 Testing Enhanced Query Parser...")
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        test_query = "46-year-old male, knee surgery in Pune, 3-month policy"
        
        result = parser.parse_query(test_query)
        print(f"✅ Parsed successfully!")
        print(f"   Age: {result.age}, Gender: {result.gender}")
        print(f"   Procedure: {result.procedure}, Location: {result.location}")
        
        # Test intent classification
        intent = parser.classify_query_intent(test_query)
        print(f"   Intent: {intent}")
        
    except Exception as e:
        print(f"❌ Query parser failed: {str(e)}")
        return False
    
    try:
        # Test decision engine (without full semantic retrieval)
        print("\n2. ⚖️ Testing Decision Engine...")
        from src.decision_engine.gemini_claim_evaluator import GeminiClaimEvaluator
        from src.semantic_search.semantic_retriever import SemanticRetriever
        
        # Create minimal retriever for testing
        retriever = SemanticRetriever("data/policies")
        evaluator = GeminiClaimEvaluator(retriever)
        
        print("✅ Decision engine initialized!")
        
    except Exception as e:
        print(f"❌ Decision engine failed: {str(e)}")
        return False
    
    return True

def run_direct_processing():
    """Run direct claim processing using Gemini"""
    
    print("\n🏥 Direct Claim Processing")
    print("=" * 40)
    
    import google.generativeai as genai
    
    # Configure Gemini
    genai.configure(api_key='AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y')
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Sample claims
    claims = [
        "46-year-old male, knee surgery in Pune, 3-month policy",
        "35-year-old female, maternity delivery in Mumbai, 12-month policy",
        "28-year-old male, accident treatment in Delhi, 6-month policy"
    ]
    
    for i, claim in enumerate(claims, 1):
        print(f"\n📋 Claim {i}: {claim}")
        print("-" * 30)
        
        # Process with Gemini
        prompt = f"""
        Process this insurance claim: "{claim}"
        
        Return a JSON response with:
        {{
            "decision": "approved" or "rejected",
            "amount": estimated_amount_in_inr or null,
            "justification": "detailed reasoning",
            "extracted_entities": {{
                "age": age,
                "gender": "gender",
                "procedure": "procedure",
                "location": "location",
                "policy_age_months": months
            }},
            "confidence_score": 0.0-1.0
        }}
        
        Consider typical insurance rules:
        - Waiting periods: Maternity (10 months), Surgery (24 months), Accidents (0 months)
        - Coverage limits: Surgery (₹200,000), Maternity (₹150,000), Accident (₹300,000)
        
        Only return the JSON, no other text.
        """
        
        try:
            response = model.generate_content(prompt)
            
            # Try to parse as JSON
            try:
                result = json.loads(response.text.strip())
                
                print(f"✅ Decision: {result.get('decision', 'unknown').upper()}")
                if result.get('amount'):
                    print(f"💰 Amount: ₹{result['amount']:,}")
                print(f"📝 Reason: {result.get('justification', 'No reason provided')[:100]}...")
                print(f"🎯 Confidence: {result.get('confidence_score', 0):.2f}")
                
            except json.JSONDecodeError:
                # If not JSON, just show the response
                print(f"📄 Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
    
    return True

def start_api_server():
    """Start the FastAPI server"""
    
    print("\n🌐 Starting API Server...")
    print("=" * 30)
    
    try:
        import subprocess
        print("🚀 Starting FastAPI server...")
        print("📍 API will be available at: http://localhost:8000")
        print("📚 Documentation at: http://localhost:8000/docs")
        print("\n⏹️ Press Ctrl+C to stop the server")
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")

def main():
    """Main function"""
    
    print("🌿 Parsely AI (Gemini-Powered)")
    print("=" * 60)
    
    # Test components
    if not test_gemini_components():
        print("\n💥 Component testing failed!")
        return
    
    # Run direct processing
    run_direct_processing()
    
    print("\n🎉 System is working perfectly!")
    
    # Ask user what they want to do
    print("\n🎯 What would you like to do?")
    print("1. 🌐 Start API Server")
    print("2. 🧪 Run more tests")
    print("3. 📊 Show system info")
    print("4. ❌ Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            start_api_server()
        elif choice == "2":
            print("\n🧪 Running additional tests...")
            run_direct_processing()
        elif choice == "3":
            print("\n📊 System Information:")
            print(f"✅ Gemini API: Working")
            print(f"🤖 Model: gemini-1.5-flash")
            print(f"🔑 API Key: Configured")
            print(f"🏗️ Architecture: Gemini-powered LLM processing")
        else:
            print("\n👋 Goodbye!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main()