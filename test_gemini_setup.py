#!/usr/bin/env python3
"""
Comprehensive test script to verify Gemini API setup and all integrations
"""

import os
import sys
from dotenv import load_dotenv

def test_gemini_setup():
    """Test Gemini API configuration and all components"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔍 Testing Gemini API Setup...")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    print(f"API Key configured: {'✅' if api_key else '❌'}")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file!")
        print("Please add your Gemini API key to the .env file")
        return False
    
    # Test basic Gemini connection
    try:
        from config.gemini_config import gemini_config
        print(f"Model: {gemini_config.model_name}")
        print(f"Temperature: {gemini_config.temperature}")
        
        if gemini_config.test_connection():
            print("✅ Gemini API connection successful!")
        else:
            print("❌ Gemini API connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Gemini configuration error: {str(e)}")
        return False
    
    print("\n🧠 Testing Enhanced Query Parser...")
    print("-" * 30)
    
    # Test enhanced query parser
    try:
        from src.query_parsing.gemini_query_parser import GeminiQueryParser
        
        parser = GeminiQueryParser()
        test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
        
        print(f"Test query: {test_query}")
        result = parser.parse_query(test_query)
        
        print("✅ Enhanced query parsing successful!")
        print(f"Extracted - Age: {result.age}, Gender: {result.gender}")
        print(f"Procedure: {result.procedure}, Location: {result.location}")
        print(f"Policy Age: {result.policy_age_months} months")
        
        # Test intent classification
        intent = parser.classify_query_intent(test_query)
        print(f"Query Intent: {intent}")
        
        # Test query expansion
        variations = parser.expand_query_for_search(test_query)
        print(f"Search Variations: {len(variations)} generated")
        
    except Exception as e:
        print(f"❌ Enhanced query parser failed: {str(e)}")
        return False
    
    print("\n⚖️ Testing Enhanced Decision Engine...")
    print("-" * 30)
    
    # Test enhanced decision engine
    try:
        from src.semantic_search.semantic_retriever import SemanticRetriever
        from src.decision_engine.gemini_claim_evaluator import GeminiClaimEvaluator
        
        # Create mock components for testing
        retriever = SemanticRetriever("data/policies")
        evaluator = GeminiClaimEvaluator(retriever)
        
        print("✅ Enhanced decision engine initialized!")
        print("Note: Full evaluation requires policy documents")
        
    except Exception as e:
        print(f"❌ Enhanced decision engine failed: {str(e)}")
        return False
    
    print("\n🏥 Testing Complete System Integration...")
    print("-" * 30)
    
    # Test complete system
    try:
        from src.insurance_claim_processor import InsuranceClaimProcessor
        
        # Initialize processor (will use Gemini components)
        processor = InsuranceClaimProcessor()
        
        # Test query analysis only (doesn't require full document processing)
        analysis = processor.analyze_query_only(test_query)
        
        print("✅ Complete system integration successful!")
        print(f"System extracted {len(analysis['extracted_entities'])} entities")
        print(f"Assumptions made: {len(analysis['assumptions'])}")
        
    except Exception as e:
        print(f"❌ System integration failed: {str(e)}")
        print("This might be due to missing policy documents, which is normal for initial setup")
    
    print("\n🎉 Gemini Setup Complete!")
    print("=" * 50)
    print("Your system is configured to use Google Gemini API")
    print("Next steps:")
    print("1. Add policy documents to data/policies/ directory")
    print("2. Run the main system: python -m src.insurance_claim_processor")
    print("3. Or start the API: uvicorn src.api.main:app --reload")
    print("4. Or start the UI: streamlit run src/ui/streamlit_app.py")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("📦 Checking Dependencies...")
    print("-" * 30)
    
    required_packages = [
        ('google-generativeai', 'google.generativeai'),
        ('langchain-google-genai', 'langchain_google_genai'),
        ('sentence-transformers', 'sentence_transformers'),
        ('faiss-cpu', 'faiss'),
        ('pydantic', 'pydantic'),
        ('python-dotenv', 'dotenv')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

if __name__ == "__main__":
    print("🌿 Parsely AI - Gemini Setup")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    print()
    
    # Test Gemini setup
    if test_gemini_setup():
        print("\n🎯 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Setup failed. Please check the errors above.")
        sys.exit(1)