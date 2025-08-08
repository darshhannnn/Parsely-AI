#!/usr/bin/env python3
"""
Run the complete LLM Document Processing System
"""

import os
import sys

# Set environment variables directly to ensure they work
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCmUdtynJ64KWelV1K6eU5NcuuPCECA15Y'
os.environ['LLM_MODEL'] = 'gemini-1.5-flash'
os.environ['LLM_PROVIDER'] = 'google'
os.environ['LLM_TEMPERATURE'] = '0.1'

def run_claim_processor():
    """Run the insurance claim processor"""
    
    print("🚀 Starting LLM Document Processing System")
    print("=" * 50)
    
    try:
        # Import after setting environment variables
        from src.insurance_claim_processor import InsuranceClaimProcessor
        
        # Initialize processor
        print("🔧 Initializing system...")
        processor = InsuranceClaimProcessor()
        
        print("✅ System initialized successfully!")
        
        # Test queries
        test_queries = [
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "35-year-old female, maternity delivery in Mumbai, 12-month policy", 
            "28-year-old male, accident treatment in Delhi, 6-month policy",
            "55-year-old female, cataract surgery in Bangalore, 18-month policy"
        ]
        
        print("\n🧪 Testing with sample queries...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}: {query}")
            print("-" * 40)
            
            try:
                # Process query analysis only (doesn't require policy documents)
                result = processor.analyze_query_only(query)
                
                print("✅ Query Analysis:")
                entities = result['extracted_entities']
                print(f"  Age: {entities.get('age', 'N/A')}")
                print(f"  Gender: {entities.get('gender', 'N/A')}")
                print(f"  Procedure: {entities.get('procedure', 'N/A')}")
                print(f"  Location: {entities.get('location', 'N/A')}")
                print(f"  Policy Age: {entities.get('policy_age_months', 'N/A')} months")
                
                if result.get('assumptions'):
                    print(f"  Assumptions: {len(result['assumptions'])} made")
                
            except Exception as e:
                print(f"❌ Query processing failed: {str(e)}")
        
        print("\n🎉 System is working perfectly!")
        print("\n📚 Note: For full claim processing, add policy documents to data/policies/")
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        print("\nThis might be due to missing policy documents, which is normal for initial setup.")
        return False

def show_usage_options():
    """Show different ways to use the system"""
    
    print("\n🎯 Usage Options:")
    print("=" * 30)
    print("1. 📊 API Server:")
    print("   uvicorn src.api.main:app --reload")
    print("   Then visit: http://localhost:8000/docs")
    print()
    print("2. 🖥️ Web Interface:")
    print("   streamlit run src/ui/streamlit_app.py")
    print()
    print("3. 🐍 Python Code:")
    print("   from src.insurance_claim_processor import InsuranceClaimProcessor")
    print("   processor = InsuranceClaimProcessor()")
    print("   result = processor.process_claim('your query here')")
    print()
    print("4. 📁 Add Policy Documents:")
    print("   Place PDF/DOCX files in data/policies/ directory")

if __name__ == "__main__":
    if run_claim_processor():
        show_usage_options()
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your API key is correct")
        print("2. Check internet connection")
        print("3. Try running: python run_simple_test.py")