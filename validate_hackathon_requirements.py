"""
Comprehensive validation script for hackathon requirements
"""

import requests
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "https://parsely-ai-production-f2ad.up.railway.app"
BEARER_TOKEN = os.getenv("HACKATHON_API_TOKEN", "8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63")

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def print_result(test_name, passed, details=""):
    """Print test result with formatting"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")

def validate_api_structure():
    """Validate basic API structure and endpoints"""
    print_section("API Structure Validation")
    
    results = {}
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        health_ok = response.status_code == 200
        print_result("Health endpoint accessible", health_ok)
        results['health'] = health_ok
        
        if health_ok:
            health_data = response.json()
            has_pipeline_stages = 'pipeline_stages' in health_data
            print_result("Pipeline stages documented", has_pipeline_stages)
            results['pipeline_stages'] = has_pipeline_stages
            
            if has_pipeline_stages:
                stages = health_data['pipeline_stages']
                expected_stages = [
                    "Input Documents", "LLM Parser", "Embedding Search",
                    "Clause Matching", "Logic Evaluation", "JSON Output"
                ]
                all_stages_present = all(stage in stages for stage in expected_stages)
                print_result("All 6 pipeline stages present", all_stages_present)
                results['all_stages'] = all_stages_present
    except Exception as e:
        print_result("Health endpoint accessible", False, str(e))
        results['health'] = False
    
    # Test API documentation
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        docs_ok = response.status_code == 200
        print_result("API documentation accessible", docs_ok)
        results['docs'] = docs_ok
    except Exception as e:
        print_result("API documentation accessible", False, str(e))
        results['docs'] = False
    
    return results

def validate_authentication():
    """Validate authentication requirements"""
    print_section("Authentication Validation")
    
    results = {}
    
    test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": ["Test question"]
    }
    
    # Test missing authentication
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_request, timeout=10)
        missing_auth_rejected = response.status_code in [401, 403]
        print_result("Missing authentication rejected", missing_auth_rejected)
        results['missing_auth'] = missing_auth_rejected
    except Exception as e:
        print_result("Missing authentication test", False, str(e))
        results['missing_auth'] = False
    
    # Test invalid token
    try:
        headers = {"Authorization": "Bearer invalid_token"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_request, timeout=10)
        invalid_token_rejected = response.status_code == 401
        print_result("Invalid token rejected", invalid_token_rejected)
        results['invalid_token'] = invalid_token_rejected
    except Exception as e:
        print_result("Invalid token test", False, str(e))
        results['invalid_token'] = False
    
    # Test valid token format
    try:
        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_request, timeout=30)
        valid_token_accepted = response.status_code != 401
        print_result("Valid token accepted", valid_token_accepted)
        results['valid_token'] = valid_token_accepted
    except Exception as e:
        print_result("Valid token test", False, str(e))
        results['valid_token'] = False
    
    return results

def validate_request_format():
    """Validate request format requirements"""
    print_section("Request Format Validation")
    
    results = {}
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    # Test missing documents field
    try:
        invalid_request = {"questions": ["Test question"]}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=invalid_request, timeout=10)
        missing_docs_rejected = response.status_code == 422
        print_result("Missing documents field rejected", missing_docs_rejected)
        results['missing_docs'] = missing_docs_rejected
    except Exception as e:
        print_result("Missing documents test", False, str(e))
        results['missing_docs'] = False
    
    # Test missing questions field
    try:
        invalid_request = {"documents": "https://example.com/test.pdf"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=invalid_request, timeout=10)
        missing_questions_rejected = response.status_code == 422
        print_result("Missing questions field rejected", missing_questions_rejected)
        results['missing_questions'] = missing_questions_rejected
    except Exception as e:
        print_result("Missing questions test", False, str(e))
        results['missing_questions'] = False
    
    # Test valid request format
    try:
        valid_request = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": ["What is this document about?"]
        }
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=valid_request, timeout=30)
        valid_format_accepted = response.status_code == 200
        print_result("Valid request format accepted", valid_format_accepted)
        results['valid_format'] = valid_format_accepted
    except Exception as e:
        print_result("Valid request format test", False, str(e))
        results['valid_format'] = False
    
    return results

def validate_response_format():
    """Validate response format requirements"""
    print_section("Response Format Validation")
    
    results = {}
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What type of document is this?"
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_request, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if response has answers field
            has_answers_field = 'answers' in data
            print_result("Response has 'answers' field", has_answers_field)
            results['has_answers'] = has_answers_field
            
            if has_answers_field:
                answers = data['answers']
                
                # Check if answers is a list
                answers_is_list = isinstance(answers, list)
                print_result("Answers field is a list", answers_is_list)
                results['answers_is_list'] = answers_is_list
                
                # Check if number of answers matches number of questions
                correct_answer_count = len(answers) == len(test_request['questions'])
                print_result("Answer count matches question count", correct_answer_count)
                results['correct_count'] = correct_answer_count
                
                # Check if all answers are strings
                all_strings = all(isinstance(answer, str) for answer in answers)
                print_result("All answers are strings", all_strings)
                results['all_strings'] = all_strings
                
                # Check if answers are non-empty
                non_empty_answers = all(answer.strip() for answer in answers)
                print_result("All answers are non-empty", non_empty_answers)
                results['non_empty'] = non_empty_answers
                
        else:
            print_result("Response format test", False, f"HTTP {response.status_code}")
            results['response_success'] = False
            
    except Exception as e:
        print_result("Response format test", False, str(e))
        results['response_test'] = False
    
    return results

def validate_document_processing():
    """Validate document processing capabilities"""
    print_section("Document Processing Validation")
    
    results = {}
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
    # Test PDF blob URL processing
    test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": ["What is the content of this document?"]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_request, timeout=120)
        processing_time = time.time() - start_time
        
        pdf_processing_works = response.status_code == 200
        print_result("PDF blob URL processing works", pdf_processing_works)
        results['pdf_processing'] = pdf_processing_works
        
        within_time_limit = processing_time < 60  # Should be under 60 seconds
        print_result(f"Processing time acceptable ({processing_time:.1f}s)", within_time_limit)
        results['processing_time'] = within_time_limit
        
        if pdf_processing_works:
            data = response.json()
            meaningful_answer = len(data['answers'][0]) > 10  # Answer should be substantial
            print_result("Answer is meaningful", meaningful_answer)
            results['meaningful_answer'] = meaningful_answer
            
    except Exception as e:
        print_result("PDF processing test", False, str(e))
        results['pdf_processing'] = False
    
    return results

def validate_technical_requirements():
    """Validate technical implementation requirements"""
    print_section("Technical Requirements Validation")
    
    results = {}
    
    # Check if required environment variables are set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    has_google_key = bool(google_api_key)
    print_result("Google API key configured", has_google_key)
    results['google_api_key'] = has_google_key
    
    # Check if hackathon token is set
    hackathon_token = os.getenv("HACKATHON_API_TOKEN")
    has_hackathon_token = bool(hackathon_token)
    print_result("Hackathon API token configured", has_hackathon_token)
    results['hackathon_token'] = has_hackathon_token
    
    # Test if server supports required content type
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    test_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": ["Test"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=headers, json=test_request, timeout=30)
        supports_json = response.status_code != 415  # Not "Unsupported Media Type"
        print_result("Supports application/json content type", supports_json)
        results['supports_json'] = supports_json
    except Exception as e:
        print_result("JSON content type test", False, str(e))
        results['supports_json'] = False
    
    return results

def generate_summary_report(all_results):
    """Generate a comprehensive summary report"""
    print_section("HACKATHON REQUIREMENTS SUMMARY")
    
    # Count total tests and passes
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        for test, passed in results.items():
            total_tests += 1
            if passed:
                passed_tests += 1
    
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
    
    # Critical requirements check
    critical_requirements = [
        ('API Structure', 'health', all_results.get('api_structure', {}).get('health', False)),
        ('Authentication', 'valid_token', all_results.get('authentication', {}).get('valid_token', False)),
        ('Request Format', 'valid_format', all_results.get('request_format', {}).get('valid_format', False)),
        ('Response Format', 'has_answers', all_results.get('response_format', {}).get('has_answers', False)),
        ('Document Processing', 'pdf_processing', all_results.get('document_processing', {}).get('pdf_processing', False)),
        ('Technical Setup', 'google_api_key', all_results.get('technical', {}).get('google_api_key', False)),
    ]
    
    print("\n🎯 Critical Requirements Status:")
    all_critical_passed = True
    for category, test, passed in critical_requirements:
        status = "✅" if passed else "❌"
        print(f"  {status} {category}: {test}")
        if not passed:
            all_critical_passed = False
    
    print(f"\n🏆 Hackathon Readiness: {'READY' if all_critical_passed else 'NEEDS WORK'}")
    
    if all_critical_passed:
        print("🎉 Your API meets all critical hackathon requirements!")
        print("📝 You can submit your webhook URL for evaluation.")
    else:
        print("⚠️  Please address the failed critical requirements before submission.")
    
    return all_critical_passed

def main():
    """Run all validation tests"""
    print("🧪 HACKATHON REQUIREMENTS VALIDATION")
    print("=" * 60)
    print("Testing API compliance with hackathon specifications...")
    
    all_results = {}
    
    # Run all validation tests
    all_results['api_structure'] = validate_api_structure()
    all_results['authentication'] = validate_authentication()
    all_results['request_format'] = validate_request_format()
    all_results['response_format'] = validate_response_format()
    all_results['document_processing'] = validate_document_processing()
    all_results['technical'] = validate_technical_requirements()
    
    # Generate summary
    ready_for_submission = generate_summary_report(all_results)
    
    return ready_for_submission

if __name__ == "__main__":
    ready = main()
    exit(0 if ready else 1)