"""
Enhanced Query Parser using Gemini API for better entity extraction
"""

import re
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
from config.gemini_config import gemini_config
from .query_parser import ClaimQuery, QueryParser

class GeminiQueryParser(QueryParser):
    """Enhanced query parser using Gemini for intelligent entity extraction"""
    
    def __init__(self):
        super().__init__()
        self.gemini_client = gemini_config.get_native_client()
        
    def parse_query(self, query: str) -> ClaimQuery:
        """Parse query using both traditional methods and Gemini AI"""
        
        # First, use traditional parsing as fallback
        traditional_result = super().parse_query(query)
        
        # Enhance with Gemini-based extraction
        try:
            gemini_result = self._parse_with_gemini(query)
            enhanced_result = self._merge_results(traditional_result, gemini_result)
            return enhanced_result
        except Exception as e:
            print(f"Gemini parsing failed, using traditional method: {str(e)}")
            return traditional_result
    
    def _parse_with_gemini(self, query: str) -> Dict[str, Any]:
        """Use Gemini to extract structured information from query"""
        
        prompt = f"""
        Extract structured information from this insurance claim query: "{query}"
        
        Return a JSON object with these fields (use null for missing information):
        {{
            "age": number or null,
            "gender": "male" or "female" or null,
            "procedure": "specific medical procedure" or null,
            "location": "city name" or null,
            "policy_age_months": number or null,
            "hospital": "hospital name" or null,
            "amount_claimed": number or null,
            "date": "date string" or null,
            "complications": ["list of complications"] or null,
            "confidence": {{
                "age": 0.0-1.0,
                "gender": 0.0-1.0,
                "procedure": 0.0-1.0,
                "location": 0.0-1.0,
                "policy_age_months": 0.0-1.0
            }},
            "assumptions": ["list of assumptions made"]
        }}
        
        Examples:
        - "46-year-old male, knee surgery in Pune, 3-month-old policy" → age: 46, gender: "male", procedure: "knee surgery", location: "Pune", policy_age_months: 3
        - "35F, maternity delivery Mumbai, 12 month policy" → age: 35, gender: "female", procedure: "maternity delivery", location: "Mumbai", policy_age_months: 12
        
        Only return the JSON object, no other text.
        """
        
        response = self.gemini_client.generate_content(prompt)
        
        # Parse JSON response
        try:
            # Clean the response text
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini JSON response: {e}")
            print(f"Response was: {response.text}")
            return {}
    
    def _merge_results(self, traditional: ClaimQuery, gemini_data: Dict[str, Any]) -> ClaimQuery:
        """Merge traditional parsing with Gemini results, preferring higher confidence"""
        
        # Start with traditional result
        result = traditional.model_copy()
        
        # Get confidence scores from Gemini
        gemini_confidence = gemini_data.get('confidence', {})
        
        # Merge fields based on confidence and availability
        if gemini_data.get('age') and gemini_confidence.get('age', 0) > 0.7:
            result.age = gemini_data['age']
        
        if gemini_data.get('gender') and gemini_confidence.get('gender', 0) > 0.7:
            result.gender = gemini_data['gender']
        
        if gemini_data.get('procedure') and gemini_confidence.get('procedure', 0) > 0.7:
            result.procedure = gemini_data['procedure']
        
        if gemini_data.get('location') and gemini_confidence.get('location', 0) > 0.7:
            result.location = gemini_data['location']
        
        if gemini_data.get('policy_age_months') and gemini_confidence.get('policy_age_months', 0) > 0.7:
            result.policy_age_months = gemini_data['policy_age_months']
        
        if gemini_data.get('hospital'):
            result.hospital = gemini_data['hospital']
        
        if gemini_data.get('amount_claimed'):
            result.amount_claimed = gemini_data['amount_claimed']
        
        if gemini_data.get('date'):
            result.date = gemini_data['date']
        
        if gemini_data.get('complications'):
            result.complications = gemini_data['complications']
        
        # Merge assumptions
        if gemini_data.get('assumptions'):
            result.assumptions.extend(gemini_data['assumptions'])
        
        # Remove duplicates from assumptions
        result.assumptions = list(set(result.assumptions))
        
        return result
    
    def classify_query_intent(self, query: str) -> str:
        """Classify the intent of the query using Gemini"""
        
        prompt = f"""
        Classify the intent of this insurance query: "{query}"
        
        Return one of these categories:
        - "claim_evaluation": User wants to know if a claim will be approved/rejected
        - "policy_inquiry": User is asking about policy terms or coverage
        - "coverage_check": User wants to know if something is covered
        - "document_search": User is looking for specific information in documents
        - "general_question": General insurance-related question
        
        Only return the category name, no other text.
        """
        
        try:
            response = self.gemini_client.generate_content(prompt)
            intent = response.text.strip().lower()
            
            valid_intents = ["claim_evaluation", "policy_inquiry", "coverage_check", "document_search", "general_question"]
            if intent in valid_intents:
                return intent
            else:
                return "claim_evaluation"  # Default
        except Exception as e:
            print(f"Intent classification failed: {e}")
            return "claim_evaluation"
    
    def expand_query_for_search(self, query: str) -> List[str]:
        """Generate multiple search variations using Gemini"""
        
        prompt = f"""
        Generate 3-5 different search variations for this insurance query: "{query}"
        
        Create variations that would help find relevant policy clauses:
        1. Focus on the medical procedure/condition
        2. Focus on coverage and benefits
        3. Focus on exclusions and limitations
        4. Focus on waiting periods and conditions
        5. Focus on specific demographics (age, location, etc.)
        
        Return as a JSON array of strings.
        Only return the JSON array, no other text.
        """
        
        try:
            response = self.gemini_client.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            variations = json.loads(response_text)
            return variations if isinstance(variations, list) else [query]
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]