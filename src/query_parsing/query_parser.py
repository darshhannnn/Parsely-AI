import re
import spacy
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime, timedelta
import json

class ClaimQuery(BaseModel):
    """Structured representation of a parsed claim query"""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration_months: Optional[int] = None
    policy_age_months: Optional[int] = None
    hospital: Optional[str] = None
    date: Optional[str] = None
    complications: Optional[List[str]] = None
    amount_claimed: Optional[float] = None
    raw_query: str
    extracted_entities: Dict[str, Any] = {}
    assumptions: List[str] = []

class QueryParser:
    """Intelligent query parser for insurance claim queries"""
    
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Using basic parsing.")
            self.nlp = None
        
        # Medical procedures dictionary
        self.medical_procedures = {
            'knee surgery': ['knee', 'surgery', 'arthroscopy', 'meniscus', 'acl', 'pcl'],
            'heart surgery': ['heart', 'cardiac', 'bypass', 'angioplasty', 'stent'],
            'eye surgery': ['eye', 'cataract', 'lasik', 'retina', 'glaucoma'],
            'dental': ['dental', 'tooth', 'root canal', 'extraction', 'implant'],
            'maternity': ['pregnancy', 'delivery', 'caesarean', 'c-section', 'maternity'],
            'cancer treatment': ['cancer', 'chemotherapy', 'radiation', 'oncology', 'tumor'],
            'accident': ['accident', 'fracture', 'injury', 'emergency', 'trauma']
        }
        
        # Indian cities for location detection
        self.indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai',
            'kolkata', 'pune', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore',
            'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna', 'vadodara',
            'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut'
        ]

    def parse_query(self, query: str) -> ClaimQuery:
        """Parse natural language query into structured claim information"""
        query_lower = query.lower()
        
        # Initialize result
        result = ClaimQuery(raw_query=query)
        
        # Extract age
        result.age = self._extract_age(query_lower)
        
        # Extract gender
        result.gender = self._extract_gender(query_lower)
        
        # Extract procedure
        result.procedure = self._extract_procedure(query_lower)
        
        # Extract location
        result.location = self._extract_location(query_lower)
        
        # Extract policy duration/age
        policy_info = self._extract_policy_info(query_lower)
        result.policy_duration_months = policy_info.get('duration')
        result.policy_age_months = policy_info.get('age')
        
        # Extract hospital
        result.hospital = self._extract_hospital(query)
        
        # Extract date
        result.date = self._extract_date(query)
        
        # Extract amount
        result.amount_claimed = self._extract_amount(query)
        
        # Use spaCy for additional entity extraction if available
        if self.nlp:
            result.extracted_entities = self._extract_entities_spacy(query)
        
        # Add assumptions for missing critical information
        result.assumptions = self._generate_assumptions(result)
        
        return result

    def _extract_age(self, query: str) -> Optional[int]:
        """Extract age from query"""
        age_patterns = [
            r'(\d{1,2})[- ]?year[s]?[- ]?old',
            r'age[:\s]*(\d{1,2})',
            r'(\d{1,2})[- ]?yr[s]?[- ]?old',
            r'(\d{1,2})[- ]?years?'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query)
            if match:
                age = int(match.group(1))
                if 0 < age < 120:  # Reasonable age range
                    return age
        return None

    def _extract_gender(self, query: str) -> Optional[str]:
        """Extract gender from query"""
        if re.search(r'\b(male|man|mr\.?|gentleman)\b', query):
            return 'male'
        elif re.search(r'\b(female|woman|mrs?\.?|ms\.?|lady)\b', query):
            return 'female'
        return None

    def _extract_procedure(self, query: str) -> Optional[str]:
        """Extract medical procedure from query"""
        for procedure, keywords in self.medical_procedures.items():
            for keyword in keywords:
                if keyword in query:
                    return procedure
        
        # Look for surgery-related terms
        surgery_match = re.search(r'(\w+)\s+surgery', query)
        if surgery_match:
            return f"{surgery_match.group(1)} surgery"
        
        # Look for treatment-related terms
        treatment_match = re.search(r'(\w+)\s+treatment', query)
        if treatment_match:
            return f"{treatment_match.group(1)} treatment"
        
        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query"""
        for city in self.indian_cities:
            if city in query:
                return city.title()
        
        # Look for "in [location]" pattern
        location_match = re.search(r'\bin\s+([a-zA-Z\s]+?)(?:\s|,|$)', query)
        if location_match:
            location = location_match.group(1).strip()
            if len(location) > 2 and location.lower() not in ['the', 'a', 'an']:
                return location.title()
        
        return None

    def _extract_policy_info(self, query: str) -> Dict[str, Optional[int]]:
        """Extract policy duration or age information"""
        result = {'duration': None, 'age': None}
        
        # Policy age patterns
        age_patterns = [
            r'(\d+)[- ]?month[s]?[- ]?old\s+(?:insurance\s+)?policy',
            r'policy\s+(?:is\s+)?(\d+)[- ]?month[s]?\s+old',
            r'(\d+)[- ]?month[s]?[- ]?old\s+insurance'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query)
            if match:
                result['age'] = int(match.group(1))
                break
        
        # Policy duration patterns
        duration_patterns = [
            r'(\d+)[- ]?month[s]?\s+policy',
            r'policy\s+(?:of\s+)?(\d+)[- ]?month[s]?',
            r'(\d+)[- ]?year[s]?\s+policy'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, query)
            if match:
                months = int(match.group(1))
                if 'year' in pattern:
                    months *= 12
                result['duration'] = months
                break
        
        return result

    def _extract_hospital(self, query: str) -> Optional[str]:
        """Extract hospital name from query"""
        hospital_patterns = [
            r'(?:at\s+)?([A-Z][a-zA-Z\s]+?)\s+hospital',
            r'hospital[:\s]+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+?)\s+medical\s+center'
        ]
        
        for pattern in hospital_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_date(self, query: str) -> Optional[str]:
        """Extract date from query"""
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})',
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def _extract_amount(self, query: str) -> Optional[float]:
        """Extract claimed amount from query"""
        amount_patterns = [
            r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹|rupees)',
            r'amount[:\s]*(?:rs\.?|inr|₹)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                return float(amount_str)
        
        return None

    def _extract_entities_spacy(self, query: str) -> Dict[str, Any]:
        """Use spaCy for additional entity extraction"""
        doc = self.nlp(query)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["person"] = ent.text
            elif ent.label_ == "GPE":  # Geopolitical entity
                entities["location"] = ent.text
            elif ent.label_ == "DATE":
                entities["date"] = ent.text
            elif ent.label_ == "MONEY":
                entities["amount"] = ent.text
            elif ent.label_ == "ORG":
                entities["organization"] = ent.text
        
        return entities

    def _generate_assumptions(self, result: ClaimQuery) -> List[str]:
        """Generate assumptions for missing information"""
        assumptions = []
        
        if not result.age:
            assumptions.append("Age not specified - assuming adult patient")
        
        if not result.gender:
            assumptions.append("Gender not specified - will apply gender-neutral policies")
        
        if not result.location:
            assumptions.append("Location not specified - assuming treatment in India")
        
        if not result.policy_age_months and not result.policy_duration_months:
            assumptions.append("Policy age/duration not specified - will check waiting periods")
        
        if not result.hospital:
            assumptions.append("Hospital not specified - assuming network hospital")
        
        return assumptions

    def to_json(self, result: ClaimQuery) -> str:
        """Convert parsed result to JSON string"""
        return result.model_dump_json(indent=2)
