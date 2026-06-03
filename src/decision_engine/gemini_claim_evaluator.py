"""
Missing module: Gemini Claim Evaluator
Evaluates insurance claims using Google Gemini AI
"""

from typing import Dict, Any, List, Optional
from .claim_evaluator import ClaimEvaluator, ClaimDecision
from ..query_parsing.query_parser import ClaimQuery
from ..semantic_search.semantic_retriever import SemanticRetriever
import google.generativeai as genai
import os
import json

class GeminiClaimEvaluator(ClaimEvaluator):
    """Enhanced claim evaluator using Gemini AI"""
    
    def __init__(self, semantic_retriever: SemanticRetriever):
        """Initialize Gemini claim evaluator"""
        super().__init__(semantic_retriever)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def evaluate_claim(self, query: ClaimQuery) -> ClaimDecision:
        """Evaluate claim using Gemini AI with semantic search"""
        try:
            # Get relevant clauses using semantic search
            relevant_clauses = self.semantic_retriever.search_relevant_clauses(query, top_k=15)
            
            # Create prompt for Gemini
            clause_text = "\n\n".join([
                f"Clause {i+1}: {clause.text}"
                for i, clause in enumerate(relevant_clauses[:5])
            ])
            
            prompt = f"""
Based on the following insurance policy clauses and patient information, evaluate this insurance claim.

Patient Information:
- Age: {query.age}
- Gender: {query.gender}
- Procedure: {query.procedure}
- Location: {query.location}
- Policy Age (months): {query.policy_age_months}
- Hospital: {query.hospital}
- Amount Claimed: {query.amount_claimed}

Relevant Policy Clauses:
{clause_text}

Provide your evaluation as a JSON object with:
- decision: "approved" or "rejected"
- amount: approved amount in rupees (or null)
- justification: brief explanation
- confidence: confidence score (0-1)

Return ONLY valid JSON:
"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            json_data = json.loads(response_text)
            
            # Format mapped clauses
            mapped_clauses = [
                {
                    "clause_id": clause.clause_id,
                    "text": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
                    "relevance": f"{clause.relevance_score:.2f}"
                }
                for clause in relevant_clauses[:5]
            ]
            
            return ClaimDecision(
                decision=json_data.get('decision', 'rejected'),
                amount=json_data.get('amount'),
                justification=json_data.get('justification', 'Unable to evaluate'),
                mapped_clauses=mapped_clauses,
                confidence_score=json_data.get('confidence', 0.5),
                assumptions_used=query.assumptions
            )
        
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            # Fall back to parent evaluator
            return super().evaluate_claim(query)
        except Exception as e:
            print(f"Error in Gemini evaluation: {e}")
            # Fall back to parent evaluator
            return super().evaluate_claim(query)
