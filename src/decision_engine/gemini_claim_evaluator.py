"""
Enhanced Claim Evaluator using Gemini for intelligent decision making
"""

import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from config.gemini_config import gemini_config
from .claim_evaluator import ClaimEvaluator, ClaimDecision
from ..query_parsing.query_parser import ClaimQuery
from ..semantic_search.semantic_retriever import PolicyClause

class GeminiClaimEvaluator(ClaimEvaluator):
    """Enhanced claim evaluator using Gemini for intelligent reasoning"""
    
    def __init__(self, semantic_retriever):
        super().__init__(semantic_retriever)
        self.gemini_client = gemini_config.get_native_client()
    
    def evaluate_claim(self, query: ClaimQuery) -> ClaimDecision:
        """Evaluate claim using both rule-based and AI-powered reasoning"""
        
        # Get relevant clauses
        relevant_clauses = self.semantic_retriever.search_relevant_clauses(query, top_k=15)
        
        # First, try traditional rule-based evaluation
        try:
            traditional_decision = super().evaluate_claim(query)
            
            # Enhance with Gemini-based reasoning
            gemini_decision = self._evaluate_with_gemini(query, relevant_clauses)
            
            # Merge decisions (prefer Gemini if confidence is high)
            final_decision = self._merge_decisions(traditional_decision, gemini_decision)
            
            return final_decision
            
        except Exception as e:
            print(f"Gemini evaluation failed, using traditional method: {str(e)}")
            return super().evaluate_claim(query)
    
    def _evaluate_with_gemini(self, query: ClaimQuery, clauses: List[PolicyClause]) -> Dict[str, Any]:
        """Use Gemini to evaluate the claim based on policy clauses"""
        
        # Prepare clause context
        clause_context = ""
        for i, clause in enumerate(clauses[:10]):  # Top 10 most relevant
            clause_context += f"\nClause {i+1} ({clause.clause_type}): {clause.text}\n"
        
        prompt = f"""
        You are an expert insurance claim evaluator. Analyze this claim based on the provided policy clauses.

        CLAIM DETAILS:
        - Age: {query.age}
        - Gender: {query.gender}
        - Procedure: {query.procedure}
        - Location: {query.location}
        - Policy Age (months): {query.policy_age_months}
        - Hospital: {query.hospital}
        - Amount Claimed: {query.amount_claimed}
        - Raw Query: {query.raw_query}

        RELEVANT POLICY CLAUSES:
        {clause_context}

        EVALUATION CRITERIA:
        1. Check if the procedure is covered under the policy
        2. Verify if any exclusions apply
        3. Check waiting period requirements
        4. Assess any conditions or limitations
        5. Calculate appropriate coverage amount

        Return your analysis as a JSON object:
        {{
            "decision": "approved" or "rejected",
            "amount": number or null,
            "confidence_score": 0.0-1.0,
            "reasoning_steps": [
                "Step 1: Analysis of coverage...",
                "Step 2: Check for exclusions...",
                "Step 3: Waiting period verification...",
                "Step 4: Final decision rationale..."
            ],
            "supporting_clauses": [
                {{
                    "clause_number": 1,
                    "relevance": "Why this clause supports the decision",
                    "weight": 0.0-1.0
                }}
            ],
            "conflicting_evidence": [
                "Any conflicting information found"
            ],
            "assumptions": [
                "Any assumptions made in the evaluation"
            ]
        }}

        Be thorough in your analysis and provide clear reasoning for your decision.
        Only return the JSON object, no other text.
        """
        
        response = self.gemini_client.generate_content(prompt)
        
        try:
            # Clean and parse response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini decision response: {e}")
            return {"decision": "rejected", "confidence_score": 0.0, "reasoning_steps": ["AI evaluation failed"]}
    
    def _merge_decisions(self, traditional: ClaimDecision, gemini_data: Dict[str, Any]) -> ClaimDecision:
        """Merge traditional and Gemini decisions, preferring higher confidence"""
        
        gemini_confidence = gemini_data.get('confidence_score', 0.0)
        traditional_confidence = traditional.confidence_score
        
        # If Gemini has high confidence (>0.8), prefer its decision
        if gemini_confidence > 0.8 and gemini_confidence > traditional_confidence:
            return self._create_decision_from_gemini(gemini_data, traditional)
        
        # Otherwise, enhance traditional decision with Gemini insights
        enhanced_decision = traditional.model_copy()
        
        # Add Gemini reasoning to justification
        if gemini_data.get('reasoning_steps'):
            ai_reasoning = " AI Analysis: " + " ".join(gemini_data['reasoning_steps'])
            enhanced_decision.justification += ai_reasoning
        
        # Update confidence if Gemini agrees
        if gemini_data.get('decision') == traditional.decision:
            enhanced_decision.confidence_score = min(1.0, (traditional_confidence + gemini_confidence) / 2 + 0.1)
        
        return enhanced_decision
    
    def _create_decision_from_gemini(self, gemini_data: Dict[str, Any], traditional: ClaimDecision) -> ClaimDecision:
        """Create decision object from Gemini analysis"""
        
        # Build detailed justification
        justification = ""
        if gemini_data.get('reasoning_steps'):
            justification = " ".join(gemini_data['reasoning_steps'])
        
        # Map supporting clauses
        mapped_clauses = []
        if gemini_data.get('supporting_clauses'):
            for clause_info in gemini_data['supporting_clauses']:
                mapped_clauses.append({
                    "clause_id": f"AI_Analysis_{clause_info.get('clause_number', 1)}",
                    "text": clause_info.get('relevance', 'Supporting evidence'),
                    "relevance": f"Weight: {clause_info.get('weight', 0.5)}"
                })
        else:
            # Use traditional mapped clauses as fallback
            mapped_clauses = traditional.mapped_clauses
        
        return ClaimDecision(
            decision=gemini_data.get('decision', 'rejected'),
            amount=gemini_data.get('amount'),
            justification=justification,
            mapped_clauses=mapped_clauses,
            confidence_score=gemini_data.get('confidence_score', 0.5),
            assumptions_used=gemini_data.get('assumptions', [])
        )
    
    def generate_detailed_justification(self, decision: ClaimDecision, query: ClaimQuery, clauses: List[PolicyClause]) -> str:
        """Generate detailed justification using Gemini"""
        
        prompt = f"""
        Generate a detailed, professional justification for this insurance claim decision:

        DECISION: {decision.decision.upper()}
        AMOUNT: {decision.amount if decision.amount else "N/A"}
        CLAIM: {query.raw_query}

        RELEVANT CLAUSES:
        {chr(10).join([f"- {clause.text[:200]}..." for clause in clauses[:5]])}

        Create a clear, professional justification that:
        1. Explains the decision in simple terms
        2. References specific policy clauses
        3. Addresses any potential concerns
        4. Is suitable for customer communication

        Keep it concise but comprehensive (2-3 paragraphs).
        """
        
        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Justification generation failed: {e}")
            return decision.justification
    
    def identify_clause_conflicts(self, clauses: List[PolicyClause]) -> List[Dict[str, Any]]:
        """Identify conflicting clauses using Gemini"""
        
        if len(clauses) < 2:
            return []
        
        clause_texts = [f"Clause {i+1}: {clause.text}" for i, clause in enumerate(clauses[:10])]
        
        prompt = f"""
        Analyze these insurance policy clauses for conflicts or contradictions:

        {chr(10).join(clause_texts)}

        Identify any clauses that:
        1. Contradict each other
        2. Have overlapping but different coverage terms
        3. Create ambiguity in coverage decisions

        Return as JSON array:
        [
            {{
                "clause_1": 1,
                "clause_2": 3,
                "conflict_type": "contradiction" or "overlap" or "ambiguity",
                "description": "Explanation of the conflict"
            }}
        ]

        Return empty array [] if no conflicts found.
        Only return the JSON array, no other text.
        """
        
        try:
            response = self.gemini_client.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            conflicts = json.loads(response_text)
            return conflicts if isinstance(conflicts, list) else []
        except Exception as e:
            print(f"Conflict identification failed: {e}")
            return []