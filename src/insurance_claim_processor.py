"""
Insurance Claim Processor - Main Module

This module provides the main interface for end-to-end intelligent claim evaluation
using natural language queries and semantic document analysis.
"""

import json
import os
from typing import Dict, Any, Optional
from loguru import logger

from .query_parsing.gemini_query_parser import GeminiQueryParser
from .query_parsing.query_parser import ClaimQuery
from .semantic_search.semantic_retriever import SemanticRetriever
from .decision_engine.gemini_claim_evaluator import GeminiClaimEvaluator
from .decision_engine.claim_evaluator import ClaimDecision

class InsuranceClaimProcessor:
    """
    Main insurance claim processor that orchestrates the entire claim evaluation pipeline.
    
    This class implements the 4-step process:
    1. Parse Query - Extract structured details from natural language
    2. Semantic Retrieval - Find relevant policy clauses
    3. Decision Reasoning - Apply logical reasoning for claim evaluation
    4. JSON Output - Return structured decision response
    """
    
    def __init__(self, data_path: str = "data/policies"):
        """
        Initialize the insurance claim processor.
        
        Args:
            data_path: Path to the directory containing policy documents
        """
        self.data_path = data_path
        
        # Initialize components
        logger.info("Initializing Insurance Claim Processor...")
        
        self.query_parser = GeminiQueryParser()
        self.semantic_retriever = SemanticRetriever(data_path)
        self.claim_evaluator = GeminiClaimEvaluator(self.semantic_retriever)
        
        # Initialize semantic search index
        self.semantic_retriever.initialize_index()
        
        logger.info("Insurance Claim Processor initialized successfully")

    def process_claim(self, query: str, return_json: bool = True) -> Dict[str, Any]:
        """
        Process an insurance claim query end-to-end.
        
        Args:
            query: Natural language query describing the claim scenario
            return_json: Whether to return JSON-formatted response
            
        Returns:
            Dictionary containing the claim decision and analysis
        """
        logger.info(f"Processing claim query: {query}")
        
        try:
            # STEP 1: Parse the Query
            logger.info("Step 1: Parsing query...")
            parsed_query = self.query_parser.parse_query(query)
            logger.info(f"Extracted entities: age={parsed_query.age}, gender={parsed_query.gender}, "
                       f"procedure={parsed_query.procedure}, location={parsed_query.location}")
            
            # STEP 2: Semantic Retrieval
            logger.info("Step 2: Performing semantic retrieval...")
            relevant_clauses = self.semantic_retriever.search_relevant_clauses(parsed_query, top_k=10)
            logger.info(f"Found {len(relevant_clauses)} relevant clauses")
            
            # STEP 3: Decision Reasoning
            logger.info("Step 3: Evaluating claim...")
            decision = self.claim_evaluator.evaluate_claim(parsed_query)
            logger.info(f"Decision: {decision.decision}, Amount: {decision.amount}")
            
            # STEP 4: JSON Output
            logger.info("Step 4: Formatting output...")
            result = self._format_output(decision, parsed_query)
            
            if return_json:
                return result
            else:
                return json.dumps(result, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            return self._format_error_response(str(e))

    def _format_output(self, decision: ClaimDecision, query: ClaimQuery) -> Dict[str, Any]:
        """Format the final output according to the specified JSON structure."""
        
        return {
            "decision": decision.decision,
            "amount": f"₹{decision.amount:,.2f}" if decision.amount else None,
            "justification": decision.justification,
            "mapped_clauses": decision.mapped_clauses,
            "confidence_score": round(decision.confidence_score, 2),
            "query_analysis": {
                "extracted_entities": {
                    "age": query.age,
                    "gender": query.gender,
                    "procedure": query.procedure,
                    "location": query.location,
                    "policy_age_months": query.policy_age_months,
                    "hospital": query.hospital,
                    "amount_claimed": query.amount_claimed
                },
                "assumptions_made": query.assumptions
            },
            "processing_metadata": {
                "total_clauses_analyzed": len(decision.mapped_clauses),
                "semantic_search_performed": True,
                "decision_engine_version": "1.0"
            }
        }

    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Format error response in the expected JSON structure."""
        
        return {
            "decision": "error",
            "amount": None,
            "justification": f"Error processing claim: {error_message}",
            "mapped_clauses": [],
            "confidence_score": 0.0,
            "query_analysis": {
                "extracted_entities": {},
                "assumptions_made": []
            },
            "processing_metadata": {
                "error": True,
                "error_message": error_message
            }
        }

    def analyze_query_only(self, query: str) -> Dict[str, Any]:
        """
        Analyze and parse query without full claim processing.
        Useful for debugging and understanding query parsing.
        """
        parsed_query = self.query_parser.parse_query(query)
        
        return {
            "raw_query": query,
            "extracted_entities": {
                "age": parsed_query.age,
                "gender": parsed_query.gender,
                "procedure": parsed_query.procedure,
                "location": parsed_query.location,
                "policy_age_months": parsed_query.policy_age_months,
                "hospital": parsed_query.hospital,
                "amount_claimed": parsed_query.amount_claimed,
                "date": parsed_query.date
            },
            "assumptions": parsed_query.assumptions,
            "spacy_entities": parsed_query.extracted_entities
        }

    def search_clauses_only(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform semantic search without full claim processing.
        Useful for understanding what clauses are being retrieved.
        """
        parsed_query = self.query_parser.parse_query(query)
        relevant_clauses = self.semantic_retriever.search_relevant_clauses(parsed_query, top_k=top_k)
        
        return {
            "query": query,
            "total_clauses_found": len(relevant_clauses),
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "section": clause.section,
                    "policy_name": clause.policy_name,
                    "text": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
                    "relevance_score": round(clause.relevance_score, 3),
                    "clause_type": clause.clause_type
                }
                for clause in relevant_clauses
            ]
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of loaded policies and their clause counts."""
        
        if not self.semantic_retriever.clauses:
            self.semantic_retriever.initialize_index()
        
        policy_stats = {}
        for clause in self.semantic_retriever.clauses:
            if clause.policy_name not in policy_stats:
                policy_stats[clause.policy_name] = {
                    "total_clauses": 0,
                    "coverage_clauses": 0,
                    "exclusion_clauses": 0,
                    "waiting_period_clauses": 0,
                    "condition_clauses": 0
                }
            
            policy_stats[clause.policy_name]["total_clauses"] += 1
            policy_stats[clause.policy_name][f"{clause.clause_type}_clauses"] += 1
        
        return {
            "total_policies": len(policy_stats),
            "total_clauses": len(self.semantic_retriever.clauses),
            "policy_breakdown": policy_stats
        }

    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the semantic search index from scratch."""
        
        logger.info("Rebuilding semantic search index...")
        self.semantic_retriever.initialize_index(force_rebuild=True)
        
        return {
            "status": "success",
            "message": "Semantic search index rebuilt successfully",
            "total_clauses": len(self.semantic_retriever.clauses)
        }


# Convenience function for direct usage
def process_insurance_claim(query: str, data_path: str = "data/policies") -> str:
    """
    Convenience function to process a single insurance claim query.
    
    Args:
        query: Natural language query describing the claim scenario
        data_path: Path to policy documents directory
        
    Returns:
        JSON string with claim decision
    """
    processor = InsuranceClaimProcessor(data_path)
    result = processor.process_claim(query)
    return json.dumps(result, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Example queries for testing
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "35-year-old female, maternity delivery in Mumbai, 12-month policy",
        "28-year-old male, accident treatment in Delhi, 6-month policy",
        "55-year-old female, cataract surgery in Bangalore, 18-month policy"
    ]
    
    processor = InsuranceClaimProcessor()
    
    print("=== Insurance Claim Processor Test ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 50)
        
        result = processor.process_claim(query)
        print(json.dumps(result, indent=2))
        print("\n" + "="*80 + "\n")
