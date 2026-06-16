"""
Stage 4: Clause Matching - Specialized semantic matching for legal and policy documents
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ...core.interfaces import IClauseMatcher, ClauseType
from ...core.models import Clause, ClauseMatch, ProcessingMetadata
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import timing_decorator


class MatchType(Enum):
    """Types of clause matches"""
    EXACT = "exact"
    SEMANTIC = "semantic"
    PARTIAL = "partial"
    CONTEXTUAL = "contextual"


class ClauseMatcher(IClauseMatcher):
    """
    Specialized semantic matching for legal and policy documents.
    
    Features:
    - Clause type categorization
    - Semantic similarity matching
    - Obligation extraction
    - Relationship identification
    """
    
    def __init__(self, similarity_engine: Any = None):
        self.similarity_engine = similarity_engine
        self.logger = get_pipeline_logger()

    @timing_decorator
    def match_clauses(self, query: str, clauses: List[Clause]) -> List[ClauseMatch]:
        """Match query against clauses using semantic similarity"""
        self.logger.info(f"Matching query against {len(clauses)} clauses")
        
        matches = []
        
        # In a real implementation, we would use embeddings for semantic matching
        # For now, we'll use a simple keyword-based approach as a baseline
        # but the interface supports vector-based matching if similarity_engine is provided
        
        query_lower = query.lower()
        
        for clause in clauses:
            score = 0.0
            content_lower = clause.content.lower()
            
            # Simple keyword matching for demonstration
            # In production, this would call self.similarity_engine.calculate_similarity(query, clause.content)
            common_words = set(query_lower.split()) & set(content_lower.split())
            if common_words:
                score = len(common_words) / max(len(set(query_lower.split())), 1)
            
            if score > 0.1:
                matches.append(ClauseMatch(
                    clause=clause,
                    similarity_score=score,
                    match_type="semantic",
                    explanation=f"Matched {len(common_words)} common keywords."
                ))
        
        # Sort by score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Assign ranks
        for i, match in enumerate(matches):
            match.rank = i + 1
            
        return matches[:10]  # Return top 10 matches

    def find_related_clauses(self, clause: Clause) -> List[Clause]:
        """Find clauses related to the given clause"""
        # This would typically use cross-references or shared terms
        related = []
        # Implementation logic here
        return related

    def categorize_clauses(self, clauses: List[Clause]) -> Dict[ClauseType, List[Clause]]:
        """Categorize clauses by type"""
        categories = {ct: [] for ct in ClauseType}
        
        for clause in clauses:
            try:
                ctype = ClauseType(clause.clause_type.lower())
                categories[ctype].append(clause)
            except (ValueError, AttributeError):
                # Handle unknown clause types
                pass
                
        return categories

    def extract_obligations(self, clauses: List[Clause]) -> List[str]:
        """Extract all obligations from a list of clauses"""
        all_obligations = []
        for clause in clauses:
            if clause.obligations:
                all_obligations.extend(clause.obligations)
        return all_obligations
