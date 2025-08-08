import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from datetime import datetime, timedelta
from ..query_parsing.query_parser import ClaimQuery
from ..semantic_search.semantic_retriever import PolicyClause, SemanticRetriever

class ClaimDecision(BaseModel):
    """Represents the final claim decision"""
    decision: str  # "approved" or "rejected"
    amount: Optional[float] = None
    justification: str
    mapped_clauses: List[Dict[str, str]]
    confidence_score: float = 0.0
    assumptions_used: List[str] = []

class ClaimEvaluator:
    """Intelligent claim evaluation engine"""
    
    def __init__(self, semantic_retriever: SemanticRetriever):
        self.semantic_retriever = semantic_retriever
        
        # Decision rules and thresholds
        self.waiting_periods = {
            'knee surgery': 24,  # months
            'heart surgery': 36,
            'eye surgery': 12,
            'dental': 6,
            'maternity': 10,
            'cancer treatment': 48,
            'accident': 0,  # No waiting period for accidents
            'critical illness': 3,  # 3 months (90 days)
            'opd': 0
        }
        
        # Coverage limits by procedure type
        self.coverage_limits = {
            'knee surgery': 200000,
            'heart surgery': 500000,
            'eye surgery': 100000,
            'dental': 50000,
            'maternity': 150000,
            'cancer treatment': 1000000,
            'accident': 300000,
            'critical illness': 2000000,  # Lump sum
            'opd': 25000
        }
        # Sub-limits for OPD
        self.opd_sub_limits = {
            'consultation': 1000,
            'diagnostics': 2000,
            'medicines': 5000,
            'dental': 2000,
            'physiotherapy': 3000
        }
        # List of critical illnesses (expandable)
        self.critical_illnesses = [
            'cancer', 'heart attack', 'stroke', 'kidney failure', 'major organ transplant',
            'multiple sclerosis', 'paralysis', 'coma', 'coronary artery bypass', 'primary pulmonary hypertension'
        ]
        
        # Age-based multipliers
        self.age_multipliers = {
            (0, 18): 1.0,
            (19, 35): 1.0,
            (36, 50): 1.1,
            (51, 65): 1.3,
            (66, 100): 1.5
        }

    def evaluate_claim(self, query: ClaimQuery) -> ClaimDecision:
        """Evaluate insurance claim and make decision"""
        
        # Get relevant clauses
        relevant_clauses = self.semantic_retriever.search_relevant_clauses(query, top_k=15)
        
        # Analyze clauses for decision factors
        analysis = self._analyze_clauses(relevant_clauses, query)
        
        # Make decision based on analysis
        decision = self._make_decision(analysis, query)
        
        # Format mapped clauses for output
        mapped_clauses = self._format_mapped_clauses(relevant_clauses[:5])  # Top 5 most relevant
        
        return ClaimDecision(
            decision=decision['decision'],
            amount=decision['amount'],
            justification=decision['justification'],
            mapped_clauses=mapped_clauses,
            confidence_score=decision['confidence'],
            assumptions_used=query.assumptions
        )

    def _analyze_clauses(self, clauses: List[PolicyClause], query: ClaimQuery) -> Dict[str, Any]:
        """Analyze retrieved clauses for decision factors, including critical illness and OPD"""
        analysis = {
            'coverage_clauses': [],
            'exclusion_clauses': [],
            'waiting_period_clauses': [],
            'condition_clauses': [],
            'coverage_found': False,
            'exclusions_found': False,
            'waiting_period_violation': False,
            'conditions_met': True,
            'max_coverage_amount': 0,
            'applicable_waiting_period': 0,
            'is_critical_illness': False,
            'is_opd': False,
            'opd_type': None,
            'opd_sub_limit': 0
        }
        # Detect critical illness
        if query.procedure and any(ci in query.procedure.lower() for ci in self.critical_illnesses):
            analysis['is_critical_illness'] = True
        # Detect OPD
        if query.procedure and ('opd' in query.procedure.lower() or 'outpatient' in query.procedure.lower()):
            analysis['is_opd'] = True
            for opd_type in self.opd_sub_limits:
                if opd_type in query.procedure.lower():
                    analysis['opd_type'] = opd_type
                    analysis['opd_sub_limit'] = self.opd_sub_limits[opd_type]
        # Detect accident
        if query.procedure and 'accident' in query.procedure.lower():
            analysis['is_accident'] = True
        else:
            analysis['is_accident'] = False
        # Detect maternity/newborn
        if query.procedure and ('maternity' in query.procedure.lower() or 'delivery' in query.procedure.lower() or 'childbirth' in query.procedure.lower()):
            analysis['is_maternity'] = True
        else:
            analysis['is_maternity'] = False
        if query.procedure and ('newborn' in query.procedure.lower() or 'baby' in query.procedure.lower()):
            analysis['is_newborn'] = True
        else:
            analysis['is_newborn'] = False
        # Sub-limits and co-pay
        analysis['sub_limit'] = None
        analysis['sub_limit_type'] = None
        analysis['co_pay_percent'] = None
        analysis['is_daycare'] = False
        analysis['is_preexisting'] = False
        analysis['is_permanent_exclusion'] = False
        for clause in clauses:
            clause_text_lower = clause.text.lower()
            # Sub-limits
            for sub in ['room rent', 'icu', 'cataract', 'knee replacement']:
                if sub in clause_text_lower:
                    amt = self._extract_amount_from_clause(clause.text)
                    if amt:
                        analysis['sub_limit'] = amt
                        analysis['sub_limit_type'] = sub
            # Co-pay
            if 'co-pay' in clause_text_lower or 'copay' in clause_text_lower:
                match = re.search(r'(\d+)%', clause.text)
                if match:
                    analysis['co_pay_percent'] = int(match.group(1))
            # Daycare
            if 'daycare' in clause_text_lower or 'less than 24 hours' in clause_text_lower:
                analysis['is_daycare'] = True
            # Pre-existing
            if 'pre-existing' in clause_text_lower or 'preexisting' in clause_text_lower:
                analysis['is_preexisting'] = True
            # Permanent exclusions
            for exclusion in ['cosmetic', 'infertility', 'hiv', 'aids', 'experimental']:
                if exclusion in clause_text_lower:
                    analysis['is_permanent_exclusion'] = True
            # Categorize clauses
            if clause.clause_type == 'coverage' or self._is_coverage_clause(clause_text_lower, query):
                analysis['coverage_clauses'].append(clause)
                analysis['coverage_found'] = True
                # Extract coverage amount if mentioned
                amount = self._extract_amount_from_clause(clause.text)
                if amount:
                    analysis['max_coverage_amount'] = max(analysis['max_coverage_amount'], amount)
            elif clause.clause_type == 'exclusion' or self._is_exclusion_clause(clause_text_lower, query):
                analysis['exclusion_clauses'].append(clause)
                if self._matches_procedure(clause_text_lower, query.procedure):
                    analysis['exclusions_found'] = True
            
            elif clause.clause_type == 'waiting_period' or self._is_waiting_period_clause(clause_text_lower):
                analysis['waiting_period_clauses'].append(clause)
                waiting_months = self._extract_waiting_period(clause.text)
                if waiting_months:
                    analysis['applicable_waiting_period'] = max(analysis['applicable_waiting_period'], waiting_months)
            
            elif clause.clause_type == 'condition':
                analysis['condition_clauses'].append(clause)
                if not self._check_condition_compliance(clause.text, query):
                    analysis['conditions_met'] = False
        
        # Check waiting period violation
        if query.policy_age_months and analysis['applicable_waiting_period']:
            if query.policy_age_months < analysis['applicable_waiting_period']:
                analysis['waiting_period_violation'] = True
        elif query.procedure and query.procedure in self.waiting_periods:
            default_waiting = self.waiting_periods[query.procedure]
            if query.policy_age_months and query.policy_age_months < default_waiting:
                analysis['waiting_period_violation'] = True
        
        return analysis

    def _make_decision(self, analysis: Dict[str, Any], query: ClaimQuery) -> Dict[str, Any]:
        """Make final claim decision based on analysis, including critical illness and OPD rules"""
        # Check exclusions first
        if analysis['exclusions_found']:
            justification = "Claim rejected due to policy exclusions: " + ", ".join([c.text[:100] for c in analysis['exclusion_clauses']])
            return {
                'decision': 'rejected',
                'amount': 0,
                'justification': justification,
                'confidence': 0.7
            }
        # Check waiting period violation
        if analysis['waiting_period_violation']:
            justification = "Claim rejected due to waiting period: " + ", ".join([c.text[:100] for c in analysis['waiting_period_clauses']])
            return {
                'decision': 'rejected',
                'amount': 0,
                'justification': justification,
                'confidence': 0.8
            }
        # Check coverage
        if not analysis['coverage_found']:
            return {
                'decision': 'rejected',
                'amount': 0,
                'justification': 'No relevant coverage found in policy.',
                'confidence': 0.5
            }
        # --- Critical Illness Logic ---
        if analysis.get('is_critical_illness'):
            # Waiting period for critical illness
            if query.policy_age_months is not None and query.policy_age_months < self.waiting_periods['critical illness']:
                justification = "Claim rejected due to critical illness waiting period ({} months).".format(self.waiting_periods['critical illness'])
                return {
                    'decision': 'rejected',
                    'amount': 0,
                    'justification': justification,
                    'confidence': 0.85
                }
            # Lump sum payout
            justification = "Critical illness claim approved. Referenced clauses: " + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': self.coverage_limits['critical illness'],
                'justification': justification,
                'confidence': 0.97
            }
        # --- OPD Logic ---
        if analysis.get('is_opd'):
            # Sub-limit for OPD type
            opd_limit = analysis.get('opd_sub_limit') or self.coverage_limits['opd']
            justification = "OPD claim approved. Sub-limit: Rs. {}. Referenced clauses: ".format(opd_limit) + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': opd_limit,
                'justification': justification,
                'confidence': 0.93
            }
        # --- Accident Logic ---
        if analysis.get('is_accident'):
            # Exclude self-inflicted, intoxication, hazardous activities (if found in exclusions)
            for clause in analysis['exclusion_clauses']:
                if any(term in clause.text.lower() for term in ['self-inflicted', 'intoxication', 'hazardous', 'suicide']):
                    justification = "Claim rejected due to accident exclusion: " + clause.text[:100]
                    return {
                        'decision': 'rejected',
                        'amount': 0,
                        'justification': justification,
                        'confidence': 0.85
                    }
            # Double payout for accident
            accident_limit = self.coverage_limits['accident'] * 2
            justification = "Accident claim approved with double payout (Rs. {}). Referenced clauses: ".format(accident_limit) + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': accident_limit,
                'justification': justification,
                'confidence': 0.96
            }
        # --- Maternity/Newborn Logic ---
        if analysis.get('is_maternity'):
            # Waiting period for maternity
            if query.policy_age_months is not None and query.policy_age_months < self.waiting_periods['maternity']:
                justification = "Claim rejected due to maternity waiting period ({} months).".format(self.waiting_periods['maternity'])
                return {
                    'decision': 'rejected',
                    'amount': 0,
                    'justification': justification,
                    'confidence': 0.85
                }
            # Cap on maternity expenses
            maternity_limit = self.coverage_limits['maternity']
            justification = "Maternity claim approved. Cap: Rs. {}. Referenced clauses: ".format(maternity_limit) + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': maternity_limit,
                'justification': justification,
                'confidence': 0.94
            }
        if analysis.get('is_newborn'):
            # Newborn covered only if specified
            for clause in analysis['coverage_clauses']:
                if 'newborn' in clause.text.lower() or 'baby' in clause.text.lower():
                    justification = "Newborn claim approved. Referenced clauses: " + clause.text[:100]
                    return {
                        'decision': 'approved',
                        'amount': self.coverage_limits.get('newborn', 50000),
                        'justification': justification,
                        'confidence': 0.92
                    }
            justification = "Claim rejected: Newborn not covered as per policy clauses."
            return {
                'decision': 'rejected',
                'amount': 0,
                'justification': justification,
                'confidence': 0.7
            }
        # --- Permanent Exclusion Logic ---
        if analysis.get('is_permanent_exclusion'):
            justification = "Claim rejected due to permanent exclusion in policy (e.g., cosmetic, experimental, HIV, infertility)."
            return {
                'decision': 'rejected',
                'amount': 0,
                'justification': justification,
                'confidence': 0.9
            }
        # --- Pre-existing Disease Waiting Period ---
        if analysis.get('is_preexisting'):
            if query.policy_age_months is not None and query.policy_age_months < 36:
                justification = "Claim rejected due to pre-existing disease waiting period (3 years)."
                return {
                    'decision': 'rejected',
                    'amount': 0,
                    'justification': justification,
                    'confidence': 0.85
                }
        # --- Daycare Procedure Logic ---
        if analysis.get('is_daycare'):
            justification = "Daycare procedure claim approved. Referenced clauses: " + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': analysis['max_coverage_amount'] or 50000,
                'justification': justification,
                'confidence': 0.93
            }
        # --- Sub-limit Logic ---
        if analysis.get('sub_limit'):
            justification = f"Claim approved with sub-limit for {analysis['sub_limit_type']}: Rs. {analysis['sub_limit']}. Referenced clauses: " + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': analysis['sub_limit'],
                'justification': justification,
                'confidence': 0.92
            }
        # --- Co-pay Logic ---
        if analysis.get('co_pay_percent'):
            base_amount = self._calculate_coverage_amount(analysis, query)
            co_pay = base_amount * (analysis['co_pay_percent'] / 100)
            final_amount = base_amount - co_pay
            justification = f"Claim approved with co-pay of {analysis['co_pay_percent']}%. Payable: Rs. {final_amount:,.2f}. Referenced clauses: " + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
            return {
                'decision': 'approved',
                'amount': final_amount,
                'justification': justification,
                'confidence': 0.91
            }
        # If all conditions met, approve
        amount = self._calculate_coverage_amount(analysis, query)
        justification = "Claim approved. Referenced clauses: " + ", ".join([c.text[:100] for c in analysis['coverage_clauses']])
        return {
            'decision': 'approved',
            'amount': amount,
            'justification': justification,
            'confidence': 0.95
        }
        
        return decision

    def _calculate_coverage_amount(self, analysis: Dict[str, Any], query: ClaimQuery) -> Optional[float]:
        """Calculate the coverage amount for approved claims"""
        
        # Use amount from clauses if available
        if analysis['max_coverage_amount'] > 0:
            base_amount = analysis['max_coverage_amount']
        
        # Use default coverage limits
        elif query.procedure and query.procedure in self.coverage_limits:
            base_amount = self.coverage_limits[query.procedure]
        
        # Use claimed amount if available
        elif query.amount_claimed:
            base_amount = min(query.amount_claimed, 500000)  # Cap at 5 lakhs
        
        else:
            return None
        
        # Apply age-based adjustments
        if query.age:
            age_multiplier = self._get_age_adjustment(query.age)
            base_amount *= age_multiplier
        
        return base_amount

    def _get_age_adjustment(self, age: int) -> float:
        """Get age-based coverage adjustment factor"""
        for (min_age, max_age), multiplier in self.age_multipliers.items():
            if min_age <= age <= max_age:
                return multiplier
        return 1.0

    def _is_coverage_clause(self, text: str, query: ClaimQuery) -> bool:
        """Check if clause indicates coverage"""
        coverage_indicators = ['covered', 'eligible', 'benefits', 'reimburse', 'payable']
        procedure_match = query.procedure and any(word in text for word in query.procedure.split())
        
        return any(indicator in text for indicator in coverage_indicators) and procedure_match

    def _is_exclusion_clause(self, text: str, query: ClaimQuery) -> bool:
        """Check if clause indicates exclusion"""
        exclusion_indicators = ['excluded', 'not covered', 'shall not', 'except', 'not eligible']
        return any(indicator in text for indicator in exclusion_indicators)

    def _is_waiting_period_clause(self, text: str) -> bool:
        """Check if clause mentions waiting period"""
        waiting_indicators = ['waiting period', 'after', 'months', 'cooling period', 'moratorium']
        return any(indicator in text for indicator in waiting_indicators)

    def _matches_procedure(self, text: str, procedure: Optional[str]) -> bool:
        """Check if clause text matches the procedure"""
        if not procedure:
            return False
        
        procedure_words = procedure.lower().split()
        return any(word in text for word in procedure_words)

    def _extract_amount_from_clause(self, text: str) -> Optional[float]:
        """Extract monetary amount from clause text"""
        amount_patterns = [
            r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹|rupees)',
            r'(?:sum insured|coverage|limit).*?(?:rs\.?|inr|₹)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+)\s*lakh',
            r'(\d+)\s*crore'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Convert lakhs and crores
                if 'lakh' in pattern:
                    amount *= 100000
                elif 'crore' in pattern:
                    amount *= 10000000
                
                return amount
        
        return None

    def _extract_waiting_period(self, text: str) -> Optional[int]:
        """Extract waiting period in months from clause text"""
        patterns = [
            r'(\d+)\s*months?',
            r'(\d+)\s*years?',
            r'after\s+(\d+)\s*months?',
            r'waiting\s+period.*?(\d+)\s*months?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                period = int(match.group(1))
                if 'year' in pattern:
                    period *= 12
                return period
        
        return None

    def _check_condition_compliance(self, clause_text: str, query: ClaimQuery) -> bool:
        """Check if query meets the conditions specified in clause"""
        # This is a simplified check - in practice, this would be more sophisticated
        
        # Check age conditions
        age_match = re.search(r'age.*?(\d+)', clause_text, re.IGNORECASE)
        if age_match and query.age:
            required_age = int(age_match.group(1))
            if 'above' in clause_text.lower() or 'over' in clause_text.lower():
                return query.age > required_age
            elif 'below' in clause_text.lower() or 'under' in clause_text.lower():
                return query.age < required_age
        
        # Check location conditions
        if query.location and 'network' in clause_text.lower():
            # Assume treatment in major cities is in network
            major_cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'pune', 'hyderabad']
            return query.location.lower() in major_cities
        
        # Default to True if no specific conditions can be checked
        return True

    def _format_mapped_clauses(self, clauses: List[PolicyClause]) -> List[Dict[str, str]]:
        """Format clauses for JSON output"""
        mapped_clauses = []
        
        for clause in clauses:
            relevance_desc = self._get_relevance_description(clause)
            
            mapped_clause = {
                "clause_id": f"{clause.section} - {clause.clause_id}",
                "text": clause.text[:200] + "..." if len(clause.text) > 200 else clause.text,
                "relevance": relevance_desc
            }
            mapped_clauses.append(mapped_clause)
        
        return mapped_clauses

    def _get_relevance_description(self, clause: PolicyClause) -> str:
        """Generate relevance description for a clause"""
        descriptions = {
            'coverage': 'Defines coverage eligibility and benefits',
            'exclusion': 'Lists exclusions and limitations',
            'waiting_period': 'Specifies waiting period requirements',
            'condition': 'Outlines policy conditions and requirements'
        }
        
        base_desc = descriptions.get(clause.clause_type, 'General policy information')
        
        if clause.relevance_score > 0.8:
            return f"High relevance: {base_desc}"
        elif clause.relevance_score > 0.6:
            return f"Medium relevance: {base_desc}"
        else:
            return f"Low relevance: {base_desc}"
