"""
Clause and Structure Identification for Legal and Policy Documents
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from ...core.logging_utils import get_pipeline_logger
from ...core.exceptions import ClauseExtractionError, ClauseCategorizationError
from ...core.utils import timing_decorator, generate_correlation_id
from .llm_integration import LLMManager, LLMRequest
from .content_chunker import ContentChunk


class ClauseType(Enum):
    """Types of clauses in legal/policy documents"""
    DEFINITION = "definition"
    OBLIGATION = "obligation"
    RIGHT = "right"
    CONDITION = "condition"
    TERM = "term"
    PENALTY = "penalty"
    TERMINATION = "termination"
    PAYMENT = "payment"
    LIABILITY = "liability"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"
    FORCE_MAJEURE = "force_majeure"
    AMENDMENT = "amendment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    NOTICE = "notice"
    ASSIGNMENT = "assignment"
    INDEMNIFICATION = "indemnification"
    WARRANTY = "warranty"
    DISCLAIMER = "disclaimer"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    DATA_PROTECTION = "data_protection"
    COMPLIANCE = "compliance"
    OTHER = "other"


class StructureType(Enum):
    """Types of document structures"""
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    SECTION = "section"
    SUBSECTION = "subsection"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    LIST_ITEM = "list_item"
    DEFINITION_ITEM = "definition_item"
    SCHEDULE = "schedule"
    APPENDIX = "appendix"
    EXHIBIT = "exhibit"
    PREAMBLE = "preamble"
    RECITAL = "recital"
    SIGNATURE_BLOCK = "signature_block"


class RelationshipType(Enum):
    """Types of relationships between clauses"""
    DEPENDS_ON = "depends_on"
    MODIFIES = "modifies"
    REFERENCES = "references"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"
    COMPLEMENTS = "complements"
    DEFINES = "defines"
    APPLIES_TO = "applies_to"
    EXCLUDES = "excludes"
    INCLUDES = "includes"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SIBLING_OF = "sibling_of"

@dataclass
class ClauseRelationship:
    """Represents a relationship between clauses"""
    source_clause_id: str
    target_clause_id: str
    relationship_type: RelationshipType
    confidence: float
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IdentifiedClause:
    """Represents an identified clause in a document"""
    id: str
    content: str
    clause_type: ClauseType
    structure_type: StructureType
    start_position: int
    end_position: int
    document_id: str
    section_path: List[str] = field(default_factory=list)  # e.g., ["Article 1", "Section 1.1"]
    numbering: Optional[str] = None  # e.g., "1.1.1", "(a)", "i."
    title: Optional[str] = None
    key_terms: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # References to other clauses/sections
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_clause_id()
    
    def _generate_clause_id(self) -> str:
        """Generate unique clause ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_clause_{self.start_position}_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clause to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "clause_type": self.clause_type.value,
            "structure_type": self.structure_type.value,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "numbering": self.numbering,
            "title": self.title,
            "key_terms": self.key_terms,
            "obligations": self.obligations,
            "conditions": self.conditions,
            "references": self.references,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document"""
    document_id: str
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    clauses: List[IdentifiedClause] = field(default_factory=list)
    relationships: List[ClauseRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_clause_by_id(self, clause_id: str) -> Optional[IdentifiedClause]:
        """Get clause by ID"""
        for clause in self.clauses:
            if clause.id == clause_id:
                return clause
        return None
    
    def get_clauses_by_type(self, clause_type: ClauseType) -> List[IdentifiedClause]:
        """Get all clauses of a specific type"""
        return [clause for clause in self.clauses if clause.clause_type == clause_type]
    
    def get_related_clauses(self, clause_id: str) -> List[Tuple[IdentifiedClause, RelationshipType]]:
        """Get all clauses related to a given clause"""
        related = []
        for relationship in self.relationships:
            if relationship.source_clause_id == clause_id:
                target_clause = self.get_clause_by_id(relationship.target_clause_id)
                if target_clause:
                    related.append((target_clause, relationship.relationship_type))
            elif relationship.target_clause_id == clause_id:
                source_clause = self.get_clause_by_id(relationship.source_clause_id)
                if source_clause:
                    related.append((source_clause, relationship.relationship_type))
        return related

class LegalPatternMatcher:
    """Pattern matcher for legal and policy document structures"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for legal document elements"""
        
        # Numbering patterns
        self.numbering_patterns = {
            'decimal': re.compile(r'^\s*(\d+(?:\.\d+)*\.?)\s+(.+)$', re.MULTILINE),
            'roman': re.compile(r'^\s*([ivxlcdm]+\.?)\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'letter': re.compile(r'^\s*([a-z]\.?)\s+(.+)$', re.MULTILINE),
            'parenthetical': re.compile(r'^\s*\(([a-z0-9]+)\)\s+(.+)$', re.MULTILINE),
            'bracket': re.compile(r'^\s*\[([a-z0-9]+)\]\s+(.+)$', re.MULTILINE),
        }
        
        # Legal structure patterns
        self.structure_patterns = {
            'article': re.compile(r'^\s*ARTICLE\s+([IVXLCDM\d]+)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'section': re.compile(r'^\s*SECTION\s+(\d+(?:\.\d+)*)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'clause': re.compile(r'^\s*CLAUSE\s+(\d+(?:\.\d+)*)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'paragraph': re.compile(r'^\s*PARAGRAPH\s+(\d+(?:\.\d+)*)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'schedule': re.compile(r'^\s*SCHEDULE\s+([A-Z\d]+)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'appendix': re.compile(r'^\s*APPENDIX\s+([A-Z\d]+)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
            'exhibit': re.compile(r'^\s*EXHIBIT\s+([A-Z\d]+)[\.\:\-\s]*(.*)$', re.MULTILINE | re.IGNORECASE),
        }
        
        # Clause type indicators
        self.clause_indicators = {
            ClauseType.DEFINITION: [
                r'\b(?:means?|defined?\s+as|shall\s+mean|definition)\b',
                r'\b(?:for\s+the\s+purposes?\s+of|in\s+this\s+agreement)\b',
                r'^\s*"[^"]+"\s+means?',
            ],
            ClauseType.OBLIGATION: [
                r'\b(?:shall|must|will|agrees?\s+to|undertakes?\s+to|obligated?\s+to)\b',
                r'\b(?:responsible\s+for|duty\s+to|required\s+to)\b',
                r'\b(?:covenant|promise|guarantee)\b',
            ],
            ClauseType.RIGHT: [
                r'\b(?:entitled\s+to|right\s+to|may|permitted\s+to|authorized\s+to)\b',
                r'\b(?:privilege|license|authority)\b',
            ],
            ClauseType.CONDITION: [
                r'\b(?:if|unless|provided\s+that|subject\s+to|conditional\s+upon)\b',
                r'\b(?:in\s+the\s+event\s+that|where|when|whenever)\b',
            ],
            ClauseType.PAYMENT: [
                r'\b(?:payment|pay|fee|cost|expense|charge|amount|sum)\b',
                r'\b(?:invoice|bill|remuneration|compensation)\b',
                r'\$[\d,]+|\b\d+\s*(?:dollars?|USD|EUR|GBP)\b',
            ],
            ClauseType.TERMINATION: [
                r'\b(?:terminat|expir|end|ceas|discontinu)\b',
                r'\b(?:breach|default|violation|non-compliance)\b',
                r'\b(?:notice\s+of\s+termination|effective\s+date)\b',
            ],
            ClauseType.LIABILITY: [
                r'\b(?:liable|liability|responsible|damages|loss|harm)\b',
                r'\b(?:indemnif|compensat|reimburse)\b',
            ],
            ClauseType.CONFIDENTIALITY: [
                r'\b(?:confidential|proprietary|non-disclosure|secret)\b',
                r'\b(?:disclose|reveal|share|divulge)\b',
            ],
            ClauseType.GOVERNING_LAW: [
                r'\b(?:governing\s+law|applicable\s+law|jurisdiction)\b',
                r'\b(?:courts?\s+of|legal\s+proceedings)\b',
            ],
            ClauseType.FORCE_MAJEURE: [
                r'\b(?:force\s+majeure|act\s+of\s+god|unforeseeable)\b',
                r'\b(?:natural\s+disaster|war|pandemic|government\s+action)\b',
            ],
        }
        
        # Reference patterns
        self.reference_patterns = [
            re.compile(r'\b(?:Section|Article|Clause|Paragraph)\s+(\d+(?:\.\d+)*)\b', re.IGNORECASE),
            re.compile(r'\b(?:subsection|subparagraph)\s+\(([a-z0-9]+)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Schedule|Appendix|Exhibit)\s+([A-Z\d]+)\b', re.IGNORECASE),
            re.compile(r'\b(?:above|below|herein|hereof|hereto|hereunder)\b', re.IGNORECASE),
        ]
    
    def extract_numbering(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract numbering from text"""
        for pattern_name, pattern in self.numbering_patterns.items():
            match = pattern.match(text.strip())
            if match:
                return match.group(1), match.group(2).strip()
        return None
    
    def identify_structure_type(self, text: str, context: str = "") -> StructureType:
        """Identify the structure type of a text segment"""
        text_lower = text.lower().strip()
        
        # Check for explicit structure indicators
        for structure_name, pattern in self.structure_patterns.items():
            if pattern.match(text):
                return StructureType(structure_name)
        
        # Check for common legal document structures
        if text_lower.startswith(('whereas', 'recital')):
            return StructureType.RECITAL
        elif text_lower.startswith('preamble'):
            return StructureType.PREAMBLE
        elif 'signature' in text_lower and ('date' in text_lower or 'sign' in text_lower):
            return StructureType.SIGNATURE_BLOCK
        elif text_lower.startswith(('schedule', 'appendix', 'exhibit')):
            return StructureType.SCHEDULE
        
        # Determine based on numbering and length
        numbering = self.extract_numbering(text)
        if numbering:
            number, content = numbering
            if '.' in number and len(number.split('.')) > 2:
                return StructureType.SUBCLAUSE
            elif '.' in number:
                return StructureType.CLAUSE
            elif len(content) > 200:
                return StructureType.SECTION
            else:
                return StructureType.SUBSECTION
        
        # Default based on length and context
        if len(text) < 100:
            return StructureType.HEADING
        elif len(text) < 500:
            return StructureType.PARAGRAPH
        else:
            return StructureType.SECTION
    
    def identify_clause_type(self, text: str) -> Tuple[ClauseType, float]:
        """Identify the type of a clause based on content"""
        text_lower = text.lower()
        scores = {}
        
        for clause_type, patterns in self.clause_indicators.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                # Normalize score by text length
                scores[clause_type] = score / (len(text) / 1000 + 1)
        
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 1.0)
            return best_type, confidence
        
        return ClauseType.OTHER, 0.5
    
    def extract_references(self, text: str) -> List[str]:
        """Extract references to other parts of the document"""
        references = []
        
        for pattern in self.reference_patterns:
            matches = pattern.findall(text)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms from text"""
        # Common legal terms and phrases
        legal_terms_patterns = [
            r'\b(?:agreement|contract|party|parties)\b',
            r'\b(?:breach|default|violation|non-compliance)\b',
            r'\b(?:damages|liability|indemnification|compensation)\b',
            r'\b(?:confidential|proprietary|intellectual\s+property)\b',
            r'\b(?:termination|expiration|renewal)\b',
            r'\b(?:governing\s+law|jurisdiction|dispute\s+resolution)\b',
            r'\b(?:force\s+majeure|act\s+of\s+god)\b',
            r'\b(?:warranty|representation|guarantee)\b',
            r'\b(?:assignment|transfer|delegation)\b',
            r'\b(?:notice|notification|communication)\b',
        ]
        
        key_terms = []
        text_lower = text.lower()
        
        for pattern in legal_terms_patterns:
            matches = re.findall(pattern, text_lower)
            key_terms.extend(matches)
        
        # Extract quoted terms (often definitions)
        quoted_terms = re.findall(r'"([^"]+)"', text)
        key_terms.extend(quoted_terms)
        
        # Extract capitalized terms (often defined terms)
        capitalized_terms = re.findall(r'\b[A-Z][A-Z\s]{2,}\b', text)
        key_terms.extend(capitalized_terms)
        
        return list(set(key_terms))  # Remove duplicates
class LLMClauseAnalyzer:
    """LLM-powered clause analysis for advanced identification"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def analyze_clause_content(self, clause_text: str, context: str = "") -> Dict[str, Any]:
        """Analyze clause content using LLM"""
        
        prompt = f"""
        Analyze the following legal/policy clause and extract structured information:

        Clause Text:
        {clause_text}

        Context (if available):
        {context}

        Please provide a JSON response with the following structure:
        {{
            "clause_type": "one of: definition, obligation, right, condition, term, payment, termination, liability, confidentiality, governing_law, force_majeure, other",
            "key_terms": ["list", "of", "important", "terms"],
            "obligations": ["list", "of", "obligations", "if", "any"],
            "conditions": ["list", "of", "conditions", "if", "any"],
            "references": ["list", "of", "references", "to", "other", "sections"],
            "summary": "brief summary of the clause",
            "confidence": 0.95
        }}

        Focus on legal and contractual language. Identify specific obligations, rights, conditions, and key terms.
        """
        
        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt="You are an expert legal analyst specializing in contract and policy analysis. Provide accurate, structured analysis of legal clauses.",
                max_tokens=500,
                temperature=0.1
            )
            
            response = self.llm_manager.generate(request)
            
            # Parse JSON response
            analysis = self._parse_llm_response(response.content)
            
            self.logger.debug(f"LLM clause analysis completed", 
                            clause_length=len(clause_text),
                            clause_type=analysis.get('clause_type', 'unknown'))
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"LLM clause analysis failed: {e}")
            return self._fallback_analysis(clause_text)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON")
            return self._fallback_analysis("")
    
    def _fallback_analysis(self, clause_text: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            "clause_type": "other",
            "key_terms": [],
            "obligations": [],
            "conditions": [],
            "references": [],
            "summary": "Analysis unavailable",
            "confidence": 0.3
        }


class ClauseStructureIdentifier:
    """Main class for clause and structure identification"""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager
        self.logger = get_pipeline_logger()
        self.pattern_matcher = LegalPatternMatcher()
        
        if llm_manager:
            self.llm_analyzer = LLMClauseAnalyzer(llm_manager)
        else:
            self.llm_analyzer = None
    
    @timing_decorator
    def identify_clauses_and_structure(self, content: str, document_id: str, 
                                     chunks: Optional[List[ContentChunk]] = None) -> DocumentStructure:
        """Main method to identify clauses and document structure"""
        
        self.logger.info(f"Starting clause and structure identification", 
                        document_id=document_id,
                        content_length=len(content))
        
        try:
            # Step 1: Extract basic document structure
            sections = self._extract_document_sections(content)
            
            # Step 2: Identify clauses from content or chunks
            if chunks:
                clauses = self._identify_clauses_from_chunks(chunks)
            else:
                clauses = self._identify_clauses_from_content(content, document_id)
            
            # Step 3: Enhance clause analysis with LLM if available
            if self.llm_analyzer:
                clauses = self._enhance_clauses_with_llm(clauses, content)
            
            # Step 4: Build document structure
            document_structure = DocumentStructure(
                document_id=document_id,
                title=self._extract_document_title(content),
                sections=sections,
                clauses=clauses,
                relationships=[],  # Simplified for now
                metadata={
                    "total_clauses": len(clauses),
                    "clause_types": self._get_clause_type_distribution(clauses),
                    "total_sections": len(sections),
                    "has_llm_analysis": self.llm_analyzer is not None,
                    "processing_time": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Clause and structure identification completed", 
                           document_id=document_id,
                           clauses_found=len(clauses))
            
            return document_structure
            
        except Exception as e:
            self.logger.error(f"Clause identification failed", document_id=document_id, error=str(e))
            raise ClauseExtractionError(f"Failed to identify clauses: {e}", document_id)
    
    def _extract_document_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract hierarchical document sections"""
        sections = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            structure_type = self.pattern_matcher.identify_structure_type(line)
            numbering = self.pattern_matcher.extract_numbering(line)
            
            if structure_type in [StructureType.SECTION, StructureType.ARTICLE, 
                                StructureType.HEADING, StructureType.SUBHEADING]:
                
                section = {
                    'title': line,
                    'structure_type': structure_type.value,
                    'line_number': i,
                    'numbering': numbering[0] if numbering else None,
                    'level': self._determine_section_level(line, numbering),
                }
                sections.append(section)
        
        return sections
    
    def _determine_section_level(self, line: str, numbering: Optional[Tuple[str, str]]) -> int:
        """Determine the hierarchical level of a section"""
        if numbering:
            number, _ = numbering
            return number.count('.') + 1
        
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ['article', 'chapter', 'part']):
            return 1
        elif any(keyword in line_lower for keyword in ['section']):
            return 2
        elif any(keyword in line_lower for keyword in ['subsection', 'clause']):
            return 3
        else:
            return 4
    
    def _identify_clauses_from_content(self, content: str, document_id: str) -> List[IdentifiedClause]:
        """Identify clauses directly from content"""
        clauses = []
        segments = self._segment_content_for_clauses(content)
        
        for segment in segments:
            if self._is_likely_clause(segment['content']):
                clause = self._create_clause_from_segment(segment, document_id)
                clauses.append(clause)
        
        return clauses
    
    def _identify_clauses_from_chunks(self, chunks: List[ContentChunk]) -> List[IdentifiedClause]:
        """Identify clauses from existing content chunks"""
        clauses = []
        
        for chunk in chunks:
            if self._is_likely_clause(chunk.content):
                clause = self._create_clause_from_chunk(chunk)
                clauses.append(clause)
        
        return clauses
    
    def _segment_content_for_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Segment content into potential clause units"""
        segments = []
        paragraphs = content.split('\n\n')
        
        position = 0
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 50:
                segments.append({
                    'content': para,
                    'start_position': position,
                    'end_position': position + len(para),
                    'numbering': self.pattern_matcher.extract_numbering(para)
                })
            position += len(para) + 2
        
        return segments
    
    def _is_likely_clause(self, text: str) -> bool:
        """Determine if text segment is likely a legal clause"""
        text_lower = text.lower()
        
        legal_indicators = [
            r'\b(?:shall|must|will|agrees?|undertakes?)\b',
            r'\b(?:party|parties|agreement|contract)\b',
            r'\b(?:liable|liability|responsible|damages)\b',
            r'\b(?:breach|default|violation|termination)\b',
            r'\b(?:confidential|proprietary|intellectual\s+property)\b',
            r'\b(?:payment|fee|cost|compensation)\b',
            r'\b(?:governing\s+law|jurisdiction|dispute)\b',
        ]
        
        indicator_count = sum(1 for pattern in legal_indicators 
                            if re.search(pattern, text_lower))
        
        return indicator_count >= 2 and 50 <= len(text) <= 2000
    
    def _create_clause_from_segment(self, segment: Dict[str, Any], document_id: str) -> IdentifiedClause:
        """Create IdentifiedClause from content segment"""
        content = segment['content']
        clause_type, confidence = self.pattern_matcher.identify_clause_type(content)
        structure_type = self.pattern_matcher.identify_structure_type(content)
        key_terms = self.pattern_matcher.extract_key_terms(content)
        references = self.pattern_matcher.extract_references(content)
        
        numbering_info = segment.get('numbering')
        numbering = numbering_info[0] if numbering_info else None
        title = numbering_info[1] if numbering_info else None
        
        return IdentifiedClause(
            id="",
            content=content,
            clause_type=clause_type,
            structure_type=structure_type,
            start_position=segment['start_position'],
            end_position=segment['end_position'],
            document_id=document_id,
            numbering=numbering,
            title=title,
            key_terms=key_terms,
            references=references,
            confidence=confidence
        )
    
    def _create_clause_from_chunk(self, chunk: ContentChunk) -> IdentifiedClause:
        """Create IdentifiedClause from ContentChunk"""
        content = chunk.content
        clause_type, confidence = self.pattern_matcher.identify_clause_type(content)
        
        if chunk.chunk_type.value in [e.value for e in StructureType]:
            structure_type = StructureType(chunk.chunk_type.value)
        else:
            structure_type = self.pattern_matcher.identify_structure_type(content)
        
        key_terms = self.pattern_matcher.extract_key_terms(content)
        references = self.pattern_matcher.extract_references(content)
        
        return IdentifiedClause(
            id="",
            content=content,
            clause_type=clause_type,
            structure_type=structure_type,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            document_id=chunk.document_id,
            section_path=[chunk.metadata.section_title] if chunk.metadata.section_title else [],
            key_terms=key_terms,
            references=references,
            confidence=confidence,
            metadata={
                'source_chunk_id': chunk.id,
                'page_number': chunk.metadata.page_number,
                'word_count': chunk.metadata.word_count
            }
        )
    
    def _enhance_clauses_with_llm(self, clauses: List[IdentifiedClause], content: str) -> List[IdentifiedClause]:
        """Enhance clause analysis using LLM"""
        enhanced_clauses = []
        
        for clause in clauses:
            try:
                context = self._get_clause_context(clause, content)
                llm_analysis = self.llm_analyzer.analyze_clause_content(clause.content, context)
                
                if llm_analysis.get('clause_type') != 'other':
                    try:
                        clause.clause_type = ClauseType(llm_analysis['clause_type'])
                    except ValueError:
                        pass
                
                clause.key_terms = list(set(clause.key_terms + llm_analysis.get('key_terms', [])))
                clause.obligations = llm_analysis.get('obligations', [])
                clause.conditions = llm_analysis.get('conditions', [])
                clause.references = list(set(clause.references + llm_analysis.get('references', [])))
                
                llm_confidence = llm_analysis.get('confidence', 0.5)
                clause.confidence = (clause.confidence + llm_confidence) / 2
                
                clause.metadata['llm_analysis'] = {
                    'summary': llm_analysis.get('summary', ''),
                    'llm_confidence': llm_confidence,
                    'analyzed_at': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed for clause {clause.id}: {e}")
            
            enhanced_clauses.append(clause)
        
        return enhanced_clauses
    
    def _get_clause_context(self, clause: IdentifiedClause, content: str, context_size: int = 500) -> str:
        """Get context around a clause for better LLM analysis"""
        start = max(0, clause.start_position - context_size)
        end = min(len(content), clause.end_position + context_size)
        return content[start:end]
    
    def _extract_document_title(self, content: str) -> Optional[str]:
        """Extract document title from content"""
        lines = content.split('\n')
        
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 200:
                if (line.isupper() or 
                    any(keyword in line.lower() for keyword in ['agreement', 'contract', 'policy', 'terms'])):
                    return line
        
        return None
    
    def _get_clause_type_distribution(self, clauses: List[IdentifiedClause]) -> Dict[str, int]:
        """Get distribution of clause types"""
        distribution = {}
        for clause in clauses:
            clause_type = clause.clause_type.value
            distribution[clause_type] = distribution.get(clause_type, 0) + 1
        return distribution