"""
Prompt templates for document parsing and structuring
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ...core.models import ExtractedContent
from ...core.interfaces import DocumentType


class PromptType(Enum):
    """Types of prompts for different tasks"""
    DOCUMENT_ANALYSIS = "document_analysis"
    CONTENT_STRUCTURING = "content_structuring"
    CLAUSE_EXTRACTION = "clause_extraction"
    SECTION_IDENTIFICATION = "section_identification"
    METADATA_EXTRACTION = "metadata_extraction"
    CONTENT_SUMMARIZATION = "content_summarization"
    QUESTION_ANSWERING = "question_answering"


@dataclass
class PromptTemplate:
    """Template for LLM prompts"""
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_prompt_template: str
    expected_output_format: str
    variables: List[str]
    document_types: List[DocumentType]
    max_tokens: int = 4000
    temperature: float = 0.1
    stop_sequences: Optional[List[str]] = None


class PromptTemplateManager:
    """Manager for prompt templates"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default prompt templates"""
        
        # Document Analysis Template
        self.templates["document_analysis"] = PromptTemplate(
            name="document_analysis",
            prompt_type=PromptType.DOCUMENT_ANALYSIS,
            system_prompt="""You are an expert document analyst. Your task is to analyze documents and extract structured information. You should:

1. Identify the document type and purpose
2. Extract key themes and topics
3. Identify the document structure (sections, headings, etc.)
4. Assess the document's content quality and completeness
5. Provide insights about the document's context and significance

Always provide your analysis in a structured, JSON format as specified.""",
            user_prompt_template="""Please analyze the following {document_type} document and provide a comprehensive analysis:

Document Content:
{content}

Please provide your analysis in the following JSON format:
{{
    "document_type": "string",
    "primary_purpose": "string",
    "key_themes": ["theme1", "theme2", "theme3"],
    "document_structure": {{
        "has_clear_sections": boolean,
        "section_count": number,
        "heading_levels": number,
        "has_table_of_contents": boolean
    }},
    "content_quality": {{
        "completeness": "high|medium|low",
        "clarity": "high|medium|low",
        "organization": "high|medium|low"
    }},
    "key_information": {{
        "main_points": ["point1", "point2", "point3"],
        "important_dates": ["date1", "date2"],
        "key_entities": ["entity1", "entity2"],
        "action_items": ["action1", "action2"]
    }},
    "context_analysis": {{
        "target_audience": "string",
        "formality_level": "formal|informal|mixed",
        "document_tone": "string",
        "estimated_reading_time_minutes": number
    }}
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=3000,
            temperature=0.1
        )
        
        # Content Structuring Template
        self.templates["content_structuring"] = PromptTemplate(
            name="content_structuring",
            prompt_type=PromptType.CONTENT_STRUCTURING,
            system_prompt="""You are an expert at structuring and organizing document content. Your task is to:

1. Identify logical sections and subsections in the document
2. Create a hierarchical structure that reflects the document's organization
3. Extract and organize key information into structured categories
4. Maintain the original meaning while improving organization
5. Identify relationships between different parts of the document

Provide your output in a clear, structured JSON format.""",
            user_prompt_template="""Please structure the following {document_type} document content into a logical, hierarchical organization:

Document Content:
{content}

Please provide the structured content in the following JSON format:
{{
    "document_title": "string",
    "document_outline": [
        {{
            "section_number": "1",
            "section_title": "string",
            "section_content": "string",
            "subsections": [
                {{
                    "subsection_number": "1.1",
                    "subsection_title": "string",
                    "subsection_content": "string"
                }}
            ]
        }}
    ],
    "key_concepts": [
        {{
            "concept": "string",
            "definition": "string",
            "section_reference": "string"
        }}
    ],
    "cross_references": [
        {{
            "from_section": "string",
            "to_section": "string",
            "relationship_type": "string"
        }}
    ],
    "content_flow": {{
        "logical_sequence": ["section1", "section2", "section3"],
        "dependencies": [
            {{
                "section": "string",
                "depends_on": ["section1", "section2"]
            }}
        ]
    }}
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Clause Extraction Template (for legal/policy documents)
        self.templates["clause_extraction"] = PromptTemplate(
            name="clause_extraction",
            prompt_type=PromptType.CLAUSE_EXTRACTION,
            system_prompt="""You are a legal document expert specializing in clause extraction and analysis. Your task is to:

1. Identify all clauses, terms, and conditions in the document
2. Categorize clauses by type (obligations, rights, definitions, etc.)
3. Extract key legal concepts and their relationships
4. Identify potential conflicts or ambiguities
5. Assess the enforceability and clarity of clauses

Focus on precision and legal accuracy in your analysis.""",
            user_prompt_template="""Please extract and analyze all clauses from the following {document_type} document:

Document Content:
{content}

Please provide the clause analysis in the following JSON format:
{{
    "document_type": "contract|policy|agreement|terms_of_service|other",
    "clauses": [
        {{
            "clause_id": "string",
            "clause_type": "obligation|right|definition|condition|penalty|termination|other",
            "clause_text": "string",
            "parties_involved": ["party1", "party2"],
            "key_terms": ["term1", "term2"],
            "obligations": [
                {{
                    "party": "string",
                    "obligation": "string",
                    "conditions": ["condition1", "condition2"]
                }}
            ],
            "rights": [
                {{
                    "party": "string",
                    "right": "string",
                    "limitations": ["limitation1", "limitation2"]
                }}
            ],
            "enforceability": "high|medium|low",
            "clarity": "clear|ambiguous|unclear"
        }}
    ],
    "definitions": [
        {{
            "term": "string",
            "definition": "string",
            "clause_reference": "string"
        }}
    ],
    "key_dates": [
        {{
            "date_type": "effective_date|expiration_date|deadline|other",
            "date": "string",
            "description": "string"
        }}
    ],
    "potential_issues": [
        {{
            "issue_type": "ambiguity|conflict|missing_clause|unclear_terms",
            "description": "string",
            "affected_clauses": ["clause_id1", "clause_id2"],
            "severity": "high|medium|low"
        }}
    ]
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Section Identification Template
        self.templates["section_identification"] = PromptTemplate(
            name="section_identification",
            prompt_type=PromptType.SECTION_IDENTIFICATION,
            system_prompt="""You are an expert at identifying and categorizing document sections. Your task is to:

1. Identify all major sections and subsections in the document
2. Determine the hierarchical structure and relationships
3. Classify sections by their purpose and content type
4. Extract section metadata (length, importance, etc.)
5. Identify any missing or incomplete sections

Provide detailed section analysis with clear categorization.""",
            user_prompt_template="""Please identify and analyze all sections in the following {document_type} document:

Document Content:
{content}

Please provide the section analysis in the following JSON format:
{{
    "document_structure": {{
        "total_sections": number,
        "max_depth_level": number,
        "has_numbered_sections": boolean,
        "has_table_of_contents": boolean
    }},
    "sections": [
        {{
            "section_id": "string",
            "section_number": "string",
            "section_title": "string",
            "section_type": "introduction|methodology|results|conclusion|appendix|other",
            "hierarchy_level": number,
            "parent_section": "string",
            "content_preview": "string",
            "word_count": number,
            "importance": "high|medium|low",
            "completeness": "complete|incomplete|missing",
            "subsections": ["subsection_id1", "subsection_id2"]
        }}
    ],
    "section_relationships": [
        {{
            "from_section": "string",
            "to_section": "string",
            "relationship_type": "follows|references|depends_on|contradicts"
        }}
    ],
    "content_gaps": [
        {{
            "expected_section": "string",
            "reason": "string",
            "importance": "high|medium|low"
        }}
    ]
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=3500,
            temperature=0.1
        )
        
        # Metadata Extraction Template
        self.templates["metadata_extraction"] = PromptTemplate(
            name="metadata_extraction",
            prompt_type=PromptType.METADATA_EXTRACTION,
            system_prompt="""You are an expert at extracting metadata and structured information from documents. Your task is to:

1. Extract all available metadata from the document
2. Identify key entities (people, organizations, dates, locations)
3. Extract contact information and references
4. Identify document properties and characteristics
5. Extract any embedded structured data

Be thorough and accurate in your metadata extraction.""",
            user_prompt_template="""Please extract comprehensive metadata from the following {document_type} document:

Document Content:
{content}

Please provide the metadata in the following JSON format:
{{
    "document_metadata": {{
        "title": "string",
        "author": "string",
        "creation_date": "string",
        "last_modified": "string",
        "version": "string",
        "language": "string",
        "page_count": number,
        "word_count": number
    }},
    "entities": {{
        "people": [
            {{
                "name": "string",
                "role": "string",
                "contact_info": "string"
            }}
        ],
        "organizations": [
            {{
                "name": "string",
                "type": "string",
                "contact_info": "string"
            }}
        ],
        "locations": [
            {{
                "name": "string",
                "type": "city|state|country|address",
                "context": "string"
            }}
        ],
        "dates": [
            {{
                "date": "string",
                "type": "deadline|meeting|event|creation",
                "description": "string"
            }}
        ]
    }},
    "contact_information": [
        {{
            "type": "email|phone|address|website",
            "value": "string",
            "context": "string"
        }}
    ],
    "references": [
        {{
            "type": "citation|link|document_reference",
            "text": "string",
            "url": "string"
        }}
    ],
    "technical_details": {{
        "format_version": "string",
        "encoding": "string",
        "security_features": ["feature1", "feature2"],
        "embedded_objects": ["object1", "object2"]
    }}
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=3000,
            temperature=0.1
        )
        
        # Content Summarization Template
        self.templates["content_summarization"] = PromptTemplate(
            name="content_summarization",
            prompt_type=PromptType.CONTENT_SUMMARIZATION,
            system_prompt="""You are an expert at creating comprehensive yet concise summaries of documents. Your task is to:

1. Create multiple levels of summaries (executive, detailed, technical)
2. Extract and highlight the most important information
3. Maintain the original context and meaning
4. Identify key takeaways and action items
5. Provide different summary formats for different audiences

Ensure your summaries are accurate, well-structured, and useful.""",
            user_prompt_template="""Please create comprehensive summaries of the following {document_type} document:

Document Content:
{content}

Please provide the summaries in the following JSON format:
{{
    "executive_summary": {{
        "overview": "string",
        "key_points": ["point1", "point2", "point3"],
        "main_conclusion": "string",
        "word_count": number
    }},
    "detailed_summary": {{
        "introduction": "string",
        "main_content": [
            {{
                "section": "string",
                "summary": "string",
                "key_details": ["detail1", "detail2"]
            }}
        ],
        "conclusion": "string",
        "word_count": number
    }},
    "technical_summary": {{
        "methodology": "string",
        "technical_details": ["detail1", "detail2"],
        "specifications": ["spec1", "spec2"],
        "limitations": ["limitation1", "limitation2"]
    }},
    "key_takeaways": [
        {{
            "takeaway": "string",
            "importance": "high|medium|low",
            "category": "action_item|insight|recommendation|warning"
        }}
    ],
    "action_items": [
        {{
            "action": "string",
            "priority": "high|medium|low",
            "deadline": "string",
            "responsible_party": "string"
        }}
    ]
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=4000,
            temperature=0.2
        )
        
        # Question Answering Template
        self.templates["question_answering"] = PromptTemplate(
            name="question_answering",
            prompt_type=PromptType.QUESTION_ANSWERING,
            system_prompt="""You are an expert at answering questions based on document content. Your task is to:

1. Carefully analyze the provided document content
2. Answer questions accurately based only on the information in the document
3. Provide detailed explanations and cite specific sections
4. Indicate when information is not available in the document
5. Assess confidence levels for your answers

Always be precise, thorough, and honest about the limitations of the available information.""",
            user_prompt_template="""Based on the following {document_type} document, please answer the questions provided:

Document Content:
{content}

Questions:
{questions}

Please provide your answers in the following JSON format:
{{
    "answers": [
        {{
            "question": "string",
            "answer": "string",
            "confidence": "high|medium|low",
            "evidence": [
                {{
                    "text": "string",
                    "section": "string",
                    "relevance": "high|medium|low"
                }}
            ],
            "limitations": "string",
            "additional_context": "string"
        }}
    ],
    "document_coverage": {{
        "questions_fully_answered": number,
        "questions_partially_answered": number,
        "questions_not_answered": number,
        "overall_document_relevance": "high|medium|low"
    }},
    "related_information": [
        {{
            "topic": "string",
            "information": "string",
            "section": "string"
        }}
    ]
}}""",
            expected_output_format="JSON",
            variables=["document_type", "content", "questions"],
            document_types=[DocumentType.PDF, DocumentType.DOCX, DocumentType.EMAIL],
            max_tokens=4000,
            temperature=0.1
        )
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(template_name)
    
    def get_templates_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """Get all templates of a specific type"""
        return [template for template in self.templates.values() if template.prompt_type == prompt_type]
    
    def get_templates_for_document_type(self, document_type: DocumentType) -> List[PromptTemplate]:
        """Get all templates suitable for a specific document type"""
        return [template for template in self.templates.values() if document_type in template.document_types]
    
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template"""
        self.templates[template.name] = template
    
    def remove_template(self, template_name: str) -> bool:
        """Remove a prompt template"""
        if template_name in self.templates:
            del self.templates[template_name]
            return True
        return False
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def format_prompt(
        self, 
        template_name: str, 
        variables: Dict[str, Any],
        include_system_prompt: bool = True
    ) -> Dict[str, str]:
        """Format a prompt template with provided variables"""
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Check if all required variables are provided
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Format the user prompt
        formatted_user_prompt = template.user_prompt_template.format(**variables)
        
        result = {
            "user_prompt": formatted_user_prompt
        }
        
        if include_system_prompt:
            result["system_prompt"] = template.system_prompt
        
        return result
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a template"""
        template = self.get_template(template_name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "type": template.prompt_type.value,
            "variables": template.variables,
            "document_types": [dt.value for dt in template.document_types],
            "max_tokens": template.max_tokens,
            "temperature": template.temperature,
            "expected_output_format": template.expected_output_format,
            "stop_sequences": template.stop_sequences
        }


# Global prompt template manager instance
_global_prompt_manager: Optional[PromptTemplateManager] = None


def get_prompt_template_manager() -> PromptTemplateManager:
    """Get the global prompt template manager instance"""
    global _global_prompt_manager
    
    if _global_prompt_manager is None:
        _global_prompt_manager = PromptTemplateManager()
    
    return _global_prompt_manager