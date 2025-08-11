"""
Tests for Clause and Structure Identifier
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.pipeline.stages.stage2_llm_parser.clause_identifier import (
    ClauseType,
    StructureType,
    RelationshipType,
    ClauseRelationship,
    IdentifiedClause,
    DocumentStructure,
    LegalPatternMatcher,
    LLMClauseAnalyzer,
    ClauseStructureIdentifier
)
from src.pipeline.stages.stage2_llm_parser.llm_integration import (
    LLMManager,
    LLMRequest,
    LLMResponse
)
from src.pipeline.stages.stage2_llm_parser.content_chunker import (
    ContentChunk,
    ChunkType,
    ChunkMetadata
)
from src.pipeline.core.exceptions import ClauseExtractionError


class TestClauseRelationship:
    """Test clause relationship"""
    
    def test_clause_relationship_creation(self):
        """Test clause relationship creation"""
        relationship = ClauseRelationship(
            source_clause_id="clause_1",
            target_clause_id="clause_2",
            relationship_type=RelationshipType.DEPENDS_ON,
            confidence=0.85,
            description="Clause 1 depends on clause 2"
        )
        
        assert relationship.source_clause_id == "clause_1"
        assert relationship.target_clause_id == "clause_2"
        assert relationship.relationship_type == RelationshipType.DEPENDS_ON
        assert relationship.confidence == 0.85
        assert relationship.description == "Clause 1 depends on clause 2"
        assert isinstance(relationship.created_at, datetime)


class TestIdentifiedClause:
    """Test identified clause"""
    
    def test_identified_clause_creation(self):
        """Test identified clause creation"""
        clause = IdentifiedClause(
            id="test_clause_1",
            content="The party shall pay all fees within 30 days.",
            clause_type=ClauseType.PAYMENT,
            structure_type=StructureType.CLAUSE,
            start_position=100,
            end_position=150,
            document_id="doc_123",
            numbering="1.1",
            title="Payment Terms",
            key_terms=["payment", "fees", "30 days"],
            obligations=["pay all fees within 30 days"],
            confidence=0.9
        )
        
        assert clause.id == "test_clause_1"
        assert clause.content == "The party shall pay all fees within 30 days."
        assert clause.clause_type == ClauseType.PAYMENT
        assert clause.structure_type == StructureType.CLAUSE
        assert clause.start_position == 100
        assert clause.end_position == 150
        assert clause.document_id == "doc_123"
        assert clause.numbering == "1.1"
        assert clause.title == "Payment Terms"
        assert clause.key_terms == ["payment", "fees", "30 days"]
        assert clause.obligations == ["pay all fees within 30 days"]
        assert clause.confidence == 0.9
    
    def test_clause_id_generation(self):
        """Test automatic clause ID generation"""
        clause = IdentifiedClause(
            id="",  # Empty ID should trigger generation
            content="Test clause content",
            clause_type=ClauseType.OTHER,
            structure_type=StructureType.CLAUSE,
            start_position=0,
            end_position=20,
            document_id="doc_123"
        )
        
        assert clause.id != ""
        assert clause.id.startswith("doc_123_clause_0_")
        assert len(clause.id.split("_")) == 4
    
    def test_clause_to_dict(self):
        """Test clause serialization to dictionary"""
        clause = IdentifiedClause(
            id="test_clause",
            content="Test content",
            clause_type=ClauseType.OBLIGATION,
            structure_type=StructureType.PARAGRAPH,
            start_position=0,
            end_position=12,
            document_id="doc_123",
            key_terms=["test", "content"],
            obligations=["test obligation"]
        )
        
        clause_dict = clause.to_dict()
        
        assert clause_dict["id"] == "test_clause"
        assert clause_dict["content"] == "Test content"
        assert clause_dict["clause_type"] == "obligation"
        assert clause_dict["structure_type"] == "paragraph"
        assert clause_dict["key_terms"] == ["test", "content"]
        assert clause_dict["obligations"] == ["test obligation"]


class TestDocumentStructure:
    """Test document structure"""
    
    @pytest.fixture
    def sample_clauses(self):
        """Create sample clauses for testing"""
        return [
            IdentifiedClause(
                id="clause_1",
                content="Payment clause",
                clause_type=ClauseType.PAYMENT,
                structure_type=StructureType.CLAUSE,
                start_position=0,
                end_position=13,
                document_id="doc_123"
            ),
            IdentifiedClause(
                id="clause_2",
                content="Termination clause",
                clause_type=ClauseType.TERMINATION,
                structure_type=StructureType.CLAUSE,
                start_position=14,
                end_position=32,
                document_id="doc_123"
            )
        ]
    
    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing"""
        return [
            ClauseRelationship(
                source_clause_id="clause_1",
                target_clause_id="clause_2",
                relationship_type=RelationshipType.REFERENCES,
                confidence=0.8
            )
        ]
    
    def test_document_structure_creation(self, sample_clauses, sample_relationships):
        """Test document structure creation"""
        structure = DocumentStructure(
            document_id="doc_123",
            title="Test Agreement",
            clauses=sample_clauses,
            relationships=sample_relationships
        )
        
        assert structure.document_id == "doc_123"
        assert structure.title == "Test Agreement"
        assert len(structure.clauses) == 2
        assert len(structure.relationships) == 1
    
    def test_get_clause_by_id(self, sample_clauses):
        """Test getting clause by ID"""
        structure = DocumentStructure(
            document_id="doc_123",
            clauses=sample_clauses
        )
        
        clause = structure.get_clause_by_id("clause_1")
        assert clause is not None
        assert clause.id == "clause_1"
        assert clause.clause_type == ClauseType.PAYMENT
        
        # Test non-existent clause
        clause = structure.get_clause_by_id("non_existent")
        assert clause is None
    
    def test_get_clauses_by_type(self, sample_clauses):
        """Test getting clauses by type"""
        structure = DocumentStructure(
            document_id="doc_123",
            clauses=sample_clauses
        )
        
        payment_clauses = structure.get_clauses_by_type(ClauseType.PAYMENT)
        assert len(payment_clauses) == 1
        assert payment_clauses[0].clause_type == ClauseType.PAYMENT
        
        termination_clauses = structure.get_clauses_by_type(ClauseType.TERMINATION)
        assert len(termination_clauses) == 1
        assert termination_clauses[0].clause_type == ClauseType.TERMINATION
        
        # Test non-existent type
        definition_clauses = structure.get_clauses_by_type(ClauseType.DEFINITION)
        assert len(definition_clauses) == 0
    
    def test_get_related_clauses(self, sample_clauses, sample_relationships):
        """Test getting related clauses"""
        structure = DocumentStructure(
            document_id="doc_123",
            clauses=sample_clauses,
            relationships=sample_relationships
        )
        
        related = structure.get_related_clauses("clause_1")
        assert len(related) == 1
        assert related[0][0].id == "clause_2"
        assert related[0][1] == RelationshipType.REFERENCES
        
        # Test reverse relationship
        related = structure.get_related_clauses("clause_2")
        assert len(related) == 1
        assert related[0][0].id == "clause_1"
        assert related[0][1] == RelationshipType.REFERENCES


class TestLegalPatternMatcher:
    """Test legal pattern matcher"""
    
    @pytest.fixture
    def pattern_matcher(self):
        """Create pattern matcher"""
        return LegalPatternMatcher()
    
    def test_extract_numbering(self, pattern_matcher):
        """Test numbering extraction"""
        # Test decimal numbering
        result = pattern_matcher.extract_numbering("1.1 This is a section")
        assert result == ("1.1", "This is a section")
        
        # Test roman numeral
        result = pattern_matcher.extract_numbering("i. This is a subsection")
        assert result == ("i.", "This is a subsection")
        
        # Test letter numbering
        result = pattern_matcher.extract_numbering("a. This is a point")
        assert result == ("a.", "This is a point")
        
        # Test parenthetical
        result = pattern_matcher.extract_numbering("(a) This is a subpoint")
        assert result == ("a", "This is a subpoint")
        
        # Test no numbering
        result = pattern_matcher.extract_numbering("This has no numbering")
        assert result is None
    
    def test_identify_structure_type(self, pattern_matcher):
        """Test structure type identification"""
        # Test explicit structures
        assert pattern_matcher.identify_structure_type("ARTICLE I - Introduction") == StructureType.ARTICLE
        assert pattern_matcher.identify_structure_type("SECTION 1.1 - Terms") == StructureType.SECTION
        assert pattern_matcher.identify_structure_type("CLAUSE 2.3 - Payment") == StructureType.CLAUSE
        
        # Test special structures
        assert pattern_matcher.identify_structure_type("WHEREAS the parties agree") == StructureType.RECITAL
        assert pattern_matcher.identify_structure_type("SCHEDULE A - Pricing") == StructureType.SCHEDULE
        
        # Test based on numbering and length
        short_numbered = "1.1 Short"
        assert pattern_matcher.identify_structure_type(short_numbered) == StructureType.SUBSECTION
        
        long_numbered = "1.1 " + "Long content " * 50
        assert pattern_matcher.identify_structure_type(long_numbered) == StructureType.SECTION
        
        # Test default cases
        short_text = "Short heading"
        assert pattern_matcher.identify_structure_type(short_text) == StructureType.HEADING
        
        long_text = "This is a very long paragraph " * 20
        assert pattern_matcher.identify_structure_type(long_text) == StructureType.SECTION
    
    def test_identify_clause_type(self, pattern_matcher):
        """Test clause type identification"""
        # Test definition clause
        definition_text = "For the purposes of this agreement, 'Confidential Information' means any proprietary information."
        clause_type, confidence = pattern_matcher.identify_clause_type(definition_text)
        assert clause_type == ClauseType.DEFINITION
        assert confidence > 0
        
        # Test obligation clause
        obligation_text = "The party shall pay all fees within 30 days and must comply with all terms."
        clause_type, confidence = pattern_matcher.identify_clause_type(obligation_text)
        assert clause_type == ClauseType.OBLIGATION
        assert confidence > 0
        
        # Test payment clause
        payment_text = "Payment of $10,000 is due upon invoice and all costs shall be reimbursed."
        clause_type, confidence = pattern_matcher.identify_clause_type(payment_text)
        assert clause_type == ClauseType.PAYMENT
        assert confidence > 0
        
        # Test termination clause
        termination_text = "This agreement may be terminated upon breach or default by either party."
        clause_type, confidence = pattern_matcher.identify_clause_type(termination_text)
        assert clause_type == ClauseType.TERMINATION
        assert confidence > 0
        
        # Test generic text
        generic_text = "This is just some regular text without legal indicators."
        clause_type, confidence = pattern_matcher.identify_clause_type(generic_text)
        assert clause_type == ClauseType.OTHER
        assert confidence == 0.5
    
    def test_extract_references(self, pattern_matcher):
        """Test reference extraction"""
        text = "As stated in Section 1.2 and Article 3, see also Appendix A and subsection (b)."
        references = pattern_matcher.extract_references(text)
        
        assert "1.2" in references
        assert "3" in references
        assert "A" in references
        assert "b" in references
    
    def test_extract_key_terms(self, pattern_matcher):
        """Test key term extraction"""
        text = 'This agreement contains "Confidential Information" and covers liability, termination, and breach of contract.'
        key_terms = pattern_matcher.extract_key_terms(text)
        
        # Should extract quoted terms
        assert "Confidential Information" in key_terms
        
        # Should extract legal terms
        assert any("liability" in term for term in key_terms)
        assert any("termination" in term for term in key_terms)
        assert any("breach" in term for term in key_terms)


class TestLLMClauseAnalyzer:
    """Test LLM clause analyzer"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock(spec=LLMManager)
        
        # Mock response with clause analysis
        mock_response = Mock()
        mock_response.content = '''
        {
            "clause_type": "payment",
            "key_terms": ["payment", "fees", "30 days"],
            "obligations": ["pay all fees within 30 days"],
            "conditions": ["upon receipt of invoice"],
            "references": ["Section 1.1"],
            "summary": "Payment obligation clause",
            "confidence": 0.95
        }
        '''
        manager.generate.return_value = mock_response
        
        return manager
    
    @pytest.fixture
    def llm_analyzer(self, mock_llm_manager):
        """Create LLM clause analyzer"""
        return LLMClauseAnalyzer(mock_llm_manager)
    
    def test_analyze_clause_content(self, llm_analyzer, mock_llm_manager):
        """Test clause content analysis"""
        clause_text = "The party shall pay all fees within 30 days upon receipt of invoice."
        
        analysis = llm_analyzer.analyze_clause_content(clause_text)
        
        # Should call LLM manager
        mock_llm_manager.generate.assert_called_once()
        
        # Should return parsed analysis
        assert analysis["clause_type"] == "payment"
        assert "payment" in analysis["key_terms"]
        assert "pay all fees within 30 days" in analysis["obligations"]
        assert analysis["confidence"] == 0.95
    
    def test_parse_llm_response(self, llm_analyzer):
        """Test LLM response parsing"""
        # Test valid JSON response
        response_text = '''
        {
            "clause_type": "obligation",
            "key_terms": ["test"],
            "confidence": 0.8
        }
        '''
        
        analysis = llm_analyzer._parse_llm_response(response_text)
        assert analysis["clause_type"] == "obligation"
        assert analysis["confidence"] == 0.8
        
        # Test JSON embedded in text
        response_with_text = '''
        Here is the analysis:
        {
            "clause_type": "payment",
            "confidence": 0.9
        }
        That's the result.
        '''
        
        analysis = llm_analyzer._parse_llm_response(response_with_text)
        assert analysis["clause_type"] == "payment"
        assert analysis["confidence"] == 0.9
    
    def test_parse_invalid_response(self, llm_analyzer):
        """Test parsing invalid LLM response"""
        invalid_response = "This is not valid JSON"
        
        analysis = llm_analyzer._parse_llm_response(invalid_response)
        
        # Should return fallback analysis
        assert analysis["clause_type"] == "other"
        assert analysis["confidence"] == 0.3
    
    def test_llm_failure_fallback(self, mock_llm_manager, llm_analyzer):
        """Test fallback when LLM fails"""
        # Make LLM manager raise exception
        mock_llm_manager.generate.side_effect = Exception("LLM API error")
        
        clause_text = "Test clause"
        analysis = llm_analyzer.analyze_clause_content(clause_text)
        
        # Should return fallback analysis
        assert analysis["clause_type"] == "other"
        assert analysis["confidence"] == 0.3


class TestClauseStructureIdentifier:
    """Test clause structure identifier"""
    
    @pytest.fixture
    def basic_identifier(self):
        """Create basic identifier without LLM"""
        return ClauseStructureIdentifier()
    
    @pytest.fixture
    def llm_identifier(self, mock_llm_manager):
        """Create identifier with LLM"""
        return ClauseStructureIdentifier(mock_llm_manager)
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock(spec=LLMManager)
        mock_response = Mock()
        mock_response.content = '''
        {
            "clause_type": "obligation",
            "key_terms": ["shall", "comply"],
            "obligations": ["comply with terms"],
            "conditions": [],
            "references": [],
            "summary": "Compliance obligation",
            "confidence": 0.9
        }
        '''
        manager.generate.return_value = mock_response
        return manager
    
    @pytest.fixture
    def sample_legal_content(self):
        """Sample legal document content"""
        return """
        AGREEMENT FOR SERVICES

        ARTICLE I - DEFINITIONS

        1.1 For the purposes of this agreement, "Services" means the professional services described in Schedule A.

        1.2 "Confidential Information" means any proprietary information disclosed by either party.

        ARTICLE II - OBLIGATIONS

        2.1 The Provider shall perform all Services in accordance with industry standards.

        2.2 The Client shall pay all fees within 30 days of invoice receipt.

        2.3 Both parties shall maintain confidentiality of all Confidential Information.

        ARTICLE III - TERMINATION

        3.1 This agreement may be terminated by either party upon 30 days written notice.

        3.2 Upon termination, all obligations shall survive except for the payment obligations.
        """
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample content chunks"""
        return [
            ContentChunk(
                id="chunk_1",
                content="The Provider shall perform all Services in accordance with industry standards.",
                document_id="doc_123",
                chunk_type=ChunkType.PARAGRAPH,
                start_position=100,
                end_position=180,
                metadata=ChunkMetadata(section_title="Obligations")
            ),
            ContentChunk(
                id="chunk_2",
                content="The Client shall pay all fees within 30 days of invoice receipt.",
                document_id="doc_123",
                chunk_type=ChunkType.PARAGRAPH,
                start_position=200,
                end_position=265,
                metadata=ChunkMetadata(section_title="Payment")
            )
        ]
    
    def test_identifier_initialization(self, basic_identifier, llm_identifier):
        """Test identifier initialization"""
        # Basic identifier
        assert basic_identifier.llm_manager is None
        assert basic_identifier.llm_analyzer is None
        assert basic_identifier.pattern_matcher is not None
        
        # LLM identifier
        assert llm_identifier.llm_manager is not None
        assert llm_identifier.llm_analyzer is not None
        assert llm_identifier.pattern_matcher is not None
    
    def test_identify_clauses_and_structure_basic(self, basic_identifier, sample_legal_content):
        """Test basic clause and structure identification"""
        structure = basic_identifier.identify_clauses_and_structure(
            sample_legal_content, "doc_123"
        )
        
        assert structure.document_id == "doc_123"
        assert structure.title == "AGREEMENT FOR SERVICES"
        assert len(structure.sections) > 0
        assert len(structure.clauses) > 0
        assert structure.metadata["has_llm_analysis"] is False
        
        # Check that clauses were identified
        clause_types = [clause.clause_type for clause in structure.clauses]
        # Should have identified some legal clause types
        legal_clause_types = {ClauseType.DEFINITION, ClauseType.OBLIGATION, ClauseType.CONFIDENTIALITY, 
                             ClauseType.PAYMENT, ClauseType.TERMINATION, ClauseType.OTHER}
        assert any(ct in legal_clause_types for ct in clause_types)
    
    def test_identify_clauses_from_chunks(self, basic_identifier, sample_chunks):
        """Test clause identification from chunks"""
        clauses = basic_identifier._identify_clauses_from_chunks(sample_chunks)
        
        assert len(clauses) == 2
        assert all(isinstance(clause, IdentifiedClause) for clause in clauses)
        assert all(clause.document_id == "doc_123" for clause in clauses)
        
        # Check that clause types were identified
        clause_types = [clause.clause_type for clause in clauses]
        assert ClauseType.OBLIGATION in clause_types or ClauseType.OTHER in clause_types
    
    def test_extract_document_sections(self, basic_identifier, sample_legal_content):
        """Test document section extraction"""
        sections = basic_identifier._extract_document_sections(sample_legal_content)
        
        assert len(sections) > 0
        
        # Should find articles
        article_sections = [s for s in sections if 'ARTICLE' in s['title']]
        assert len(article_sections) >= 3  # ARTICLE I, II, III
        
        # Check section structure
        for section in sections:
            assert 'title' in section
            assert 'structure_type' in section
            assert 'level' in section
    
    def test_is_likely_clause(self, basic_identifier):
        """Test clause likelihood detection"""
        # Legal clause
        legal_text = "The party shall pay all fees and comply with the agreement terms upon breach."
        assert basic_identifier._is_likely_clause(legal_text) is True
        
        # Non-legal text
        non_legal_text = "This is just a regular paragraph without legal language."
        assert basic_identifier._is_likely_clause(non_legal_text) is False
        
        # Too short
        short_text = "Short text"
        assert basic_identifier._is_likely_clause(short_text) is False
        
        # Too long
        long_text = "Very long text " * 200
        assert basic_identifier._is_likely_clause(long_text) is False
    
    def test_enhance_clauses_with_llm(self, llm_identifier, sample_legal_content, mock_llm_manager):
        """Test LLM enhancement of clauses"""
        # First identify clauses without LLM
        basic_identifier = ClauseStructureIdentifier()
        clauses = basic_identifier._identify_clauses_from_content(sample_legal_content, "doc_123")
        
        # Then enhance with LLM
        enhanced_clauses = llm_identifier._enhance_clauses_with_llm(clauses, sample_legal_content)
        
        assert len(enhanced_clauses) == len(clauses)
        
        # Should have called LLM for analysis
        assert mock_llm_manager.generate.called
        
        # Check that clauses were enhanced
        for clause in enhanced_clauses:
            assert 'llm_analysis' in clause.metadata
            assert 'summary' in clause.metadata['llm_analysis']
    
    def test_extract_document_title(self, basic_identifier):
        """Test document title extraction"""
        # Test with clear title
        content_with_title = "SERVICE AGREEMENT\n\nThis agreement is between..."
        title = basic_identifier._extract_document_title(content_with_title)
        assert title == "SERVICE AGREEMENT"
        
        # Test with contract keyword
        content_with_contract = "Employment Contract\n\nThe parties agree..."
        title = basic_identifier._extract_document_title(content_with_contract)
        assert title == "Employment Contract"
        
        # Test with no clear title
        content_no_title = "This is just regular content without a title."
        title = basic_identifier._extract_document_title(content_no_title)
        assert title is None
    
    def test_get_clause_type_distribution(self, basic_identifier):
        """Test clause type distribution calculation"""
        clauses = [
            IdentifiedClause(
                id="c1", content="test", clause_type=ClauseType.PAYMENT,
                structure_type=StructureType.CLAUSE, start_position=0, end_position=4,
                document_id="doc"
            ),
            IdentifiedClause(
                id="c2", content="test", clause_type=ClauseType.PAYMENT,
                structure_type=StructureType.CLAUSE, start_position=5, end_position=9,
                document_id="doc"
            ),
            IdentifiedClause(
                id="c3", content="test", clause_type=ClauseType.TERMINATION,
                structure_type=StructureType.CLAUSE, start_position=10, end_position=14,
                document_id="doc"
            )
        ]
        
        distribution = basic_identifier._get_clause_type_distribution(clauses)
        
        assert distribution["payment"] == 2
        assert distribution["termination"] == 1
        assert len(distribution) == 2
    
    def test_error_handling(self, basic_identifier):
        """Test error handling"""
        # Test with empty content
        with pytest.raises(ClauseExtractionError):
            basic_identifier.identify_clauses_and_structure("", "doc_123")


class TestClauseIdentifierEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_content_with_no_legal_language(self):
        """Test content without legal language"""
        identifier = ClauseStructureIdentifier()
        
        non_legal_content = """
        This is a regular document.
        It contains normal paragraphs.
        There are no legal terms or clauses.
        Just regular text content.
        """
        
        structure = identifier.identify_clauses_and_structure(non_legal_content, "doc_123")
        
        # Should still create structure but with few/no clauses
        assert structure.document_id == "doc_123"
        assert len(structure.clauses) == 0  # No legal clauses identified
    
    def test_content_with_special_characters(self):
        """Test content with special characters and unicode"""
        identifier = ClauseStructureIdentifier()
        
        special_content = """
        ACUERDO DE SERVICIOS
        
        1.1 El proveedor deberá cumplir con todos los términos.
        
        § 2.1 Die Partei soll alle Gebühren zahlen.
        
        Article 3: The party shall pay €1,000 within 30 days.
        """
        
        structure = identifier.identify_clauses_and_structure(special_content, "doc_123")
        
        assert structure.document_id == "doc_123"
        assert len(structure.sections) > 0
        # Should handle unicode and special characters gracefully
    
    def test_very_long_document(self):
        """Test processing very long document"""
        identifier = ClauseStructureIdentifier()
        
        # Create long document
        long_content = """
        ARTICLE I - DEFINITIONS
        
        1.1 The party shall comply with all terms and conditions.
        """ * 100  # Repeat to create long document
        
        structure = identifier.identify_clauses_and_structure(long_content, "doc_123")
        
        assert structure.document_id == "doc_123"
        assert len(structure.clauses) > 0
        # Should handle long documents without performance issues
    
    def test_malformed_document_structure(self):
        """Test document with malformed structure"""
        identifier = ClauseStructureIdentifier()
        
        malformed_content = """
        1.1.1.1.1.1 Deep nesting
        
        ARTICLE
        
        Section without number
        
        (a) (b) (c) Missing content
        
        The party shall... incomplete clause
        """
        
        structure = identifier.identify_clauses_and_structure(malformed_content, "doc_123")
        
        # Should handle malformed structure gracefully
        assert structure.document_id == "doc_123"
        # May have fewer clauses due to malformed structure, but shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__])