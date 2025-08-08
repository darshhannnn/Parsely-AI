import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from pydantic import BaseModel
from ..query_parsing.query_parser import ClaimQuery
from ..document_processing.document_processor import InsuranceDocumentProcessor

class PolicyClause(BaseModel):
    """Represents a policy clause with metadata"""
    clause_id: str
    text: str
    section: str
    policy_name: str
    relevance_score: float = 0.0
    clause_type: str = "general"  # coverage, exclusion, waiting_period, condition

class SemanticRetriever:
    """Semantic search engine for insurance policy clauses"""
    
    def __init__(self, data_path: str = "data/policies", embedding_model: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_processor = InsuranceDocumentProcessor(data_path)
        
        # Storage for processed clauses and embeddings
        self.clauses: List[PolicyClause] = []
        self.clause_embeddings = None
        self.index = None
        
        # Clause type patterns for classification
        self.clause_patterns = {
            'coverage': [
                r'covered', r'benefits?', r'eligible', r'reimburse', r'payable',
                r'sum insured', r'coverage', r'treatment.*covered'
            ],
            'exclusion': [
                r'excluded?', r'not covered', r'shall not', r'except', r'excluding',
                r'limitation', r'restriction', r'not eligible', r'not payable'
            ],
            'waiting_period': [
                r'waiting period', r'after.*months?', r'cooling period',
                r'initial.*period', r'moratorium', r'pre-existing.*months?'
            ],
            'condition': [
                r'condition', r'provided', r'subject to', r'terms',
                r'requirements?', r'criteria', r'qualification'
            ]
        }
        
    def initialize_index(self, force_rebuild: bool = False):
        """Initialize or load the semantic search index"""
        index_path = os.path.join(self.data_path, "../embeddings/clause_index.faiss")
        clauses_path = os.path.join(self.data_path, "../embeddings/clauses.pkl")
        
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(clauses_path):
            # Load existing index
            self._load_index(index_path, clauses_path)
        else:
            # Build new index
            self._build_index()
            self._save_index(index_path, clauses_path)
    
    def _build_index(self):
        """Build semantic search index from policy documents"""
        print("Building semantic search index...")
        
        # Process all policy documents
        file_paths = self.document_processor.get_file_paths()
        all_clauses = []
        
        if not file_paths:
            print("Warning: No policy documents found, creating sample clauses for testing")
            all_clauses = self._create_sample_clauses()
        else:
            for policy_name, file_path in file_paths.items():
                print(f"Processing {policy_name}...")
                
                try:
                    # Extract text from document
                    full_text = self.document_processor.extract_text_from_pdf(file_path)
                    
                    # Parse clauses from document
                    clauses = self._parse_clauses(full_text, policy_name)
                    all_clauses.extend(clauses)
                    print(f"  Extracted {len(clauses)} clauses from {policy_name}")
                    
                except Exception as e:
                    print(f"Warning: Failed to process {policy_name}: {e}")
                    # Add some sample clauses for this policy
                    sample_clauses = self._create_sample_clauses_for_policy(policy_name)
                    all_clauses.extend(sample_clauses)
        
        self.clauses = all_clauses
        
        # Generate embeddings
        clause_texts = [clause.text for clause in self.clauses]
        self.clause_embeddings = self.embedding_model.encode(clause_texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.clause_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.clause_embeddings)
        self.index.add(self.clause_embeddings.astype('float32'))
        
        print(f"Index built with {len(self.clauses)} clauses")
    
    def _parse_clauses(self, text: str, policy_name: str) -> List[PolicyClause]:
        """Parse clauses from policy document text with robust error handling"""
        clauses = []
        
        try:
            # Clean and normalize text
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('\n', ' ')
            
            # If text is too short, create a basic clause
            if len(text.strip()) < 100:
                return [PolicyClause(
                    clause_id=f"{policy_name}_BASIC_1",
                    text=text.strip() if text.strip() else "Basic policy clause",
                    section="General",
                    policy_name=policy_name,
                    clause_type="general"
                )]
            
            # Split into sections based on common patterns
            section_patterns = [
                r'(?:SECTION|Section|CHAPTER|Chapter)\s+(\d+(?:\.\d+)*)[:\s]*([A-Z][^.]*?)(?=(?:SECTION|Section|CHAPTER|Chapter|\Z))',
                r'(\d+(?:\.\d+)+)\s+([A-Z][^.]*?)(?=\d+(?:\.\d+)+|\Z)',
                r'([A-Z][A-Z\s]{10,}?)(?=[A-Z][A-Z\s]{10,}|\Z)'
            ]
            
            sections = []
            
            # Try to find sections using patterns
            for pattern in section_patterns:
                try:
                    matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        if len(match.groups()) >= 1:
                            if len(match.groups()) == 2:
                                section_id, section_text = match.groups()
                            else:
                                section_id = f"Section_{len(sections)+1}"
                                section_text = match.group(0)
                            sections.append((section_id.strip(), section_text.strip()))
                except Exception as e:
                    print(f"Warning: Pattern matching failed: {e}")
                    continue
            
            # If no sections found, create simple chunks
            if not sections:
                # Split text into chunks of reasonable size
                words = text.split()
                chunk_size = 200  # words per chunk
                for i in range(0, len(words), chunk_size):
                    chunk_text = ' '.join(words[i:i+chunk_size])
                    if len(chunk_text.strip()) > 50:
                        sections.append((f"Chunk_{i//chunk_size+1}", chunk_text))
            
            # Process each section
            for section_id, section_text in sections:
                if len(section_text.strip()) < 30:  # Skip very short sections
                    continue
                
                try:
                    # Simple approach: split into sentences and create clauses
                    clause_texts = self._split_into_clauses(section_text)
                    for i, clause_text in enumerate(clause_texts):
                        if len(clause_text.strip()) < 20:
                            continue
                        
                        clause_id = f"{section_id}_{i+1}"
                        clause_type = self._classify_clause(clause_text)
                        clause = PolicyClause(
                            clause_id=clause_id,
                            text=clause_text.strip(),
                            section=section_id,
                            policy_name=policy_name,
                            clause_type=clause_type
                        )
                        clauses.append(clause)
                        
                except Exception as e:
                    print(f"Warning: Failed to process section {section_id}: {e}")
                    # Create a single clause for the entire section
                    clause = PolicyClause(
                        clause_id=f"{section_id}_FULL",
                        text=section_text[:500] + "..." if len(section_text) > 500 else section_text,
                        section=section_id,
                        policy_name=policy_name,
                        clause_type="general"
                    )
                    clauses.append(clause)
            
            return clauses
            
        except Exception as e:
            print(f"Warning: Document parsing failed for {policy_name}: {e}")
            # Return a basic clause as fallback
            return [PolicyClause(
                clause_id=f"{policy_name}_FALLBACK_1",
                text=text[:500] + "..." if len(text) > 500 else text,
                section="General",
                policy_name=policy_name,
                clause_type="general"
            )]
    
    def _split_into_clauses(self, text: str) -> List[str]:
        """Split section text into individual clauses"""
        # Split by sentence endings with a simpler approach
        # First replace common abbreviations to avoid splitting on them
        text = text.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs').replace('Dr.', 'Dr')
        text = text.replace('vs.', 'vs').replace('etc.', 'etc').replace('i.e.', 'ie').replace('e.g.', 'eg')
        
        # Now split on periods followed by whitespace
        sentences = re.split(r'\.\s+', text)
        
        # Merge very short sentences with the next one
        merged_sentences = []
        i = 0
        while i < len(sentences):
            current = sentences[i].strip()
            if len(current) < 50 and i + 1 < len(sentences):
                # Merge with next sentence
                current += ". " + sentences[i + 1].strip()
                i += 2
            else:
                i += 1
            
            if current:
                merged_sentences.append(current)
        
        return merged_sentences
    
    def _classify_clause(self, text: str) -> str:
        """Classify clause type based on content patterns"""
        text_lower = text.lower()
        
        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return clause_type
        
        return "general"
    
    def search_relevant_clauses(self, query: ClaimQuery, top_k: int = 10) -> List[PolicyClause]:
        """Search for clauses relevant to the claim query"""
        if self.index is None:
            self.initialize_index()
        
        # Build search query from structured claim data
        search_queries = self._build_search_queries(query)
        
        all_results = []
        
        for search_query in search_queries:
            # Encode search query
            query_embedding = self.embedding_model.encode([search_query])
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Collect results
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.clauses):
                    clause = self.clauses[idx].model_copy()
                    clause.relevance_score = float(score)
                    all_results.append(clause)
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for clause in all_results:
            key = f"{clause.policy_name}_{clause.clause_id}"
            if key not in unique_results or clause.relevance_score > unique_results[key].relevance_score:
                unique_results[key] = clause
        
        # Sort by relevance score
        sorted_results = sorted(unique_results.values(), key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_results[:top_k]
    
    def _build_search_queries(self, query: ClaimQuery) -> List[str]:
        """Build multiple search queries from structured claim data"""
        queries = []
        
        # Main procedure query
        if query.procedure:
            main_query = f"{query.procedure}"
            if query.age:
                main_query += f" age {query.age}"
            if query.location:
                main_query += f" in {query.location}"
            queries.append(main_query)
        
        # Coverage query
        coverage_query = "coverage benefits eligible"
        if query.procedure:
            coverage_query += f" {query.procedure}"
        queries.append(coverage_query)
        
        # Exclusion query
        exclusion_query = "exclusion not covered excluded"
        if query.procedure:
            exclusion_query += f" {query.procedure}"
        queries.append(exclusion_query)
        
        # Waiting period query
        if query.policy_age_months:
            waiting_query = f"waiting period {query.policy_age_months} months"
            if query.procedure:
                waiting_query += f" {query.procedure}"
            queries.append(waiting_query)
        
        # Age-specific query
        if query.age:
            age_query = f"age {query.age} years old"
            if query.procedure:
                age_query += f" {query.procedure}"
            queries.append(age_query)
        
        # Location-specific query
        if query.location:
            location_query = f"treatment in {query.location} hospital network"
            queries.append(location_query)
        
        # Fallback query using raw text
        if not queries:
            queries.append(query.raw_query)
        
        return queries
    
    def _save_index(self, index_path: str, clauses_path: str):
        """Save the search index and clauses to disk"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save clauses
        with open(clauses_path, 'wb') as f:
            pickle.dump(self.clauses, f)
        
        print(f"Index saved to {index_path}")
    
    def _load_index(self, index_path: str, clauses_path: str):
        """Load the search index and clauses from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load clauses
        with open(clauses_path, 'rb') as f:
            self.clauses = pickle.load(f)
        
        print(f"Index loaded from {index_path} with {len(self.clauses)} clauses")
    
    def get_clauses_by_type(self, clause_type: str) -> List[PolicyClause]:
        """Get all clauses of a specific type"""
        return [clause for clause in self.clauses if clause.clause_type == clause_type]
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[PolicyClause]:
        """Search clauses by keywords (fallback method)"""
        results = []
        
        for clause in self.clauses:
            score = 0
            text_lower = clause.text.lower()
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                clause_copy = clause.model_copy()
                clause_copy.relevance_score = score / len(keywords)
                results.append(clause_copy)
        
        # Sort by score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    
    def _create_sample_clauses(self) -> List[PolicyClause]:
        """Create sample clauses for testing when no documents are available"""
        sample_clauses = []
        
        # Sample coverage clauses
        sample_clauses.extend([
            PolicyClause(
                clause_id="SAMPLE_COV_1",
                text="Orthopedic surgeries including knee replacement, hip replacement, and arthroscopy are covered under this policy subject to waiting period and sub-limits.",
                section="Coverage",
                policy_name="Sample_Policy",
                clause_type="coverage"
            ),
            PolicyClause(
                clause_id="SAMPLE_COV_2", 
                text="Maternity benefits including normal delivery and caesarean section are covered after completion of 10 months waiting period.",
                section="Maternity Benefits",
                policy_name="Sample_Policy",
                clause_type="coverage"
            ),
            PolicyClause(
                clause_id="SAMPLE_EXC_1",
                text="Pre-existing diseases are excluded for the first 3 years of the policy unless declared and accepted by the company.",
                section="Exclusions",
                policy_name="Sample_Policy", 
                clause_type="exclusion"
            ),
            PolicyClause(
                clause_id="SAMPLE_WAIT_1",
                text="Waiting period of 24 months applies for orthopedic procedures including knee and hip surgeries.",
                section="Waiting Periods",
                policy_name="Sample_Policy",
                clause_type="waiting_period"
            )
        ])
        
        return sample_clauses
    
    def _create_sample_clauses_for_policy(self, policy_name: str) -> List[PolicyClause]:
        """Create sample clauses for a specific policy"""
        return [
            PolicyClause(
                clause_id=f"{policy_name}_SAMPLE_1",
                text=f"This is a sample clause from {policy_name} policy for testing purposes.",
                section="Sample Section",
                policy_name=policy_name,
                clause_type="general"
            )
        ]
