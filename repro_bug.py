from src.pipeline.stages.stage2_llm_parser.clause_identifier import ClauseStructureIdentifier

identifier = ClauseStructureIdentifier()

long_content = """
        ARTICLE I - DEFINITIONS
        
        1.1 The party shall comply with all terms and conditions.
        """ * 100

print(f"Content length: {len(long_content)}")
segments = identifier._segment_content_for_clauses(long_content)
print(f"Segments found: {len(segments)}")

clauses_found = 0
for i, seg in enumerate(segments[:5]): # Check first 5
    print(f"Segment {i}: {seg['content'][:50]}...")
    print(f"Length: {len(seg['content'])}")
    is_clause = identifier._is_likely_clause(seg['content'])
    print(f"Is likely clause: {is_clause}")
    if is_clause:
        clauses_found += 1

structure = identifier.identify_clauses_and_structure(long_content, "doc_123")
print(f"Total Clauses found: {len(structure.clauses)}")
