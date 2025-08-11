"""
Integration test for the intelligent content chunker
"""

from src.pipeline.stages.stage2_llm_parser.content_chunker import (
    IntelligentContentChunker, ChunkingConfig, ChunkingStrategy, ChunkType
)

def test_content_chunker():
    """Test the content chunker with different strategies"""
    
    # Sample document content
    sample_content = """# Introduction

This is the introduction section of the document. It provides an overview of what will be covered in the following sections. The introduction sets the context and explains the purpose of the document.

## Background

The background section contains important context information. This information helps readers understand the motivation and context for the work described in this document. It includes relevant history and previous work in the field.

### Historical Context

The historical context provides a timeline of important developments. This helps readers understand how the current work fits into the broader landscape of research and development in this area.

## Methods

The methods section describes the approach taken. It includes detailed steps and procedures that were followed during the research or development process.

### Data Collection

Data was collected using various methods. The collection process was systematic and thorough to ensure data quality and completeness. Multiple sources were used to validate the data.

### Analysis

The analysis phase involved processing the collected data. Multiple analytical techniques were applied to extract meaningful insights from the raw data. The analysis was conducted using both quantitative and qualitative methods.

## Results

The results section presents the findings from the analysis. Key discoveries and patterns are highlighted and discussed in detail. The results are presented in a clear and organized manner.

### Key Findings

The key findings include several important discoveries:

1. First major finding about the data patterns
2. Second important discovery regarding correlations
3. Third significant result about implications

### Statistical Analysis

Statistical analysis revealed significant trends in the data. The analysis used appropriate statistical methods to ensure the validity of the conclusions.

## Discussion

The discussion section interprets the results and places them in context. It explains what the findings mean and how they relate to the original research questions.

## Conclusion

The conclusion summarizes the main findings and their implications. It also suggests areas for future research and development. The conclusion ties together all the main points from the document."""

    print("🚀 Testing Intelligent Content Chunker")
    
    # Test different chunking strategies
    strategies = [
        (ChunkingStrategy.PARAGRAPH_BASED, "Paragraph-based"),
        (ChunkingStrategy.FIXED_SIZE, "Fixed-size"),
        (ChunkingStrategy.SENTENCE_BASED, "Sentence-based"),
        (ChunkingStrategy.STRUCTURE_AWARE, "Structure-aware"),
        (ChunkingStrategy.HYBRID, "Hybrid")
    ]
    
    for strategy, name in strategies:
        print(f"\n📝 Testing {name} chunking...")
        
        config = ChunkingConfig(
            strategy=strategy,
            max_chunk_size=400,
            min_chunk_size=50,
            overlap_size=25,
            use_llm_for_semantic_boundaries=False  # Disable LLM for testing
        )
        
        chunker = IntelligentContentChunker(config)
        
        try:
            chunks = chunker.chunk_content(sample_content, "test_doc_123")
            
            print(f"✅ Generated {len(chunks)} chunks")
            print(f"   Average chunk size: {sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0} chars")
            print(f"   Chunk types: {set(c.chunk_type.value for c in chunks)}")
            
            # Validate chunks
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"   Chunk {i+1}: {len(chunk.content)} chars, type={chunk.chunk_type.value}")
                print(f"     Preview: {chunk.content[:100]}...")
                print(f"     Metadata: page={chunk.metadata.page_number}, section='{chunk.metadata.section_title}'")
                print(f"     Relationships: {len(chunk.metadata.relationships)}")
            
            # Test chunk serialization
            chunk_dict = chunks[0].to_dict()
            print(f"   Serialization: {len(chunk_dict)} fields")
            
        except Exception as e:
            print(f"❌ {name} chunking failed: {e}")
    
    # Test chunking configuration optimization
    print(f"\n🔧 Testing configuration optimization...")
    
    base_config = ChunkingConfig(strategy=ChunkingStrategy.HYBRID)
    chunker = IntelligentContentChunker(base_config)
    
    try:
        optimized_config = chunker.optimize_chunking_config(sample_content, target_chunk_count=8)
        print(f"✅ Optimized configuration:")
        print(f"   Original max chunk size: {base_config.max_chunk_size}")
        print(f"   Optimized max chunk size: {optimized_config.max_chunk_size}")
        print(f"   Optimized min chunk size: {optimized_config.min_chunk_size}")
        print(f"   Optimized overlap size: {optimized_config.overlap_size}")
        
        # Test with optimized config
        optimized_chunker = IntelligentContentChunker(optimized_config)
        optimized_chunks = optimized_chunker.chunk_content(sample_content, "test_doc_optimized")
        print(f"   Generated {len(optimized_chunks)} chunks with optimized config")
        
    except Exception as e:
        print(f"❌ Configuration optimization failed: {e}")
    
    # Test chunking statistics
    print(f"\n📊 Testing chunking statistics...")
    
    try:
        stats = chunker.get_chunking_stats()
        print(f"✅ Chunking statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
    
    print(f"\n🎉 Content chunker integration test completed!")

if __name__ == "__main__":
    test_content_chunker()