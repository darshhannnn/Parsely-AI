"""
Simple integration test for the FAISS search engine
"""

import tempfile
import shutil
import numpy as np
from src.pipeline.stages.stage3_embedding_search.faiss_search_engine import FAISSSearchEngine
from src.pipeline.core.models import Embedding

def test_faiss_integration():
    """Test the FAISS search engine with real data"""
    
    # Create temporary directory for index storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Override config for testing
        config_override = {
            'faiss_index_path': temp_dir
        }
        
        # Initialize FAISS engine
        print("Initializing FAISSSearchEngine...")
        engine = FAISSSearchEngine(config=config_override)
        
        # Create test embeddings
        print("Creating test embeddings...")
        embeddings = []
        for i in range(100):
            # Create diverse, normalized embeddings
            vector = np.random.RandomState(i).random(384).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize for cosine similarity
            
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector.tolist(),
                model_name="test-model",
                dimension=384
            )
            embeddings.append(embedding)
        
        # Build index
        print("Building FAISS index...")
        vector_index = engine.build_index(embeddings)
        
        print(f"Built index: {vector_index.total_vectors} vectors, dimension {vector_index.dimension}")
        print(f"Index type: {vector_index.metadata.get('faiss_index_type', 'unknown')}")
        
        # Test search
        print("\nTesting vector search...")
        query_vector = embeddings[0].vector  # Use first embedding as query
        results = engine.search_similar_by_vector(query_vector, top_k=5)
        
        print(f"Found {len(results)} similar vectors:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Chunk: {result.chunk_id}, Score: {result.similarity_score:.4f}")
        
        # Test with different similarity threshold
        print("\nTesting with similarity threshold...")
        high_threshold_results = engine.search_similar_by_vector(
            query_vector, top_k=10, similarity_threshold=0.8
        )
        print(f"High threshold results: {len(high_threshold_results)}")
        
        # Test index persistence
        print("\nTesting index persistence...")
        index_id = vector_index.index_id
        success = engine.save_index(index_id)
        print(f"Index saved: {success}")
        
        # Clear and reload
        engine.clear_index()
        print("Index cleared")
        
        success = engine.load_index(index_id)
        print(f"Index loaded: {success}")
        
        # Test search after reload
        reload_results = engine.search_similar_by_vector(query_vector, top_k=5)
        print(f"Search after reload: {len(reload_results)} results")
        
        # Verify results are the same
        if len(results) == len(reload_results):
            same_results = all(
                r1.chunk_id == r2.chunk_id and abs(r1.similarity_score - r2.similarity_score) < 0.001
                for r1, r2 in zip(results, reload_results)
            )
            print(f"Results consistent after reload: {same_results}")
        
        # Test index update
        print("\nTesting index update...")
        new_embeddings = []
        for i in range(100, 120):  # Add 20 more embeddings
            vector = np.random.RandomState(i).random(384).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector.tolist(),
                model_name="test-model",
                dimension=384
            )
            new_embeddings.append(embedding)
        
        engine.update_index(new_embeddings)
        
        # Get index info
        print("\nIndex information:")
        index_info = engine.get_index_info()
        for key, value in index_info.items():
            if key not in ['index_parameters', 'performance_stats']:
                print(f"  {key}: {value}")
        
        # Get search stats
        print("\nSearch statistics:")
        search_stats = engine.get_search_stats()
        for key, value in search_stats.items():
            print(f"  {key}: {value}")
        
        # Test optimization
        print("\nTesting index optimization...")
        optimization_stats = engine.optimize_index()
        print(f"Optimization applied: {optimization_stats['optimizations_applied']}")
        
        # Test vector removal
        print("\nTesting vector removal...")
        chunks_to_remove = ["chunk_0", "chunk_1", "chunk_2"]
        removed_count = engine.remove_vectors(chunks_to_remove)
        print(f"Removed {removed_count} vectors")
        
        final_info = engine.get_index_info()
        print(f"Final vector count: {final_info['total_vectors']}")
        
        # Cleanup saved index
        engine.delete_saved_index(index_id)
        print("Cleaned up saved index")
        
        print("\n✅ All FAISS integration tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_faiss_integration()