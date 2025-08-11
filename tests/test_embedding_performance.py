"""
Performance tests for the Embedding Generation System
"""

import pytest
import time
import statistics
from unittest.mock import Mock, patch
import numpy as np
import torch

from src.pipeline.stages.stage3_embedding_search.embedding_generator import EmbeddingGenerator
from src.pipeline.core.models import ContentChunk


class TestEmbeddingPerformance:
    """Performance test suite for EmbeddingGenerator"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for performance testing"""
        config = Mock()
        config.embedding.model_name = "all-MiniLM-L6-v2"
        config.embedding.model_cache_dir = "./temp_test_cache"
        config.embedding.batch_size = 32
        config.embedding.max_sequence_length = 512
        config.embedding.device = "cpu"
        config.database.cache_ttl_seconds = 3600
        return config
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer optimized for performance testing"""
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.max_seq_length = 512
        
        def mock_encode(texts, **kwargs):
            """Fast mock encoding that simulates real performance characteristics"""
            if isinstance(texts, str):
                texts = [texts]
            
            # Simulate processing time based on batch size
            batch_size = len(texts)
            base_time = 0.001  # Base time per text
            batch_efficiency = min(batch_size / 32, 1.0)  # Efficiency improves with batch size
            processing_time = base_time * batch_size * (1.0 - batch_efficiency * 0.5)
            time.sleep(processing_time)
            
            # Generate consistent embeddings
            embeddings = []
            for text in texts:
                # Create deterministic embedding based on text hash
                embedding = np.random.RandomState(hash(text) % 2**32).random(384).astype(np.float32)
                embeddings.append(embedding)
            
            if len(embeddings) == 1:
                return torch.tensor(embeddings[0])
            return torch.tensor(embeddings)
        
        mock_model.encode = mock_encode
        return mock_model
    
    def generate_test_content(self, count: int, avg_length: int = 100) -> list:
        """Generate test content of varying lengths"""
        contents = []
        for i in range(count):
            # Vary content length around average
            length = max(10, avg_length + np.random.randint(-50, 50))
            content = f"Test content {i}: " + "word " * (length // 5)
            contents.append(content)
        return contents
    
    def generate_test_chunks(self, count: int, avg_length: int = 100) -> list:
        """Generate test ContentChunk objects"""
        contents = self.generate_test_content(count, avg_length)
        chunks = []
        for i, content in enumerate(contents):
            chunk = ContentChunk(
                id=f"chunk_{i}",
                content=content,
                document_id=f"doc_{i // 10}",
                page_number=(i // 10) + 1
            )
            chunks.append(chunk)
        return chunks
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_single_embedding_performance(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test single embedding generation performance"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        content = "This is a test sentence for performance measurement."
        
        # Warm up
        generator.generate_single_embedding(content)
        
        # Measure performance
        times = []
        for _ in range(10):
            start_time = time.time()
            embedding = generator.generate_single_embedding(f"{content} {_}")
            end_time = time.time()
            times.append(end_time - start_time)
            
            assert len(embedding.vector) == 384
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"\nSingle embedding performance:")
        print(f"Average time: {avg_time:.4f}s")
        print(f"Standard deviation: {std_time:.4f}s")
        print(f"Min time: {min(times):.4f}s")
        print(f"Max time: {max(times):.4f}s")
        
        # Performance assertions
        assert avg_time < 0.1, f"Single embedding too slow: {avg_time:.4f}s"
        assert std_time < 0.05, f"Performance too inconsistent: {std_time:.4f}s"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_batch_embedding_performance(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test batch embedding generation performance"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32, 64]
        results = {}
        
        for batch_size in batch_sizes:
            contents = self.generate_test_content(batch_size)
            
            # Warm up
            generator.generate_batch_embeddings(contents[:min(2, len(contents))])
            
            # Measure performance
            start_time = time.time()
            result = generator.generate_batch_embeddings(contents)
            end_time = time.time()
            
            processing_time = end_time - start_time
            time_per_embedding = processing_time / batch_size
            
            results[batch_size] = {
                'total_time': processing_time,
                'time_per_embedding': time_per_embedding,
                'embeddings_count': len(result.embeddings),
                'cache_hits': result.cache_hits,
                'cache_misses': result.cache_misses
            }
            
            assert len(result.embeddings) == batch_size
            assert result.cache_misses == batch_size  # First run, no cache hits
        
        # Print performance results
        print(f"\nBatch embedding performance:")
        print(f"{'Batch Size':<10} {'Total Time':<12} {'Time/Embedding':<15} {'Efficiency':<10}")
        print("-" * 50)
        
        baseline_time_per_embedding = results[1]['time_per_embedding']
        
        for batch_size, result in results.items():
            efficiency = baseline_time_per_embedding / result['time_per_embedding']
            print(f"{batch_size:<10} {result['total_time']:<12.4f} {result['time_per_embedding']:<15.6f} {efficiency:<10.2f}x")
        
        # Performance assertions
        # Larger batches should be more efficient per embedding
        assert results[32]['time_per_embedding'] < results[1]['time_per_embedding'], \
            "Batch processing should be more efficient than single processing"
        
        # Total time should scale reasonably with batch size
        time_ratio = results[32]['total_time'] / results[1]['total_time']
        assert time_ratio < 20, f"Batch processing scaling too poor: {time_ratio:.2f}x"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_cache_performance_impact(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test performance impact of caching"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        contents = self.generate_test_content(20)
        
        # First run - populate cache
        start_time = time.time()
        result1 = generator.generate_batch_embeddings(contents)
        first_run_time = time.time() - start_time
        
        # Second run - should hit cache
        start_time = time.time()
        result2 = generator.generate_batch_embeddings(contents)
        second_run_time = time.time() - start_time
        
        cache_speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
        
        print(f"\nCache performance impact:")
        print(f"First run (cache miss): {first_run_time:.4f}s")
        print(f"Second run (cache hit): {second_run_time:.4f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")
        print(f"Cache hit rate: {result2.cache_hits / (result2.cache_hits + result2.cache_misses):.2%}")
        
        # Performance assertions
        assert result1.cache_hits == 0, "First run should have no cache hits"
        assert result2.cache_hits == len(contents), "Second run should have all cache hits"
        assert cache_speedup > 2, f"Cache should provide significant speedup: {cache_speedup:.2f}x"
        assert second_run_time < first_run_time * 0.5, "Cached run should be at least 2x faster"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_content_chunk_processing_performance(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test performance of processing ContentChunk objects"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Test different chunk counts
        chunk_counts = [10, 50, 100, 200]
        results = {}
        
        for count in chunk_counts:
            chunks = self.generate_test_chunks(count)
            
            # Measure performance
            start_time = time.time()
            embeddings = generator.create_embeddings(chunks)
            end_time = time.time()
            
            processing_time = end_time - start_time
            time_per_chunk = processing_time / count
            
            results[count] = {
                'total_time': processing_time,
                'time_per_chunk': time_per_chunk,
                'embeddings_count': len(embeddings)
            }
            
            assert len(embeddings) == count
            
            # Verify chunk IDs are properly assigned
            for chunk, embedding in zip(chunks, embeddings):
                assert embedding.chunk_id == chunk.id
        
        # Print performance results
        print(f"\nContentChunk processing performance:")
        print(f"{'Chunk Count':<12} {'Total Time':<12} {'Time/Chunk':<12} {'Throughput':<12}")
        print("-" * 50)
        
        for count, result in results.items():
            throughput = count / result['total_time']
            print(f"{count:<12} {result['total_time']:<12.4f} {result['time_per_chunk']:<12.6f} {throughput:<12.1f}/s")
        
        # Performance assertions
        # Should process at least 10 chunks per second
        min_throughput = 10
        for count, result in results.items():
            throughput = count / result['total_time']
            assert throughput >= min_throughput, \
                f"Throughput too low for {count} chunks: {throughput:.1f}/s < {min_throughput}/s"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_memory_usage_scaling(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test memory usage scaling with batch size"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Test memory usage with different batch sizes
        batch_sizes = [10, 50, 100, 200]
        
        for batch_size in batch_sizes:
            contents = self.generate_test_content(batch_size, avg_length=200)
            
            # Clear cache to ensure consistent memory usage
            generator.clear_cache()
            
            # Process batch
            result = generator.generate_batch_embeddings(contents)
            
            # Check memory cache size
            stats = generator.get_embedding_stats()
            memory_cache_size = stats['memory_cache_size']
            
            print(f"Batch size {batch_size}: Memory cache entries = {memory_cache_size}")
            
            assert len(result.embeddings) == batch_size
            assert memory_cache_size == batch_size, \
                f"Memory cache size mismatch: {memory_cache_size} != {batch_size}"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_concurrent_processing_simulation(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test performance under simulated concurrent load"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Simulate concurrent requests by processing multiple batches rapidly
        batch_count = 10
        batch_size = 20
        
        total_start_time = time.time()
        all_results = []
        
        for i in range(batch_count):
            contents = self.generate_test_content(batch_size)
            
            start_time = time.time()
            result = generator.generate_batch_embeddings(contents)
            end_time = time.time()
            
            all_results.append({
                'batch_id': i,
                'processing_time': end_time - start_time,
                'embeddings_count': len(result.embeddings),
                'cache_hits': result.cache_hits,
                'cache_misses': result.cache_misses
            })
        
        total_time = time.time() - total_start_time
        
        # Calculate statistics
        processing_times = [r['processing_time'] for r in all_results]
        avg_processing_time = statistics.mean(processing_times)
        std_processing_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        
        total_embeddings = sum(r['embeddings_count'] for r in all_results)
        overall_throughput = total_embeddings / total_time
        
        print(f"\nConcurrent processing simulation:")
        print(f"Total batches: {batch_count}")
        print(f"Batch size: {batch_size}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Overall throughput: {overall_throughput:.1f} embeddings/s")
        print(f"Average batch time: {avg_processing_time:.4f}s")
        print(f"Batch time std dev: {std_processing_time:.4f}s")
        
        # Performance assertions
        assert overall_throughput > 50, f"Overall throughput too low: {overall_throughput:.1f}/s"
        assert std_processing_time < avg_processing_time * 0.5, \
            f"Processing time too inconsistent: {std_processing_time:.4f}s"
        
        # Check that all batches completed successfully
        for result in all_results:
            assert result['embeddings_count'] == batch_size, \
                f"Batch {result['batch_id']} incomplete: {result['embeddings_count']}/{batch_size}"
    
    @patch('src.pipeline.core.config.get_config')
    @patch('src.pipeline.stages.stage3_embedding_search.embedding_generator.SentenceTransformer')
    def test_cache_cleanup_performance(self, mock_st_class, mock_get_config, mock_config, mock_sentence_transformer):
        """Test performance of cache cleanup operations"""
        mock_get_config.return_value = mock_config
        mock_st_class.return_value = mock_sentence_transformer
        
        generator = EmbeddingGenerator()
        
        # Populate cache with many entries
        large_batch_size = 500
        contents = self.generate_test_content(large_batch_size)
        generator.generate_batch_embeddings(contents)
        
        # Measure cache cleanup performance
        start_time = time.time()
        removed_count = generator.cleanup_expired_cache()
        cleanup_time = time.time() - start_time
        
        print(f"\nCache cleanup performance:")
        print(f"Cache entries before cleanup: {large_batch_size}")
        print(f"Entries removed: {removed_count}")
        print(f"Cleanup time: {cleanup_time:.4f}s")
        
        # Measure cache clear performance
        start_time = time.time()
        generator.clear_cache()
        clear_time = time.time() - start_time
        
        print(f"Cache clear time: {clear_time:.4f}s")
        
        # Performance assertions
        assert cleanup_time < 1.0, f"Cache cleanup too slow: {cleanup_time:.4f}s"
        assert clear_time < 0.1, f"Cache clear too slow: {clear_time:.4f}s"
        
        # Verify cache is actually cleared
        stats = generator.get_embedding_stats()
        assert stats['memory_cache_size'] == 0, "Cache not properly cleared"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])