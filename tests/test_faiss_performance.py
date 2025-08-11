"""
Performance tests for the FAISS Search Engine
"""

import pytest
import time
import statistics
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.pipeline.stages.stage3_embedding_search.faiss_search_engine import FAISSSearchEngine
from src.pipeline.core.models import Embedding


class TestFAISSPerformance:
    """Performance test suite for FAISSSearchEngine"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Mock configuration for performance testing"""
        config = Mock()
        config.database.faiss_index_path = temp_dir
        config.database.vector_db_type = "faiss"
        return config
    
    def generate_test_embeddings(self, count: int, dimension: int = 384) -> list:
        """Generate test embeddings for performance testing"""
        embeddings = []
        for i in range(count):
            # Create diverse embeddings for realistic testing
            vector = np.random.RandomState(i).random(dimension).astype(np.float32)
            # Add some structure to make embeddings more realistic
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            embedding = Embedding(
                chunk_id=f"chunk_{i}",
                vector=vector.tolist(),
                model_name="test-model",
                dimension=dimension
            )
            embeddings.append(embedding)
        return embeddings
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_build_performance(self, mock_get_config, mock_config):
        """Test index building performance with different sizes"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        sizes = [100, 500, 1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            embeddings = self.generate_test_embeddings(size)
            
            # Measure build time
            start_time = time.time()
            vector_index = engine.build_index(embeddings)
            build_time = time.time() - start_time
            
            # Calculate metrics
            vectors_per_second = size / build_time if build_time > 0 else float('inf')
            memory_usage = size * 384 * 4  # Approximate bytes
            
            results[size] = {
                'build_time': build_time,
                'vectors_per_second': vectors_per_second,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'index_type': vector_index.metadata.get('faiss_index_type', 'unknown')
            }
            
            assert vector_index.total_vectors == size
            
            # Clear for next test
            engine.clear_index()
        
        # Print performance results
        print(f"\nIndex Build Performance:")
        print(f"{'Size':<8} {'Build Time':<12} {'Vectors/sec':<12} {'Memory MB':<12} {'Index Type':<12}")
        print("-" * 60)
        
        for size, result in results.items():
            print(f"{size:<8} {result['build_time']:<12.3f} {result['vectors_per_second']:<12.1f} "
                  f"{result['memory_usage_mb']:<12.1f} {result['index_type']:<12}")
        
        # Performance assertions
        # Build time should scale reasonably
        assert results[10000]['build_time'] < 60, "Large index build too slow"
        
        # Should achieve reasonable throughput
        for size, result in results.items():
            if size >= 1000:  # Only check for larger sizes
                assert result['vectors_per_second'] > 100, f"Build throughput too low for size {size}"
    
    @patch('src.pipeline.core.config.get_config')
    def test_search_performance(self, mock_get_config, mock_config):
        """Test search performance with different index sizes and configurations"""
        mock_get_config.return_value = mock_config
        
        engine = FAISSSearchEngine()
        sizes = [1000, 5000, 10000]
        top_k_values = [1, 5, 10, 50]
        results = {}
        
        for size in sizes:
            embeddings = self.generate_test_embeddings(size)
            engine.build_index(embeddings)
            
            size_results = {}
            
            for top_k in top_k_values:
                # Use a random query vector
                query_vector = embeddings[size // 2].vector  # Middle embedding
                
                # Warm up
                engine.search_similar_by_vector(query_vector, top_k=top_k)
                
                # Measure search performance
                search_times = []
                for _ in range(10):  # Multiple runs for average
                    start_time = time.time()
                    search_results = engine.search_similar_by_vector(query_vector, top_k=top_k)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                    
                    assert len(search_results) <= top_k
                
                avg_search_time = statistics.mean(search_times)
                std_search_time = statistics.stdev(search_times) if len(search_times) > 1 else 0
                
                size_results[top_k] = {
                    'avg_time': avg_search_time,
                    'std_time': std_search_time,
                    'min_time': min(search_times),
                    'max_time': max(search_times),
                    'searches_per_second': 1.0 / avg_search_time if avg_search_time > 0 else float('inf')
                }
            
            results[size] = size_results
            engine.clear_index()
        
        # Print performance results
        print(f"\nSearch Performance:")
        for size in sizes:
            print(f"\nIndex Size: {size} vectors")
            print(f"{'Top-K':<8} {'Avg Time':<12} {'Std Time':<12} {'Min Time':<12} {'Max Time':<12} {'Searches/sec':<12}")
            print("-" * 72)
            
            for top_k, result in results[size].items():
                print(f"{top_k:<8} {result['avg_time']:<12.6f} {result['std_time']:<12.6f} "
                      f"{result['min_time']:<12.6f} {result['max_time']:<12.6f} {result['searches_per_second']:<12.1f}")
        
        # Performance assertions
        for size in sizes:
            for top_k in top_k_values:
                result = results[size][top_k]
                # Search should be fast
                assert result['avg_time'] < 0.1, f"Search too slow for size {size}, top_k {top_k}"
                # Should achieve good throughput
                assert result['searches_per_second'] > 10, f"Search throughput too low for size {size}, top_k {top_k}"
                # Should be consistent
                assert result['std_time'] < result['avg_time'], f"Search time too inconsistent for size {size}, top_k {top_k}"
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_type_performance_comparison(self, mock_get_config, mock_config):
        """Compare performance of different FAISS index types"""
        mock_get_config.return_value = mock_config
        
        # Generate test data
        size = 5000
        embeddings = self.generate_test_embeddings(size)
        query_vector = embeddings[0].vector
        
        # Test different index types
        index_types = ['flat', 'ivf', 'hnsw']
        results = {}
        
        for index_type in index_types:
            engine = FAISSSearchEngine()
            
            # Force specific index type by modifying the internal method
            original_create_index = engine._create_index
            
            def force_index_type(dimension, forced_type=index_type, **kwargs):
                return original_create_index(dimension, forced_type, **kwargs)
            
            engine._create_index = force_index_type
            
            try:
                # Build index
                start_time = time.time()
                vector_index = engine.build_index(embeddings)
                build_time = time.time() - start_time
                
                # Test search performance
                search_times = []
                for _ in range(5):
                    start_time = time.time()
                    search_results = engine.search_similar_by_vector(query_vector, top_k=10)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                
                avg_search_time = statistics.mean(search_times)
                
                results[index_type] = {
                    'build_time': build_time,
                    'avg_search_time': avg_search_time,
                    'build_throughput': size / build_time if build_time > 0 else float('inf'),
                    'search_throughput': 1.0 / avg_search_time if avg_search_time > 0 else float('inf'),
                    'memory_usage': engine.get_index_info().get('memory_usage_bytes', 0) / (1024 * 1024)
                }
                
            except Exception as e:
                print(f"Failed to test {index_type}: {e}")
                results[index_type] = None
            
            engine.clear_index()
        
        # Print comparison results
        print(f"\nIndex Type Performance Comparison ({size} vectors):")
        print(f"{'Type':<8} {'Build Time':<12} {'Search Time':<12} {'Build Tput':<12} {'Search Tput':<12} {'Memory MB':<12}")
        print("-" * 72)
        
        for index_type, result in results.items():
            if result:
                print(f"{index_type:<8} {result['build_time']:<12.3f} {result['avg_search_time']:<12.6f} "
                      f"{result['build_throughput']:<12.1f} {result['search_throughput']:<12.1f} {result['memory_usage']:<12.1f}")
            else:
                print(f"{index_type:<8} {'FAILED':<12}")
        
        # Performance assertions
        valid_results = {k: v for k, v in results.items() if v is not None}
        assert len(valid_results) > 0, "No index types worked"
        
        # Flat should be most accurate but potentially slower for large datasets
        if 'flat' in valid_results:
            assert valid_results['flat']['build_time'] < 30, "Flat index build too slow"
        
        # IVF should be faster for search on large datasets
        if 'ivf' in valid_results:
            assert valid_results['ivf']['avg_search_time'] < 0.1, "IVF search too slow"
    
    @patch('src.pipeline.core.config.get_config')
    def test_concurrent_search_performance(self, mock_get_config, mock_config):
        """Test search performance under concurrent load"""
        import threading
        import queue
        
        mock_get_config.return_value = mock_config
        
        # Build index
        size = 2000
        embeddings = self.generate_test_embeddings(size)
        engine = FAISSSearchEngine()
        engine.build_index(embeddings)
        
        # Concurrent search test
        num_threads = 10
        searches_per_thread = 20
        results_queue = queue.Queue()
        
        def search_worker(worker_id):
            worker_results = []
            query_vector = embeddings[worker_id % len(embeddings)].vector
            
            for i in range(searches_per_thread):
                start_time = time.time()
                try:
                    search_results = engine.search_similar_by_vector(query_vector, top_k=5)
                    search_time = time.time() - start_time
                    worker_results.append({
                        'worker_id': worker_id,
                        'search_id': i,
                        'search_time': search_time,
                        'results_count': len(search_results),
                        'success': True
                    })
                except Exception as e:
                    worker_results.append({
                        'worker_id': worker_id,
                        'search_id': i,
                        'error': str(e),
                        'success': False
                    })
            
            results_queue.put(worker_results)
        
        # Start concurrent searches
        start_time = time.time()
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=search_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            worker_results = results_queue.get()
            all_results.extend(worker_results)
        
        # Analyze results
        successful_searches = [r for r in all_results if r['success']]
        failed_searches = [r for r in all_results if not r['success']]
        
        if successful_searches:
            search_times = [r['search_time'] for r in successful_searches]
            avg_search_time = statistics.mean(search_times)
            total_searches = len(successful_searches)
            overall_throughput = total_searches / total_time
            
            print(f"\nConcurrent Search Performance:")
            print(f"Threads: {num_threads}")
            print(f"Searches per thread: {searches_per_thread}")
            print(f"Total searches: {total_searches}")
            print(f"Failed searches: {len(failed_searches)}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Average search time: {avg_search_time:.6f}s")
            print(f"Overall throughput: {overall_throughput:.1f} searches/sec")
            print(f"Success rate: {len(successful_searches) / len(all_results) * 100:.1f}%")
            
            # Performance assertions
            assert len(failed_searches) == 0, f"Some searches failed: {failed_searches[:3]}"
            assert avg_search_time < 0.1, f"Average search time too slow: {avg_search_time:.6f}s"
            assert overall_throughput > 50, f"Overall throughput too low: {overall_throughput:.1f}/s"
        else:
            pytest.fail("All concurrent searches failed")
    
    @patch('src.pipeline.core.config.get_config')
    def test_index_persistence_performance(self, mock_get_config, mock_config):
        """Test performance of index saving and loading"""
        mock_get_config.return_value = mock_config
        
        sizes = [1000, 5000, 10000]
        results = {}
        
        for size in sizes:
            embeddings = self.generate_test_embeddings(size)
            engine = FAISSSearchEngine()
            
            # Build index
            vector_index = engine.build_index(embeddings)
            index_id = vector_index.index_id
            
            # Measure save time
            start_time = time.time()
            engine.save_index(index_id)
            save_time = time.time() - start_time
            
            # Clear index
            engine.clear_index()
            
            # Measure load time
            start_time = time.time()
            engine.load_index(index_id)
            load_time = time.time() - start_time
            
            # Verify loaded index works
            query_vector = embeddings[0].vector
            search_results = engine.search_similar_by_vector(query_vector, top_k=5)
            
            results[size] = {
                'save_time': save_time,
                'load_time': load_time,
                'save_throughput': size / save_time if save_time > 0 else float('inf'),
                'load_throughput': size / load_time if load_time > 0 else float('inf'),
                'search_works': len(search_results) > 0
            }
            
            # Cleanup
            engine.delete_saved_index(index_id)
            engine.clear_index()
        
        # Print results
        print(f"\nIndex Persistence Performance:")
        print(f"{'Size':<8} {'Save Time':<12} {'Load Time':<12} {'Save Tput':<12} {'Load Tput':<12} {'Works':<8}")
        print("-" * 64)
        
        for size, result in results.items():
            print(f"{size:<8} {result['save_time']:<12.3f} {result['load_time']:<12.3f} "
                  f"{result['save_throughput']:<12.1f} {result['load_throughput']:<12.1f} {result['search_works']:<8}")
        
        # Performance assertions
        for size, result in results.items():
            assert result['save_time'] < 30, f"Save too slow for size {size}"
            assert result['load_time'] < 30, f"Load too slow for size {size}"
            assert result['search_works'], f"Loaded index doesn't work for size {size}"
    
    @patch('src.pipeline.core.config.get_config')
    def test_memory_usage_scaling(self, mock_get_config, mock_config):
        """Test memory usage scaling with index size"""
        mock_get_config.return_value = mock_config
        
        sizes = [500, 1000, 2000, 5000]
        results = {}
        
        for size in sizes:
            embeddings = self.generate_test_embeddings(size)
            engine = FAISSSearchEngine()
            
            # Build index and get info
            engine.build_index(embeddings)
            index_info = engine.get_index_info()
            
            # Calculate memory metrics
            theoretical_memory = size * 384 * 4  # 4 bytes per float
            reported_memory = index_info.get('memory_usage_bytes', 0)
            memory_efficiency = theoretical_memory / reported_memory if reported_memory > 0 else 0
            
            results[size] = {
                'theoretical_mb': theoretical_memory / (1024 * 1024),
                'reported_mb': reported_memory / (1024 * 1024),
                'memory_efficiency': memory_efficiency,
                'vectors_per_mb': size / (reported_memory / (1024 * 1024)) if reported_memory > 0 else 0
            }
            
            engine.clear_index()
        
        # Print results
        print(f"\nMemory Usage Scaling:")
        print(f"{'Size':<8} {'Theoretical MB':<15} {'Reported MB':<12} {'Efficiency':<12} {'Vectors/MB':<12}")
        print("-" * 60)
        
        for size, result in results.items():
            print(f"{size:<8} {result['theoretical_mb']:<15.1f} {result['reported_mb']:<12.1f} "
                  f"{result['memory_efficiency']:<12.2f} {result['vectors_per_mb']:<12.1f}")
        
        # Memory usage should scale linearly
        memory_values = [r['reported_mb'] for r in results.values()]
        size_values = list(sizes)
        
        # Check that memory usage increases with size
        for i in range(1, len(memory_values)):
            assert memory_values[i] > memory_values[i-1], "Memory usage should increase with size"
        
        # Memory efficiency should be reasonable (not too much overhead)
        for size, result in results.items():
            if result['memory_efficiency'] > 0:
                assert result['memory_efficiency'] > 0.5, f"Memory efficiency too low for size {size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])