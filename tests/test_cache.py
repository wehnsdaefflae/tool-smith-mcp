"""Tests for caching functionality."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from tool_smith_mcp.utils.cache import (
    SimpleCache,
    cache_result,
    embedding_cache_key,
    tool_code_cache_key,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_simple_cache_basic_operations(temp_cache_dir):
    """Test basic cache operations."""
    cache = SimpleCache(temp_cache_dir, default_ttl=60)

    # Test set and get
    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value"

    # Test non-existent key
    assert cache.get("non_existent") is None

    # Test different data types
    cache.set("dict_key", {"nested": {"data": [1, 2, 3]}})
    cached_dict = cache.get("dict_key")
    assert cached_dict == {"nested": {"data": [1, 2, 3]}}

    cache.set("list_key", [1, "two", {"three": 3}])
    cached_list = cache.get("list_key")
    assert cached_list == [1, "two", {"three": 3}]


def test_cache_expiration(temp_cache_dir):
    """Test cache TTL and expiration."""
    cache = SimpleCache(temp_cache_dir, default_ttl=1)  # 1 second TTL

    # Set a value
    cache.set("expiring_key", "expiring_value")

    # Should be available immediately
    assert cache.get("expiring_key") == "expiring_value"

    # Wait for expiration
    time.sleep(1.1)

    # Should be None after expiration
    assert cache.get("expiring_key") is None


def test_cache_custom_ttl(temp_cache_dir):
    """Test cache with custom TTL."""
    cache = SimpleCache(temp_cache_dir, default_ttl=60)

    # Set with custom short TTL
    cache.set("short_ttl_key", "short_value", ttl=1)

    # Set with default TTL
    cache.set("long_ttl_key", "long_value")

    # Both should be available immediately
    assert cache.get("short_ttl_key") == "short_value"
    assert cache.get("long_ttl_key") == "long_value"

    # Wait for short TTL to expire
    time.sleep(1.1)

    # Short TTL should be expired, long TTL should still be available
    assert cache.get("short_ttl_key") is None
    assert cache.get("long_ttl_key") == "long_value"


def test_cache_delete(temp_cache_dir):
    """Test cache deletion."""
    cache = SimpleCache(temp_cache_dir)

    # Set and verify
    cache.set("delete_me", "delete_value")
    assert cache.get("delete_me") == "delete_value"

    # Delete and verify
    result = cache.delete("delete_me")
    assert result is True
    assert cache.get("delete_me") is None

    # Delete non-existent key
    result = cache.delete("non_existent")
    assert result is False


def test_cache_clear(temp_cache_dir):
    """Test cache clearing."""
    cache = SimpleCache(temp_cache_dir)

    # Set multiple values
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Verify all are set
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

    # Clear cache
    cache.clear()

    # Verify all are gone
    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") is None


def test_cache_cleanup_expired(temp_cache_dir):
    """Test cleanup of expired cache entries."""
    cache = SimpleCache(temp_cache_dir, default_ttl=60)

    # Set some values with different TTLs
    cache.set("keep_me", "keep_value", ttl=60)  # Long TTL
    cache.set("expire_me1", "expire_value1", ttl=1)  # Short TTL
    cache.set("expire_me2", "expire_value2", ttl=1)  # Short TTL

    # Wait for short TTLs to expire
    time.sleep(1.1)

    # Run cleanup
    removed_count = cache.cleanup_expired()

    # Should have removed the expired ones
    assert removed_count >= 2
    assert cache.get("keep_me") == "keep_value"
    assert cache.get("expire_me1") is None
    assert cache.get("expire_me2") is None


def test_cache_corrupted_file_handling(temp_cache_dir):
    """Test handling of corrupted cache files."""
    cache = SimpleCache(temp_cache_dir)

    # Create a corrupted cache file manually
    corrupted_file = temp_cache_dir / "corrupted.cache"
    corrupted_file.write_text("This is not valid pickle data")

    # Should handle corrupted file gracefully
    assert cache.get("corrupted") is None

    # File should be cleaned up
    assert not corrupted_file.exists() or len(corrupted_file.read_text()) == 0


def test_cache_result_decorator(temp_cache_dir):
    """Test the cache_result decorator."""
    cache = SimpleCache(temp_cache_dir)

    call_count = 0

    @cache_result(cache, ttl=60)
    def expensive_function(x: int, y: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"{y}_{x}"

    # First call
    result1 = expensive_function(42, "test")
    assert result1 == "test_42"
    assert call_count == 1

    # Second call with same arguments - should use cache
    result2 = expensive_function(42, "test")
    assert result2 == "test_42"
    assert call_count == 1  # Should not have incremented

    # Call with different arguments
    result3 = expensive_function(99, "other")
    assert result3 == "other_99"
    assert call_count == 2  # Should have incremented


def test_cache_result_decorator_custom_key_func(temp_cache_dir):
    """Test cache_result decorator with custom key function."""
    cache = SimpleCache(temp_cache_dir)

    call_count = 0

    def custom_key_func(x: int, y: str) -> str:
        return f"custom_{x}_{y}"

    @cache_result(cache, key_func=custom_key_func, ttl=60)
    def function_with_custom_key(x: int, y: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result_{x}_{y}"

    # Call function
    result = function_with_custom_key(1, "a")
    assert result == "result_1_a"
    assert call_count == 1

    # Check that custom key was used in cache
    cached_value = cache.get("custom_1_a")
    assert cached_value == "result_1_a"


def test_embedding_cache_key():
    """Test embedding cache key generation."""
    query1 = "Calculate the area of a circle"
    query2 = "Calculate the area of a circle"
    query3 = "Calculate the area of a square"

    key1 = embedding_cache_key(query1)
    key2 = embedding_cache_key(query2)
    key3 = embedding_cache_key(query3)

    # Same queries should generate same keys
    assert key1 == key2

    # Different queries should generate different keys
    assert key1 != key3

    # Keys should have the expected prefix
    assert key1.startswith("embedding|")


def test_tool_code_cache_key():
    """Test tool code cache key generation."""
    task1 = "Calculate something"
    args1 = {"x": 1, "y": 2}

    task2 = "Calculate something"
    args2 = {"x": 1, "y": 2}

    task3 = "Calculate something else"
    args3 = {"x": 1, "y": 2}

    key1 = tool_code_cache_key(task1, args1)
    key2 = tool_code_cache_key(task2, args2)
    key3 = tool_code_cache_key(task3, args3)

    # Same task and args should generate same keys
    assert key1 == key2

    # Different task should generate different key
    assert key1 != key3

    # Keys should have the expected prefix
    assert key1.startswith("tool_code|")


def test_cache_thread_safety(temp_cache_dir):
    """Test cache operations under concurrent access."""
    import threading

    cache = SimpleCache(temp_cache_dir)
    results = []

    def cache_worker(worker_id: int):
        # Each worker sets and gets its own values
        for i in range(10):
            key = f"worker_{worker_id}_key_{i}"
            value = f"worker_{worker_id}_value_{i}"

            cache.set(key, value)
            retrieved = cache.get(key)
            results.append((key, value, retrieved))

    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=cache_worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify all operations succeeded
    assert len(results) == 50  # 5 workers * 10 operations each

    for key, expected_value, retrieved_value in results:
        assert retrieved_value == expected_value


def test_cache_large_data(temp_cache_dir):
    """Test cache with large data objects."""
    cache = SimpleCache(temp_cache_dir)

    # Create a large data structure
    large_data = {
        "embeddings": [[float(i + j) for j in range(100)] for i in range(100)],
        "metadata": {"description": "Large embedding matrix" * 1000},
        "nested": {"deep": {"structure": [{"item": i} for i in range(1000)]}},
    }

    # Should be able to cache and retrieve large data
    cache.set("large_data", large_data)
    retrieved_data = cache.get("large_data")

    assert retrieved_data == large_data
    assert len(retrieved_data["embeddings"]) == 100
    assert len(retrieved_data["nested"]["deep"]["structure"]) == 1000


def test_cache_persistence_across_instances(temp_cache_dir):
    """Test that cache persists across different cache instances."""
    # First cache instance
    cache1 = SimpleCache(temp_cache_dir)
    cache1.set("persistent_key", "persistent_value")

    # Second cache instance using same directory
    cache2 = SimpleCache(temp_cache_dir)
    retrieved_value = cache2.get("persistent_key")

    assert retrieved_value == "persistent_value"


@pytest.mark.asyncio
async def test_cache_in_async_context(temp_cache_dir):
    """Test cache operations in async context."""
    cache = SimpleCache(temp_cache_dir)

    async def async_cache_worker(worker_id: int):
        # Simulate async work
        await asyncio.sleep(0.01)

        cache.set(f"async_key_{worker_id}", f"async_value_{worker_id}")

        await asyncio.sleep(0.01)

        return cache.get(f"async_key_{worker_id}")

    # Run multiple async workers
    tasks = [async_cache_worker(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify all results
    for i, result in enumerate(results):
        assert result == f"async_value_{i}"
