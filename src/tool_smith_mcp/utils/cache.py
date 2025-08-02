"""Simple caching utilities for Tool Smith MCP."""

import contextlib
import hashlib
import json
import logging
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SimpleCache:
    """Simple file-based cache with TTL support."""

    def __init__(self, cache_dir: Path, default_ttl: int = 3600) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized cache at {cache_dir}")

    def _get_cache_key(self, key: str) -> str:
        """Generate a safe cache key from input."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Check expiration
            if time.time() > cache_data["expires_at"]:
                cache_file.unlink()  # Remove expired cache
                return None

            logger.debug(f"Cache hit for key: {key[:50]}...")
            return cache_data["value"]

        except Exception as e:
            logger.warning(f"Error reading cache file {cache_file}: {e}")
            # Remove corrupted cache file
            with contextlib.suppress(Exception):
                cache_file.unlink()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_file = self._get_cache_file(key)

        cache_data = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            logger.debug(f"Cached value for key: {key[:50]}...")

        except Exception as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")

    def delete(self, key: str) -> bool:
        """Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_file = self._get_cache_file(key)

        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.debug(f"Deleted cache for key: {key[:50]}...")
                return True
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_file}: {e}")

        return False

    def clear(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("Cleared all cache files")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired cache files.

        Returns:
            Number of files removed
        """
        removed_count = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                if current_time > cache_data["expires_at"]:
                    cache_file.unlink()
                    removed_count += 1

            except Exception:
                # Remove corrupted files
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception:
                    pass

        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} expired cache files")

        return removed_count


def cache_result(
    cache: SimpleCache,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None,
) -> Callable[[F], F]:
    """Decorator to cache function results.

    Args:
        cache: Cache instance to use
        key_func: Function to generate cache key from args/kwargs
        ttl: Time-to-live for cached results

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                if args:
                    key_parts.extend(str(arg) for arg in args)
                if kwargs:
                    key_parts.append(json.dumps(kwargs, sort_keys=True))
                cache_key = "|".join(key_parts)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        # Add cache management methods to the wrapper
        wrapper._cache = cache  # type: ignore
        wrapper._clear_cache = lambda: cache.clear()  # type: ignore

        return wrapper  # type: ignore

    return decorator


def embedding_cache_key(query: str) -> str:
    """Generate cache key for embeddings."""
    return f"embedding|{hashlib.md5(query.encode()).hexdigest()}"


def tool_code_cache_key(task_description: str, arguments: Dict[str, Any]) -> str:
    """Generate cache key for generated tool code."""
    key_data = {"task": task_description, "args": arguments}
    key_json = json.dumps(key_data, sort_keys=True)
    return f"tool_code|{hashlib.md5(key_json.encode()).hexdigest()}"
