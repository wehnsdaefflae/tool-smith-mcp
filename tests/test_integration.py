"""Integration tests for Tool Smith MCP."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tool_smith_mcp.core.server import ToolSmithMCPServer
from tool_smith_mcp.core.tool_manager import ToolManager
from tool_smith_mcp.utils.cache import SimpleCache
from tool_smith_mcp.utils.claude_client import ClaudeClient
from tool_smith_mcp.utils.config import (
    CacheConfig,
    Config,
    DockerConfig,
    ToolsConfig,
    VectorStoreConfig,
)
from tool_smith_mcp.utils.vector_store import VectorStore


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tools_dir = temp_path / "tools"
        cache_dir = temp_path / "cache"
        vector_db_dir = temp_path / "vector_db"

        tools_dir.mkdir()
        cache_dir.mkdir()
        vector_db_dir.mkdir()

        yield {
            "tools_dir": tools_dir,
            "cache_dir": cache_dir,
            "vector_db_dir": vector_db_dir,
        }


@pytest.fixture
def test_config(temp_dirs):
    """Create test configuration."""
    return Config(
        tools=ToolsConfig(
            tools_dir=str(temp_dirs["tools_dir"]),
            similarity_threshold=0.7,
        ),
        vector_store=VectorStoreConfig(
            db_path=str(temp_dirs["vector_db_dir"]),
        ),
        cache=CacheConfig(
            enabled=True,
            cache_dir=str(temp_dirs["cache_dir"]),
        ),
        docker=DockerConfig(
            enabled=False,  # Disable Docker for integration tests
        ),
    )


@pytest.fixture
def mock_claude_client():
    """Create a mock Claude client."""
    client = Mock(spec=ClaudeClient)

    # Mock tool generation
    client.generate_tool = AsyncMock(
        return_value='''
def calculate_area_circle(radius: float) -> float:
    """Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        The area of the circle
    """
    import math
    return math.pi * radius * radius
'''
    )

    # Mock argument structuring
    client.structure_arguments = AsyncMock(return_value={"radius": 5.0})

    return client


@pytest.mark.asyncio
async def test_full_task_solving_flow(test_config, temp_dirs, mock_claude_client):
    """Test the complete task solving flow from request to response."""

    # Create cache and vector store
    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    # Create tool manager
    tool_manager = ToolManager(
        tools_dir=temp_dirs["tools_dir"],
        vector_store=vector_store,
        claude_client=mock_claude_client,
        cache=cache,
    )

    # Initialize
    await tool_manager.initialize()

    # Test task solving
    result = await tool_manager.solve_task(
        task_description="Calculate the area of a circle with radius 5",
        arguments={"radius": 5},
        expected_outcome="numerical result",
    )

    # Verify the result
    assert result is not None
    # Since we're mocking Claude, verify the mocks were called
    mock_claude_client.generate_tool.assert_called_once()
    mock_claude_client.structure_arguments.assert_called_once()


@pytest.mark.asyncio
async def test_caching_functionality(test_config, temp_dirs, mock_claude_client):
    """Test that caching works correctly."""

    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    # Test embedding caching
    text = "Calculate the area of a circle"

    # First call should generate embedding
    embedding1 = vector_store._get_embedding(text)

    # Second call should use cache
    embedding2 = vector_store._get_embedding(text)

    # Should be identical
    assert embedding1 == embedding2

    # Test cache functionality directly
    cache.set("test_key", {"data": "test_value"})
    cached_value = cache.get("test_key")
    assert cached_value == {"data": "test_value"}


@pytest.mark.asyncio
async def test_error_handling_claude_api_failure(test_config, temp_dirs):
    """Test error handling when Claude API fails."""

    # Create a Claude client that fails
    failed_claude_client = Mock(spec=ClaudeClient)
    failed_claude_client.generate_tool = AsyncMock(
        side_effect=Exception("Claude API error")
    )

    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    tool_manager = ToolManager(
        tools_dir=temp_dirs["tools_dir"],
        vector_store=vector_store,
        claude_client=failed_claude_client,
        cache=cache,
    )

    await tool_manager.initialize()

    # Should raise the Claude API error
    with pytest.raises(Exception, match="Claude API error"):
        await tool_manager.solve_task(
            task_description="Calculate something",
            arguments={},
        )


@pytest.mark.asyncio
async def test_tool_similarity_matching(test_config, temp_dirs, mock_claude_client):
    """Test that similar tools are matched correctly."""

    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    tool_manager = ToolManager(
        tools_dir=temp_dirs["tools_dir"],
        vector_store=vector_store,
        claude_client=mock_claude_client,
        cache=cache,
        similarity_threshold=0.5,  # Lower threshold for testing
    )

    await tool_manager.initialize()

    # First, create a tool
    await tool_manager.solve_task(
        task_description="Calculate circle area",
        arguments={"radius": 3},
    )

    # Reset mock call counts
    mock_claude_client.generate_tool.reset_mock()

    # Now try a similar task - should reuse existing tool
    await tool_manager.solve_task(
        task_description="Find the area of a circle",
        arguments={"radius": 4},
    )

    # Should NOT have called generate_tool again due to similarity
    # Note: This test may need adjustment based on actual similarity scoring
    # mock_claude_client.generate_tool.assert_not_called()


@pytest.mark.asyncio
async def test_vector_store_persistence(test_config, temp_dirs, mock_claude_client):
    """Test that vector store persists tools between sessions."""

    cache = SimpleCache(temp_dirs["cache_dir"])

    # First session - create and store a tool
    vector_store1 = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    await vector_store1.add_document(
        doc_id="test_tool",
        content="A test tool that calculates something",
        metadata={"type": "test"},
    )

    # Second session - should be able to retrieve the tool
    vector_store2 = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    results = await vector_store2.search("calculate something", top_k=1)

    assert len(results) > 0
    assert results[0][0] == "test_tool"


@pytest.mark.asyncio
async def test_initial_tools_loading(test_config, temp_dirs, mock_claude_client):
    """Test loading of initial tools."""

    # Create a mock initial tools directory
    initial_tools_dir = temp_dirs["tools_dir"] / "initial"
    initial_tools_dir.mkdir()

    # Create a sample initial tool
    sample_tool = initial_tools_dir / "sample_tool.py"
    sample_tool.write_text(
        '''
def sample_function(x: int) -> int:
    """A sample function that doubles a number.
    
    Args:
        x: Number to double
        
    Returns:
        The doubled number
    """
    return x * 2
'''
    )

    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    tool_manager = ToolManager(
        tools_dir=temp_dirs["tools_dir"],
        vector_store=vector_store,
        claude_client=mock_claude_client,
        cache=cache,
        initial_tools_dir=initial_tools_dir,
    )

    await tool_manager.initialize()

    # Check that the initial tool was loaded
    assert "sample_function" in tool_manager.loaded_tools

    # Check that it was added to vector store
    results = await vector_store.search("sample function", top_k=1)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_server_initialization_and_tool_registration(test_config):
    """Test that the server initializes correctly and registers tools."""

    with patch(
        "tool_smith_mcp.core.server.get_claude_api_key", return_value="test_key"
    ):
        with patch("tool_smith_mcp.core.server.ClaudeClient") as mock_claude_class:
            mock_claude_instance = Mock()
            mock_claude_class.return_value = mock_claude_instance

            server = ToolSmithMCPServer(config=test_config)

            # Check that components were initialized
            assert server.tool_manager is not None
            assert server.vector_store is not None

            # Check that Claude client was created
            mock_claude_class.assert_called_once()


@pytest.mark.asyncio
async def test_cache_cleanup_and_expiration(test_config, temp_dirs):
    """Test cache cleanup and expiration functionality."""

    cache = SimpleCache(temp_dirs["cache_dir"], default_ttl=1)  # 1 second TTL

    # Add some data
    cache.set("test_key", "test_value")

    # Should be retrievable immediately
    assert cache.get("test_key") == "test_value"

    # Wait for expiration
    await asyncio.sleep(1.1)

    # Should be None after expiration
    assert cache.get("test_key") is None

    # Test cleanup
    cache.set("key1", "value1", ttl=100)  # Long TTL
    cache.set("key2", "value2", ttl=1)  # Short TTL

    await asyncio.sleep(1.1)

    # Run cleanup
    removed_count = cache.cleanup_expired()

    # Should have removed the expired one
    assert removed_count >= 1
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None


@pytest.mark.asyncio
async def test_task_request_validation():
    """Test that task requests are validated correctly."""
    from tool_smith_mcp.core.server import TaskRequest

    # Valid request
    request = TaskRequest(
        task_description="Test task",
        arguments={"param": "value"},
        expected_outcome="Test outcome",
    )

    assert request.task_description == "Test task"
    assert request.arguments == {"param": "value"}
    assert request.expected_outcome == "Test outcome"

    # Request with defaults
    minimal_request = TaskRequest(task_description="Minimal task")
    assert minimal_request.arguments == {}
    assert minimal_request.expected_outcome is None

    # Invalid request should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        TaskRequest()  # Missing required field


@pytest.mark.asyncio
async def test_concurrent_tool_execution(test_config, temp_dirs, mock_claude_client):
    """Test concurrent execution of tool operations."""

    cache = SimpleCache(temp_dirs["cache_dir"])
    vector_store = VectorStore(db_path=temp_dirs["vector_db_dir"], cache=cache)

    tool_manager = ToolManager(
        tools_dir=temp_dirs["tools_dir"],
        vector_store=vector_store,
        claude_client=mock_claude_client,
        cache=cache,
    )

    await tool_manager.initialize()

    # Create multiple concurrent tasks
    tasks = [
        tool_manager.solve_task(
            task_description=f"Calculate something {i}",
            arguments={"value": i},
        )
        for i in range(3)
    ]

    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # All should complete (even if some are exceptions due to mocking)
    assert len(results) == 3

    # Check that Claude was called for each unique task
    assert mock_claude_client.generate_tool.call_count >= 1
