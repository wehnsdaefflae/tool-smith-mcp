"""Shared test fixtures and utilities for Tool Smith MCP tests.

This module provides common fixtures and utilities for testing individual modules
without requiring the full MCP server to be running.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tool_smith_mcp.core.tool_manager import ToolManager
from tool_smith_mcp.models import ToolInfo, ToolType
from tool_smith_mcp.utils.cache import SimpleCache
from tool_smith_mcp.utils.claude_client import ClaudeClient
from tool_smith_mcp.utils.config import Config
from tool_smith_mcp.utils.docker_executor import DockerExecutor
from tool_smith_mcp.utils.vector_store import VectorStore


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_tools_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for test tools."""
    tools_dir = temp_dir / "tools"
    tools_dir.mkdir(exist_ok=True)
    return tools_dir


@pytest.fixture
def temp_initial_tools_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for initial tools."""
    initial_tools_dir = temp_dir / "initial_tools"
    initial_tools_dir.mkdir(exist_ok=True)
    return initial_tools_dir


@pytest.fixture
def temp_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for cache."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_vector_db_path(temp_dir: Path) -> Path:
    """Create a temporary directory for vector database."""
    return temp_dir / "vector_db"


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock configuration object."""
    config = Mock(spec=Config)
    config.claude = Mock()
    config.claude.model = "claude-3-5-sonnet-20241022"
    config.claude.max_tokens = 4000
    config.claude.temperature = 0.1
    config.claude.structure_args_temperature = 0.0
    config.claude.structure_args_max_tokens = 1000
    
    config.tools = Mock()
    config.tools.similarity_threshold = 0.7
    config.tools.search_top_k = 3
    config.tools.tools_dir = "./test-tools"
    config.tools.initial_tools_dir = "./test-initial-tools"
    
    config.vector_store = Mock()
    config.vector_store.db_path = "./test-vector-db"
    config.vector_store.collection_name = "test_tool_descriptions"
    config.vector_store.model_name = "all-MiniLM-L6-v2"
    
    config.docker = Mock()
    config.docker.enabled = True
    config.docker.force_local_for_debugging = False
    config.docker.image_name = "python:3.10-slim"
    config.docker.container_timeout = 30
    config.docker.memory_limit = "256m"
    config.docker.cpu_limit = 0.5
    
    config.cache = Mock()
    config.cache.enabled = True
    config.cache.cache_dir = "./test-cache"
    config.cache.embedding_ttl = 3600
    config.cache.tool_code_ttl = 1800
    config.cache.cleanup_interval = 3600
    
    return config


@pytest.fixture
def mock_claude_client() -> Mock:
    """Create a mock Claude client for testing."""
    mock_client = Mock(spec=ClaudeClient)
    mock_client.generate_tool = AsyncMock()
    mock_client.structure_arguments = AsyncMock()
    
    # Default return values
    mock_client.generate_tool.return_value = '''def test_generated_tool(value: str) -> str:
    """A test generated tool.
    
    Args:
        value: Input value
        
    Returns:
        Processed value
    """
    return value.upper()
'''
    mock_client.structure_arguments.return_value = {"value": "test"}
    
    return mock_client


@pytest.fixture
def mock_docker_executor() -> Mock:
    """Create a mock Docker executor for testing."""
    mock_executor = Mock(spec=DockerExecutor)
    mock_executor.execute_tool = AsyncMock()
    mock_executor.is_available = Mock(return_value=True)
    
    # Default return value
    mock_executor.execute_tool.return_value = "mocked_result"
    
    return mock_executor


@pytest.fixture
def vector_store(temp_vector_db_path: Path) -> VectorStore:
    """Create a VectorStore instance for testing."""
    # Use a real vector store but with temporary storage
    return VectorStore(
        db_path=temp_vector_db_path,
        collection_name="test_tools",
        model_name="all-MiniLM-L6-v2"
    )


@pytest.fixture
def mock_vector_store() -> Mock:
    """Create a mock VectorStore for testing."""
    mock_store = Mock(spec=VectorStore)
    mock_store.initialize = AsyncMock()
    mock_store.add_document = AsyncMock()
    mock_store.search = AsyncMock()
    mock_store.delete_document = AsyncMock()
    
    # Default search results
    mock_store.search.return_value = [
        ("test_tool", 0.8, "A test tool", {"type": "test"})
    ]
    
    return mock_store


@pytest.fixture
def simple_cache(temp_cache_dir: Path) -> SimpleCache:
    """Create a SimpleCache instance for testing."""
    return SimpleCache(
        cache_dir=temp_cache_dir,
        default_ttl=3600
    )


@pytest.fixture
def mock_cache() -> Mock:
    """Create a mock cache for testing."""
    mock_cache = Mock(spec=SimpleCache)
    mock_cache.get = Mock(return_value=None)  # Cache miss by default
    mock_cache.set = Mock()
    mock_cache.delete = Mock()
    mock_cache.clear = Mock()
    
    return mock_cache


@pytest.fixture
def tool_manager(
    temp_tools_dir: Path,
    temp_initial_tools_dir: Path,
    mock_vector_store: Mock,
    mock_claude_client: Mock,
    mock_docker_executor: Mock,
    mock_cache: Mock,
) -> ToolManager:
    """Create a ToolManager instance for testing with all dependencies mocked."""
    return ToolManager(
        tools_dir=temp_tools_dir,
        vector_store=mock_vector_store,
        claude_client=mock_claude_client,
        similarity_threshold=0.7,
        initial_tools_dir=temp_initial_tools_dir,
        docker_executor=mock_docker_executor,
        cache=mock_cache,
    )


@pytest.fixture
def real_tool_manager(
    temp_tools_dir: Path,
    temp_initial_tools_dir: Path,
    vector_store: VectorStore,
    mock_claude_client: Mock,
    simple_cache: SimpleCache,
) -> ToolManager:
    """Create a ToolManager with real vector store and cache for integration testing."""
    return ToolManager(
        tools_dir=temp_tools_dir,
        vector_store=vector_store,
        claude_client=mock_claude_client,
        similarity_threshold=0.7,
        initial_tools_dir=temp_initial_tools_dir,
        docker_executor=None,  # No Docker for integration tests
        cache=simple_cache,
    )


def create_test_tool_file(tools_dir: Path, tool_name: str, tool_code: str) -> Path:
    """Helper function to create a test tool file."""
    tool_file = tools_dir / f"{tool_name}.py"
    tool_file.write_text(tool_code)
    return tool_file


def create_sample_tool_code(
    function_name: str, 
    description: str = "A sample tool for testing",
    param_name: str = "value",
    param_type: str = "str",
    return_type: str = "str",
    implementation: str = "return value.upper()"
) -> str:
    """Helper function to generate sample tool code."""
    return f'''def {function_name}({param_name}: {param_type}) -> {return_type}:
    """{description}
    
    Args:
        {param_name}: Input {param_name}
        
    Returns:
        Processed {param_name}
    """
    {implementation}
'''


def create_sample_initial_tool(
    initial_tools_dir: Path,
    tool_name: str,
    function_name: str = None
) -> Path:
    """Helper function to create a sample initial tool."""
    if function_name is None:
        function_name = tool_name.replace("-", "_")
    
    tool_code = create_sample_tool_code(
        function_name,
        f"Sample initial tool: {tool_name}",
        implementation=f'return f"processed_{{value}}_by_{function_name}"'
    )
    
    return create_test_tool_file(initial_tools_dir, tool_name, tool_code)


class MockMCPContext:
    """Mock context for MCP operations without running a server."""
    
    def __init__(self):
        self.calls = []
        self.responses = {}
    
    def add_response(self, tool_name: str, response: Any) -> None:
        """Add a mock response for a tool call."""
        self.responses[tool_name] = response
    
    def get_response(self, tool_name: str) -> Any:
        """Get a mock response for a tool call."""
        return self.responses.get(tool_name, "mock_response")
    
    def record_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call for verification."""
        self.calls.append({"tool": tool_name, "arguments": arguments})


@pytest.fixture
def mock_mcp_context() -> MockMCPContext:
    """Create a mock MCP context for testing."""
    return MockMCPContext()


# Helper functions for creating test scenarios

def setup_claude_mock_responses(
    mock_claude_client: Mock,
    tool_generation_response: str = None,
    structure_args_response: Dict[str, Any] = None
) -> None:
    """Helper to set up Claude client mock responses."""
    if tool_generation_response:
        mock_claude_client.generate_tool.return_value = tool_generation_response
    
    if structure_args_response:
        mock_claude_client.structure_arguments.return_value = structure_args_response


def setup_vector_store_mock_responses(
    mock_vector_store: Mock,
    search_results: list = None
) -> None:
    """Helper to set up vector store mock responses."""
    if search_results:
        mock_vector_store.search.return_value = search_results


def setup_docker_mock_responses(
    mock_docker_executor: Mock,
    execution_result: Any = None,
    is_available: bool = True
) -> None:
    """Helper to set up Docker executor mock responses."""
    mock_docker_executor.is_available.return_value = is_available
    if execution_result is not None:
        mock_docker_executor.execute_tool.return_value = execution_result


# Async test helpers

async def run_tool_manager_test(
    tool_manager: ToolManager,
    task_description: str,
    arguments: Dict[str, Any] = None,
    expected_outcome: str = None
) -> Any:
    """Helper to run a tool manager test with proper setup."""
    if arguments is None:
        arguments = {}
    
    await tool_manager.initialize()
    
    return await tool_manager.solve_task(
        task_description=task_description,
        arguments=arguments,
        expected_outcome=expected_outcome
    )


# Test data generators

def generate_tool_infos(count: int = 3) -> list[ToolInfo]:
    """Generate a list of test ToolInfo objects."""
    return [
        ToolInfo(
            name=f"test_tool_{i}",
            signature=f"(value: str) -> str",
            docstring=f"Test tool {i} description",
            type=ToolType.INITIAL if i % 2 == 0 else ToolType.GENERATED,
            file=f"test_tool_{i}.py"
        )
        for i in range(count)
    ]


def generate_mock_task_scenarios() -> list[Dict[str, Any]]:
    """Generate common task scenarios for testing."""
    return [
        {
            "description": "Calculate mathematical expression",
            "arguments": {"expression": "2 + 3"},
            "expected_tool": "calculate_math",
            "expected_result": 5.0
        },
        {
            "description": "Format text to uppercase",
            "arguments": {"text": "hello world"},
            "expected_tool": "format_text",
            "expected_result": "HELLO WORLD"
        },
        {
            "description": "Get current timestamp",
            "arguments": {},
            "expected_tool": "datetime_utils",
            "expected_result": "2023-01-01T00:00:00Z"
        }
    ]