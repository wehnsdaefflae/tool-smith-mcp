"""Simplified tests for second-layer tools."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tool_smith_mcp.core.tool_manager import ToolManager
from tool_smith_mcp.utils.cache import SimpleCache
from tool_smith_mcp.utils.claude_client import ClaudeClient
from tool_smith_mcp.utils.docker_executor import DockerExecutor
from tool_smith_mcp.utils.vector_store import VectorStore


class TestToolManagerBasics:
    """Basic tests for ToolManager functionality."""

    @pytest.fixture
    def tool_manager_sync(self):
        """Create a simple tool manager for sync testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            tools_dir.mkdir()

            # Mock dependencies
            vector_store = AsyncMock(spec=VectorStore)
            claude_client = AsyncMock(spec=ClaudeClient)
            cache = AsyncMock(spec=SimpleCache)

            # Use real initial tools directory
            initial_tools_dir = (
                Path(__file__).parent.parent / "resources" / "initial_tools"
            )

            manager = ToolManager(
                tools_dir=tools_dir,
                vector_store=vector_store,
                claude_client=claude_client,
                initial_tools_dir=initial_tools_dir,
                cache=cache,
            )

            yield manager

    def test_tool_manager_creation(self, tool_manager_sync):
        """Test that ToolManager can be created."""
        assert tool_manager_sync is not None
        assert tool_manager_sync.tools_dir.exists()
        assert tool_manager_sync.similarity_threshold == 0.7

    def test_list_tools_empty(self, tool_manager_sync):
        """Test list_tools when no tools are loaded."""
        tools = tool_manager_sync.list_tools()
        assert isinstance(tools, dict)
        assert len(tools) == 0  # No tools loaded yet

    @pytest.mark.asyncio
    async def test_tool_manager_initialization(self, tool_manager_sync):
        """Test that ToolManager can be initialized."""
        await tool_manager_sync.initialize()

        # Should have loaded some initial tools
        tools = tool_manager_sync.list_tools()
        assert len(tools) > 0

        # Check that calculate_math tool is loaded
        assert "calculate_math" in tools
        tool_info = tools["calculate_math"]
        assert tool_info["type"] == "initial"
        assert "expression" in tool_info["signature"]

    @pytest.mark.asyncio
    async def test_execute_calculate_math_tool(self, tool_manager_sync):
        """Test executing the calculate_math tool."""
        await tool_manager_sync.initialize()

        result = await tool_manager_sync.execute_tool_directly(
            tool_name="calculate_math",
            arguments={"expression": "2 + 3"},
            force_local=True,
        )

        assert result == 5.0

    @pytest.mark.asyncio
    async def test_execute_calculate_math_complex(self, tool_manager_sync):
        """Test executing calculate_math with a complex expression."""
        await tool_manager_sync.initialize()

        result = await tool_manager_sync.execute_tool_directly(
            tool_name="calculate_math",
            arguments={"expression": "(10 + 5) * 2"},
            force_local=True,
        )

        assert result == 30.0

    @pytest.mark.asyncio
    async def test_tool_not_found_error(self, tool_manager_sync):
        """Test that executing non-existent tool raises KeyError."""
        await tool_manager_sync.initialize()

        with pytest.raises(KeyError, match="Tool 'nonexistent' not found"):
            await tool_manager_sync.execute_tool_directly(
                tool_name="nonexistent",
                arguments={},
                force_local=True,
            )


class TestDockerExecutorBasics:
    """Basic tests for DockerExecutor."""

    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client for testing."""
        with patch('docker.from_env') as mock_docker:
            client = mock_docker.return_value
            client.ping.return_value = True
            client.images.get.return_value = True  # Image exists
            yield client

    def test_docker_executor_creation(self, mock_docker_client):
        """Test that DockerExecutor can be created."""
        executor = DockerExecutor()
        assert executor.image_name == "python:3.11-slim"
        assert executor.container_timeout == 30
        assert executor.memory_limit == "256m"
        assert executor.cpu_limit == 0.5

    def test_docker_executor_custom_config(self, mock_docker_client):
        """Test DockerExecutor with custom configuration."""
        executor = DockerExecutor(
            image_name="python:3.12",
            container_timeout=60,
            memory_limit="512m",
            cpu_limit=1.0,
        )

        assert executor.image_name == "python:3.12"
        assert executor.container_timeout == 60
        assert executor.memory_limit == "512m"
        assert executor.cpu_limit == 1.0

    def test_execution_script_creation(self, mock_docker_client):
        """Test execution script generation."""
        executor = DockerExecutor()

        tool_code = '''
def test_func(x: int) -> int:
    return x * 2
'''

        script = executor._create_execution_script(
            tool_code=tool_code,
            function_name="test_func",
            arguments={"x": 5},
        )

        # Check that script contains expected components
        assert "exec(\"\"\"" in script
        assert "def test_func" in script
        assert "test_func" in script
        assert '{"x": 5}' in script
        assert "import json" in script

    def test_parse_successful_output(self, mock_docker_client):
        """Test parsing successful container output."""
        executor = DockerExecutor()

        output = '{"success": true, "result": "test_result", "error": null}'
        result = executor._parse_container_output(output)

        assert result["success"] is True
        assert result["result"] == "test_result"
        assert result["error"] is None

    def test_parse_error_output(self, mock_docker_client):
        """Test parsing error container output."""
        executor = DockerExecutor()

        output = '{"success": false, "result": null, "error": {"type": "RuntimeError", "message": "Something went wrong"}}'
        result = executor._parse_container_output(output)

        assert result["success"] is False
        assert result["result"] is None
        assert result["error"]["type"] == "RuntimeError"
        assert result["error"]["message"] == "Something went wrong"

    def test_handle_successful_result(self, mock_docker_client):
        """Test handling successful execution result."""
        executor = DockerExecutor()

        result_data = {
            "success": True,
            "result": "expected_result",
            "error": None
        }

        result = executor._handle_execution_result(result_data)
        assert result == "expected_result"

    def test_handle_error_result(self, mock_docker_client):
        """Test handling error execution result."""
        executor = DockerExecutor()

        result_data = {
            "success": False,
            "result": None,
            "error": {
                "type": "ValueError",
                "message": "Test error"
            }
        }

        with pytest.raises(ValueError, match="Test error"):
            executor._handle_execution_result(result_data)


class TestOptionalContainerization:
    """Test the optional containerization feature."""

    @pytest.fixture
    def tool_manager_with_docker(self):
        """Create a tool manager with Docker support."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            tools_dir.mkdir()

            # Mock dependencies
            vector_store = AsyncMock(spec=VectorStore)
            claude_client = AsyncMock(spec=ClaudeClient)
            cache = AsyncMock(spec=SimpleCache)
            docker_executor = AsyncMock(spec=DockerExecutor)

            manager = ToolManager(
                tools_dir=tools_dir,
                vector_store=vector_store,
                claude_client=claude_client,
                docker_executor=docker_executor,
                cache=cache,
            )

            yield manager

    @pytest.mark.asyncio
    async def test_force_local_execution_parameter(self, tool_manager_with_docker):
        """Test that force_local parameter works."""
        # Setup a mock tool
        tool_manager_with_docker.loaded_tools = {
            "test_tool": lambda x: x * 2
        }
        tool_manager_with_docker.tool_metadata = {
            "test_tool": {"type": "generated"}
        }

        # Execute with force_local=True
        result = await tool_manager_with_docker._execute_tool_safely(
            tool_func=tool_manager_with_docker.loaded_tools["test_tool"],
            arguments={"x": 5},
            task_description="test",
            force_local=True,
        )

        assert result == 10
        # Docker should not have been called
        tool_manager_with_docker.docker_executor.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_docker_execution_for_generated_tools(self, tool_manager_with_docker):
        """Test that generated tools use Docker by default."""
        # Setup a mock tool
        def mock_tool(x):
            return x * 3

        tool_manager_with_docker.loaded_tools = {"test_tool": mock_tool}
        tool_manager_with_docker.tool_metadata = {
            "test_tool": {"type": "generated", "code": "def test_tool(x): return x * 3"}
        }

        # Mock Docker execution result
        tool_manager_with_docker.docker_executor.execute_tool.return_value = 15

        # Execute without force_local
        result = await tool_manager_with_docker._execute_tool_safely(
            tool_func=mock_tool,
            arguments={"x": 5},
            task_description="test",
            force_local=False,
        )

        assert result == 15
        # Docker should have been called
        tool_manager_with_docker.docker_executor.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_execution_for_initial_tools(self, tool_manager_with_docker):
        """Test that initial tools always execute locally."""
        # Setup a mock initial tool
        def mock_tool(x):
            return x + 10

        tool_manager_with_docker.loaded_tools = {"test_tool": mock_tool}
        tool_manager_with_docker.tool_metadata = {
            "test_tool": {"type": "initial"}
        }

        # Execute - should use local even with Docker available
        result = await tool_manager_with_docker._execute_tool_safely(
            tool_func=mock_tool,
            arguments={"x": 5},
            task_description="test",
            force_local=False,
        )

        assert result == 15
        # Docker should NOT have been called for initial tools
        tool_manager_with_docker.docker_executor.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_docker_fallback_on_failure(self, tool_manager_with_docker):
        """Test fallback to local execution when Docker fails."""
        # Setup a mock tool
        def mock_tool(x):
            return x * 4

        tool_manager_with_docker.loaded_tools = {"test_tool": mock_tool}
        tool_manager_with_docker.tool_metadata = {
            "test_tool": {"type": "generated", "code": "def test_tool(x): return x * 4"}
        }

        # Mock Docker execution to fail
        tool_manager_with_docker.docker_executor.execute_tool.side_effect = Exception("Docker failed")

        # Execute - should fall back to local
        result = await tool_manager_with_docker._execute_tool_safely(
            tool_func=mock_tool,
            arguments={"x": 6},
            task_description="test",
            force_local=False,
        )

        assert result == 24  # Local execution result
        # Docker should have been attempted
        tool_manager_with_docker.docker_executor.execute_tool.assert_called_once()
