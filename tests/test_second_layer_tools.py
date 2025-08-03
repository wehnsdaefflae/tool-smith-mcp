"""Tests for second-layer tools (initial and generated tools)."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tool_smith_mcp.core.tool_manager import ToolManager
from src.tool_smith_mcp.utils.cache import SimpleCache
from src.tool_smith_mcp.utils.claude_client import ClaudeClient
from src.tool_smith_mcp.utils.docker_executor import DockerExecutor
from src.tool_smith_mcp.utils.vector_store import VectorStore


class TestInitialTools:
    """Test suite for initial/built-in tools."""

    @pytest.fixture
    async def tool_manager(self):
        """Create a tool manager for testing."""
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

            await manager.initialize()
            yield manager

    async def test_load_initial_tools(self, tool_manager):
        """Test that initial tools are loaded correctly."""
        tools = tool_manager.list_tools()

        # Should have loaded several initial tools
        assert len(tools) > 0

        # Check specific tools exist
        expected_tools = [
            "calculate_math",
            "format_text",
            "file_operations",
            "datetime_utils",
            "encoding_utils",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Expected tool {tool_name} not found"
            assert tools[tool_name]["type"] == "initial"

    async def test_calculate_math_tool(self, tool_manager):
        """Test the calculate_math initial tool."""
        result = await tool_manager.execute_tool_directly(
            tool_name="calculate_math",
            arguments={"expression": "2 + 3 * 4"},
            force_local=True,
        )

        assert result == 14.0

    async def test_calculate_math_tool_complex(self, tool_manager):
        """Test calculate_math with complex expression."""
        result = await tool_manager.execute_tool_directly(
            tool_name="calculate_math",
            arguments={"expression": "(10 + 5) / 3"},
            force_local=True,
        )

        assert result == 5.0

    async def test_calculate_math_tool_error(self, tool_manager):
        """Test calculate_math with invalid expression."""
        with pytest.raises(Exception):
            await tool_manager.execute_tool_directly(
                tool_name="calculate_math",
                arguments={"expression": "invalid expression"},
                force_local=True,
            )

    async def test_initial_tools_execute_locally(self, tool_manager):
        """Test that initial tools always execute locally."""
        with patch.object(
            tool_manager,
            "_execute_tool_safely",
            wraps=tool_manager._execute_tool_safely,
        ) as mock_execute:
            await tool_manager.execute_tool_directly(
                tool_name="calculate_math",
                arguments={"expression": "1 + 1"},
                force_local=False,  # Even without forcing local
            )

            # Should have been called with the tool function
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[1]["force_local"] is False


class TestGeneratedTools:
    """Test suite for generated/runtime tools."""

    @pytest.fixture
    async def tool_manager_with_docker(self):
        """Create a tool manager with Docker support for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            tools_dir.mkdir()

            # Mock dependencies
            vector_store = AsyncMock(spec=VectorStore)
            claude_client = AsyncMock(spec=ClaudeClient)
            cache = AsyncMock(spec=SimpleCache)
            docker_executor = AsyncMock(spec=DockerExecutor)

            # Configure Claude client mock to return a simple tool
            claude_client.generate_tool.return_value = '''
def simple_add(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b
'''

            manager = ToolManager(
                tools_dir=tools_dir,
                vector_store=vector_store,
                claude_client=claude_client,
                docker_executor=docker_executor,
                cache=cache,
            )

            await manager.initialize()
            yield manager

    @pytest.fixture
    async def tool_manager_no_docker(self):
        """Create a tool manager without Docker for testing local execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            tools_dir.mkdir()

            # Mock dependencies
            vector_store = AsyncMock(spec=VectorStore)
            claude_client = AsyncMock(spec=ClaudeClient)
            cache = AsyncMock(spec=SimpleCache)

            # Configure Claude client mock
            claude_client.generate_tool.return_value = '''
def simple_multiply(x: float, y: float) -> float:
    """Multiply two numbers.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Product of x and y
    """
    return x * y
'''

            manager = ToolManager(
                tools_dir=tools_dir,
                vector_store=vector_store,
                claude_client=claude_client,
                docker_executor=None,  # No Docker
                cache=cache,
            )

            await manager.initialize()
            yield manager

    async def test_create_and_execute_generated_tool_local(
        self, tool_manager_no_docker
    ):
        """Test creating and executing a generated tool locally."""
        # Mock vector store to return no similar tools
        tool_manager_no_docker.vector_store.search.return_value = []

        # Mock Claude structured arguments response
        tool_manager_no_docker.claude_client.structure_arguments.return_value = {
            "x": 3.0,
            "y": 4.0,
        }

        # Create and execute a tool
        result = await tool_manager_no_docker.solve_task(
            task_description="Multiply 3 by 4",
            arguments={"x": 3, "y": 4},
        )

        assert result == 12.0

        # Verify the tool was created and stored
        tools = tool_manager_no_docker.list_tools()
        assert "simple_multiply" in tools
        assert tools["simple_multiply"]["type"] == "generated"

    async def test_generated_tool_docker_execution(self, tool_manager_with_docker):
        """Test that generated tools use Docker when available."""
        # Mock vector store to return no similar tools
        tool_manager_with_docker.vector_store.search.return_value = []

        # Mock Claude structured arguments response
        tool_manager_with_docker.claude_client.structure_arguments.return_value = {
            "a": 5,
            "b": 7,
        }

        # Mock Docker executor response
        tool_manager_with_docker.docker_executor.execute_tool.return_value = 12

        # Create and execute a tool
        result = await tool_manager_with_docker.solve_task(
            task_description="Add 5 and 7",
            arguments={"a": 5, "b": 7},
        )

        assert result == 12

        # Verify Docker was called
        tool_manager_with_docker.docker_executor.execute_tool.assert_called_once()

    async def test_generated_tool_fallback_to_local(self, tool_manager_with_docker):
        """Test that generated tools fall back to local execution when Docker fails."""
        # Mock vector store to return no similar tools
        tool_manager_with_docker.vector_store.search.return_value = []

        # Mock Claude structured arguments response
        tool_manager_with_docker.claude_client.structure_arguments.return_value = {
            "a": 2,
            "b": 3,
        }

        # Mock Docker executor to fail
        tool_manager_with_docker.docker_executor.execute_tool.side_effect = Exception(
            "Docker failed"
        )

        # Create and execute a tool - should fall back to local
        result = await tool_manager_with_docker.solve_task(
            task_description="Add 2 and 3",
            arguments={"a": 2, "b": 3},
        )

        assert result == 5  # Should work locally

        # Verify Docker was attempted but failed
        tool_manager_with_docker.docker_executor.execute_tool.assert_called_once()

    async def test_force_local_execution(self, tool_manager_with_docker):
        """Test forcing local execution even with Docker available."""
        # Mock vector store to return no similar tools
        tool_manager_with_docker.vector_store.search.return_value = []

        # Mock Claude structured arguments response
        tool_manager_with_docker.claude_client.structure_arguments.return_value = {
            "a": 10,
            "b": 20,
        }

        # Create the tool first
        await tool_manager_with_docker.solve_task(
            task_description="Add 10 and 20",
            arguments={"a": 10, "b": 20},
        )

        # Reset Docker mock
        tool_manager_with_docker.docker_executor.reset_mock()

        # Execute directly with force_local=True
        result = await tool_manager_with_docker.execute_tool_directly(
            tool_name="simple_add",
            arguments={"a": 10, "b": 20},
            force_local=True,
        )

        assert result == 30

        # Verify Docker was NOT called
        tool_manager_with_docker.docker_executor.execute_tool.assert_not_called()


class TestToolExecutionModes:
    """Test different tool execution modes and configurations."""

    @pytest.fixture
    async def tool_manager_configurable(self):
        """Create a tool manager that can be configured for different test scenarios."""
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

            # Add a sample generated tool manually for testing
            sample_tool_code = '''
def test_tool(value: str) -> str:
    """A test tool that returns the input value.
    
    Args:
        value: Input value to return
        
    Returns:
        The same input value
    """
    return value
'''

            # Save the tool
            tool_file = tools_dir / "test_tool.py"
            tool_file.write_text(sample_tool_code)

            # Load it manually
            import importlib.util

            spec = importlib.util.spec_from_file_location("test_tool", tool_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                manager.loaded_tools["test_tool"] = module.test_tool
                manager.tool_metadata["test_tool"] = {
                    "code": sample_tool_code,
                    "file": str(tool_file),
                    "function": "test_tool",
                    "type": "generated",
                }

            await manager.initialize()
            yield manager

    async def test_docker_execution_mode(self, tool_manager_configurable):
        """Test Docker execution mode."""
        tool_manager_configurable.docker_executor.execute_tool.return_value = (
            "docker_result"
        )

        result = await tool_manager_configurable.execute_tool_directly(
            tool_name="test_tool",
            arguments={"value": "test"},
            force_local=False,
        )

        assert result == "docker_result"
        tool_manager_configurable.docker_executor.execute_tool.assert_called_once()

    async def test_local_execution_mode(self, tool_manager_configurable):
        """Test local execution mode."""
        result = await tool_manager_configurable.execute_tool_directly(
            tool_name="test_tool",
            arguments={"value": "test"},
            force_local=True,
        )

        assert result == "test"
        tool_manager_configurable.docker_executor.execute_tool.assert_not_called()

    async def test_no_docker_available(self, tool_manager_configurable):
        """Test execution when Docker is not available."""
        # Remove Docker executor
        tool_manager_configurable.docker_executor = None

        result = await tool_manager_configurable.execute_tool_directly(
            tool_name="test_tool",
            arguments={"value": "local_test"},
            force_local=False,
        )

        assert result == "local_test"

    async def test_list_tools_functionality(self, tool_manager_configurable):
        """Test the list_tools functionality."""
        tools = tool_manager_configurable.list_tools()

        assert "test_tool" in tools
        tool_info = tools["test_tool"]

        assert tool_info["type"] == "generated"
        assert "value: str" in tool_info["signature"]
        assert "A test tool" in tool_info["docstring"]


@pytest.mark.integration
class TestToolIntegration:
    """Integration tests for tool execution pipeline."""

    async def test_end_to_end_tool_creation_and_execution(self):
        """Test the complete pipeline from task to execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            tools_dir.mkdir()

            # Create real instances (mocked Claude API)
            vector_store = AsyncMock(spec=VectorStore)
            vector_store.search.return_value = []  # No existing tools

            claude_client = AsyncMock(spec=ClaudeClient)
            claude_client.generate_tool.return_value = '''
def string_reverse(text: str) -> str:
    """Reverse a string.
    
    Args:
        text: The string to reverse
        
    Returns:
        The reversed string
    """
    return text[::-1]
'''
            claude_client.structure_arguments.return_value = {"text": "hello"}

            cache = AsyncMock(spec=SimpleCache)

            # Test with and without Docker
            for use_docker in [True, False]:
                docker_executor = AsyncMock(spec=DockerExecutor) if use_docker else None
                if docker_executor:
                    docker_executor.execute_tool.return_value = "olleh"

                manager = ToolManager(
                    tools_dir=tools_dir,
                    vector_store=vector_store,
                    claude_client=claude_client,
                    docker_executor=docker_executor,
                    cache=cache,
                )

                await manager.initialize()

                # Execute a task that should create a new tool
                result = await manager.solve_task(
                    task_description="Reverse the word hello",
                    arguments={"text": "hello"},
                )

                assert result == "olleh"

                # Verify tool was created
                tools = manager.list_tools()
                assert "string_reverse" in tools

                if use_docker:
                    docker_executor.execute_tool.assert_called_once()

                # Clean up for next iteration
                claude_client.reset_mock()
                if docker_executor:
                    docker_executor.reset_mock()
