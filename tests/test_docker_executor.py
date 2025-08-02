"""Tests for Docker executor."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tool_smith_mcp.utils.docker_executor import DockerExecutor


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tools_dir = temp_path / "tools"
        initial_tools_dir = temp_path / "initial_tools"

        tools_dir.mkdir()
        initial_tools_dir.mkdir()

        # Create sample tool file
        sample_tool = tools_dir / "helper_tool.py"
        sample_tool.write_text(
            '''
def helper_function(x: int) -> int:
    """Helper function that adds 1 to input."""
    return x + 1
'''
        )

        yield {
            "tools_dir": tools_dir,
            "initial_tools_dir": initial_tools_dir,
        }


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    mock_client = Mock()

    # Mock successful container execution
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.logs.return_value = b'{"success": true, "result": 42, "error": null}'

    mock_client.containers.run.return_value = mock_container
    mock_client.images.get.return_value = Mock()  # Image exists
    mock_client.ping.return_value = True

    return mock_client


@patch("tool_smith_mcp.utils.docker_executor.docker")
def test_docker_executor_initialization(mock_docker, mock_docker_client):
    """Test Docker executor initialization."""
    mock_docker.from_env.return_value = mock_docker_client

    executor = DockerExecutor()

    assert executor.image_name == "python:3.11-slim"
    assert executor.container_timeout == 30
    mock_docker_client.ping.assert_called_once()


@patch("tool_smith_mcp.utils.docker_executor.docker")
def test_docker_executor_initialization_failure(mock_docker):
    """Test Docker executor initialization when Docker is not available."""
    mock_docker.from_env.side_effect = Exception("Docker not available")

    with pytest.raises(Exception, match="Docker not available"):
        DockerExecutor()


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
async def test_successful_tool_execution(mock_docker, mock_docker_client, temp_dirs):
    """Test successful tool execution in Docker."""
    mock_docker.from_env.return_value = mock_docker_client

    executor = DockerExecutor()

    tool_code = '''
def calculate_square(x: int) -> int:
    """Calculate the square of a number."""
    return x * x
'''

    result = await executor.execute_tool(
        tool_code=tool_code,
        function_name="calculate_square",
        arguments={"x": 5},
        tools_dir=temp_dirs["tools_dir"],
        initial_tools_dir=temp_dirs["initial_tools_dir"],
    )

    assert result == 42  # From mock response
    mock_docker_client.containers.run.assert_called_once()


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
async def test_tool_execution_with_error(mock_docker, mock_docker_client, temp_dirs):
    """Test tool execution when the tool raises an error."""
    mock_docker.from_env.return_value = mock_docker_client

    # Mock container that returns an error
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.logs.return_value = b"""{"success": false, "result": null, "error": {"type": "ValueError", "message": "Invalid input", "traceback": "..."}}"""

    mock_docker_client.containers.run.return_value = mock_container

    executor = DockerExecutor()

    tool_code = '''
def failing_function(x: int) -> int:
    """A function that always fails."""
    raise ValueError("Invalid input")
'''

    with pytest.raises(ValueError, match="Invalid input"):
        await executor.execute_tool(
            tool_code=tool_code,
            function_name="failing_function",
            arguments={"x": 5},
            tools_dir=temp_dirs["tools_dir"],
            initial_tools_dir=temp_dirs["initial_tools_dir"],
        )


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
@patch("tool_smith_mcp.utils.docker_executor.time")
async def test_tool_execution_timeout(
    mock_time, mock_docker, mock_docker_client, temp_dirs
):
    """Test tool execution timeout."""
    mock_docker.from_env.return_value = mock_docker_client

    # Mock container that never exits
    mock_container = Mock()
    mock_container.status = "running"  # Never becomes "exited"

    mock_docker_client.containers.run.return_value = mock_container

    # Mock time.time() to simulate timeout
    mock_time.time.side_effect = [0, 0.1, 31]  # Start, check, timeout
    mock_time.sleep = Mock()

    executor = DockerExecutor(container_timeout=30)

    tool_code = '''
def slow_function() -> int:
    """A function that takes too long."""
    import time
    time.sleep(100)
    return 42
'''

    with pytest.raises(TimeoutError, match="exceeded 30s timeout"):
        await executor.execute_tool(
            tool_code=tool_code,
            function_name="slow_function",
            arguments={},
            tools_dir=temp_dirs["tools_dir"],
            initial_tools_dir=temp_dirs["initial_tools_dir"],
        )

    # Should have killed the container
    mock_container.kill.assert_called_once()


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
async def test_tool_execution_invalid_json_response(
    mock_docker, mock_docker_client, temp_dirs
):
    """Test handling of invalid JSON response from container."""
    mock_docker.from_env.return_value = mock_docker_client

    # Mock container that returns invalid JSON
    mock_container = Mock()
    mock_container.status = "exited"
    mock_container.logs.return_value = b"invalid json response"

    mock_docker_client.containers.run.return_value = mock_container

    executor = DockerExecutor()

    tool_code = '''
def simple_function() -> int:
    """A simple function."""
    return 42
'''

    with pytest.raises(RuntimeError, match="Invalid JSON output"):
        await executor.execute_tool(
            tool_code=tool_code,
            function_name="simple_function",
            arguments={},
            tools_dir=temp_dirs["tools_dir"],
            initial_tools_dir=temp_dirs["initial_tools_dir"],
        )


def test_execution_script_generation():
    """Test the execution script generation."""
    executor = DockerExecutor()

    tool_code = '''
def test_function(x: int, y: str) -> str:
    """Test function."""
    return f"{y}: {x}"
'''

    script = executor._create_execution_script(
        tool_code=tool_code,
        function_name="test_function",
        arguments={"x": 42, "y": "answer"},
    )

    # Should contain the tool code
    assert "def test_function(x: int, y: str) -> str:" in script

    # Should contain the function call
    assert '"test_function"' in script

    # Should contain the arguments
    assert '"x": 42' in script
    assert '"y": "answer"' in script

    # Should have error handling
    assert "try:" in script
    assert "except Exception" in script


@patch("tool_smith_mcp.utils.docker_executor.docker")
def test_docker_executor_cleanup(mock_docker, mock_docker_client):
    """Test Docker executor cleanup."""
    mock_docker.from_env.return_value = mock_docker_client

    # Mock containers list
    mock_container1 = Mock()
    mock_container2 = Mock()
    mock_docker_client.containers.list.return_value = [mock_container1, mock_container2]

    executor = DockerExecutor()
    executor.cleanup()

    # Should have listed containers and removed them
    mock_docker_client.containers.list.assert_called_once()
    mock_container1.remove.assert_called_once_with(force=True)
    mock_container2.remove.assert_called_once_with(force=True)


@patch("tool_smith_mcp.utils.docker_executor.docker")
def test_ensure_image_available_pull(mock_docker, mock_docker_client):
    """Test image pulling when image is not available."""
    mock_docker.from_env.return_value = mock_docker_client

    # Mock image not found, then successful pull
    from docker.errors import ImageNotFound

    mock_docker_client.images.get.side_effect = ImageNotFound("Image not found")
    mock_docker_client.images.pull.return_value = Mock()

    executor = DockerExecutor()

    # Should have attempted to pull the image
    mock_docker_client.images.pull.assert_called_once_with("python:3.11-slim")


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
async def test_container_resource_limits(mock_docker, mock_docker_client, temp_dirs):
    """Test that container resource limits are applied correctly."""
    mock_docker.from_env.return_value = mock_docker_client

    executor = DockerExecutor(memory_limit="512m", cpu_limit=0.8)

    tool_code = """
def simple_function() -> int:
    return 42
"""

    await executor.execute_tool(
        tool_code=tool_code,
        function_name="simple_function",
        arguments={},
        tools_dir=temp_dirs["tools_dir"],
        initial_tools_dir=temp_dirs["initial_tools_dir"],
    )

    # Check that resource limits were applied
    call_args = mock_docker_client.containers.run.call_args
    assert call_args[1]["mem_limit"] == "512m"
    assert call_args[1]["cpu_quota"] == 80000  # 0.8 * 100000
    assert call_args[1]["network_disabled"] is True
    assert call_args[1]["read_only"] is True


@pytest.mark.asyncio
@patch("tool_smith_mcp.utils.docker_executor.docker")
async def test_tool_with_dependencies(mock_docker, mock_docker_client, temp_dirs):
    """Test tool execution with dependencies on other tools."""
    mock_docker.from_env.return_value = mock_docker_client

    executor = DockerExecutor()

    # Create a tool that imports from another tool
    tool_code = '''
import sys
sys.path.insert(0, "/workspace/tools")
from helper_tool import helper_function

def dependent_function(x: int) -> int:
    """Function that depends on another tool."""
    return helper_function(x) * 2
'''

    result = await executor.execute_tool(
        tool_code=tool_code,
        function_name="dependent_function",
        arguments={"x": 5},
        tools_dir=temp_dirs["tools_dir"],
        initial_tools_dir=temp_dirs["initial_tools_dir"],
    )

    assert result == 42  # From mock response

    # Verify that the workspace was set up with tool dependencies
    call_args = mock_docker_client.containers.run.call_args
    volumes = call_args[1]["volumes"]

    # Should have mounted the workspace
    assert any("/workspace" in str(volume) for volume in volumes.values())
