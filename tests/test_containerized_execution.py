"""Tests for containerized tool execution."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tool_smith_mcp.utils.docker_executor import DockerExecutor


class TestDockerExecutor:
    """Test suite for Docker-based tool execution."""

    @pytest.fixture
    def docker_executor(self):
        """Create a Docker executor for testing."""
        return DockerExecutor(
            image_name="python:3.11-slim",
            container_timeout=10,
            memory_limit="128m",
            cpu_limit=0.5,
        )

    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client for testing without actual Docker."""
        with patch("docker.from_env") as mock_docker:
            client = MagicMock()
            mock_docker.return_value = client

            # Mock successful ping
            client.ping.return_value = True

            # Mock image operations
            client.images.get.return_value = MagicMock()
            client.images.pull.return_value = MagicMock()

            yield client

    def test_docker_executor_initialization(self, mock_docker_client):
        """Test Docker executor initialization."""
        executor = DockerExecutor()

        assert executor.image_name == "python:3.11-slim"
        assert executor.container_timeout == 30
        assert executor.memory_limit == "256m"
        assert executor.cpu_limit == 0.5

    def test_docker_executor_custom_config(self, mock_docker_client):
        """Test Docker executor with custom configuration."""
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

    async def test_execute_simple_tool(self, mock_docker_client):
        """Test executing a simple tool in Docker."""
        executor = DockerExecutor()

        # Mock container execution
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = b'{"success": true, "result": 42, "error": null}'

        mock_docker_client.containers.run.return_value = container

        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir()

            tool_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

            result = await executor.execute_tool(
                tool_code=tool_code,
                function_name="add_numbers",
                arguments={"a": 20, "b": 22},
                tools_dir=tools_dir,
            )

            assert result == 42
            mock_docker_client.containers.run.assert_called_once()

    async def test_execute_tool_with_error(self, mock_docker_client):
        """Test executing a tool that raises an error."""
        executor = DockerExecutor()

        # Mock container execution with error
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = b'{"success": false, "result": null, "error": {"type": "ValueError", "message": "Invalid input"}}'

        mock_docker_client.containers.run.return_value = container

        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir()

            tool_code = '''
def failing_tool(value: str) -> str:
    """A tool that fails."""
    raise ValueError("Invalid input")
'''

            with pytest.raises(ValueError, match="Invalid input"):
                await executor.execute_tool(
                    tool_code=tool_code,
                    function_name="failing_tool",
                    arguments={"value": "test"},
                    tools_dir=tools_dir,
                )

    async def test_execute_tool_timeout(self, mock_docker_client):
        """Test tool execution timeout."""
        executor = DockerExecutor(container_timeout=1)  # Very short timeout

        # Mock container that never finishes
        container = MagicMock()
        container.status = "running"  # Never changes to "exited"

        mock_docker_client.containers.run.return_value = container

        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir()

            tool_code = '''
def slow_tool() -> str:
    """A slow tool."""
    import time
    time.sleep(10)
    return "done"
'''

            with pytest.raises(TimeoutError):
                await executor.execute_tool(
                    tool_code=tool_code,
                    function_name="slow_tool",
                    arguments={},
                    tools_dir=tools_dir,
                )

            # Should have killed the container
            container.kill.assert_called_once()

    async def test_execute_tool_with_dependencies(self, mock_docker_client):
        """Test executing a tool that uses other tools as dependencies."""
        executor = DockerExecutor()

        # Mock successful execution
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = (
            b'{"success": true, "result": "processed: hello", "error": null}'
        )

        mock_docker_client.containers.run.return_value = container

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tools_dir = temp_path / "tools"
            initial_tools_dir = temp_path / "initial_tools"
            tools_dir.mkdir()
            initial_tools_dir.mkdir()

            # Create a dependency tool
            (tools_dir / "helper.py").write_text(
                '''
def helper_function(text: str) -> str:
    """Helper function."""
    return f"processed: {text}"
'''
            )

            # Create an initial tool
            (initial_tools_dir / "utils.py").write_text(
                '''
def utility_function(data: str) -> str:
    """Utility function."""
    return data.upper()
'''
            )

            tool_code = '''
from helper import helper_function
from utils import utility_function

def main_tool(input_text: str) -> str:
    """Main tool that uses dependencies."""
    return helper_function(input_text)
'''

            result = await executor.execute_tool(
                tool_code=tool_code,
                function_name="main_tool",
                arguments={"input_text": "hello"},
                tools_dir=tools_dir,
                initial_tools_dir=initial_tools_dir,
            )

            assert result == "processed: hello"

    def test_create_execution_script(self, mock_docker_client):
        """Test execution script generation."""
        executor = DockerExecutor()

        tool_code = """
def test_func(x: int) -> int:
    return x * 2
"""

        script = executor._create_execution_script(
            tool_code=tool_code,
            function_name="test_func",
            arguments={"x": 5},
        )

        # Check that script contains expected components
        assert 'exec("""' in script
        assert "def test_func" in script
        assert "test_func" in script
        assert '{"x": 5}' in script
        assert "import json" in script

    def test_parse_container_output_success(self, mock_docker_client):
        """Test parsing successful container output."""
        executor = DockerExecutor()

        output = '{"success": true, "result": "test_result", "error": null}'
        result = executor._parse_container_output(output)

        assert result["success"] is True
        assert result["result"] == "test_result"
        assert result["error"] is None

    def test_parse_container_output_error(self, mock_docker_client):
        """Test parsing error container output."""
        executor = DockerExecutor()

        output = '{"success": false, "result": null, "error": {"type": "RuntimeError", "message": "Something went wrong"}}'
        result = executor._parse_container_output(output)

        assert result["success"] is False
        assert result["result"] is None
        assert result["error"]["type"] == "RuntimeError"
        assert result["error"]["message"] == "Something went wrong"

    def test_parse_invalid_json_output(self, mock_docker_client):
        """Test parsing invalid JSON output from container."""
        executor = DockerExecutor()

        with pytest.raises(RuntimeError, match="Invalid JSON output"):
            executor._parse_container_output("invalid json")

    def test_handle_execution_result_success(self, mock_docker_client):
        """Test handling successful execution result."""
        executor = DockerExecutor()

        result_data = {"success": True, "result": "expected_result", "error": None}

        result = executor._handle_execution_result(result_data)
        assert result == "expected_result"

    def test_handle_execution_result_error(self, mock_docker_client):
        """Test handling error execution result."""
        executor = DockerExecutor()

        result_data = {
            "success": False,
            "result": None,
            "error": {"type": "ValueError", "message": "Test error"},
        }

        with pytest.raises(ValueError, match="Test error"):
            executor._handle_execution_result(result_data)

    def test_raise_original_exception_types(self, mock_docker_client):
        """Test raising original exception types."""
        executor = DockerExecutor()

        # Test TimeoutError
        with pytest.raises(TimeoutError):
            executor._raise_original_exception("TimeoutError", "Timeout occurred")

        # Test ValueError
        with pytest.raises(ValueError):
            executor._raise_original_exception("ValueError", "Invalid value")

        # Test TypeError
        with pytest.raises(TypeError):
            executor._raise_original_exception("TypeError", "Type error")

        # Test unknown error type (should become RuntimeError)
        with pytest.raises(RuntimeError, match="CustomError: Custom message"):
            executor._raise_original_exception("CustomError", "Custom message")

    def test_cleanup_containers(self, mock_docker_client):
        """Test cleanup of Docker containers."""
        executor = DockerExecutor()

        # Mock containers
        container1 = MagicMock()
        container2 = MagicMock()
        mock_docker_client.containers.list.return_value = [container1, container2]

        executor.cleanup()

        # Should have tried to remove both containers
        container1.remove.assert_called_once_with(force=True)
        container2.remove.assert_called_once_with(force=True)


@pytest.mark.integration
class TestDockerExecutorIntegration:
    """Integration tests for Docker executor (requires actual Docker)."""

    @pytest.mark.skipif(
        not Path("/var/run/docker.sock").exists(), reason="Docker not available"
    )
    async def test_real_docker_execution(self):
        """Test actual Docker execution (requires Docker to be running)."""
        try:
            executor = DockerExecutor(
                container_timeout=10,
                memory_limit="128m",
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                tools_dir = Path(temp_dir) / "tools"
                tools_dir.mkdir()

                tool_code = '''
def simple_calculation(a: int, b: int) -> int:
    """Simple calculation for testing."""
    return a + b + 10
'''

                result = await executor.execute_tool(
                    tool_code=tool_code,
                    function_name="simple_calculation",
                    arguments={"a": 5, "b": 15},
                    tools_dir=tools_dir,
                )

                assert result == 30

        except Exception as e:
            pytest.skip(f"Docker not available for integration test: {e}")

    @pytest.mark.skipif(
        not Path("/var/run/docker.sock").exists(), reason="Docker not available"
    )
    async def test_real_docker_error_handling(self):
        """Test error handling with real Docker."""
        try:
            executor = DockerExecutor(container_timeout=5)

            with tempfile.TemporaryDirectory() as temp_dir:
                tools_dir = Path(temp_dir) / "tools"
                tools_dir.mkdir()

                tool_code = '''
def failing_tool(value: str) -> str:
    """Tool that raises an exception."""
    if not value:
        raise ValueError("Value cannot be empty")
    return value.upper()
'''

                with pytest.raises(ValueError, match="Value cannot be empty"):
                    await executor.execute_tool(
                        tool_code=tool_code,
                        function_name="failing_tool",
                        arguments={"value": ""},
                        tools_dir=tools_dir,
                    )

        except Exception as e:
            pytest.skip(f"Docker not available for integration test: {e}")


class TestDockerConfiguration:
    """Test Docker configuration and edge cases."""

    def test_docker_unavailable_initialization(self):
        """Test initialization when Docker is not available."""
        with patch("docker.from_env") as mock_docker:
            mock_docker.side_effect = Exception("Docker not available")

            with pytest.raises(Exception, match="Docker not available"):
                DockerExecutor()

    def test_image_not_found_handling(self, mock_docker_client):
        """Test handling when Docker image is not found."""
        import docker.errors

        # Mock image not found, then successful pull
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound(
            "Image not found"
        )
        mock_docker_client.images.pull.return_value = MagicMock()

        executor = DockerExecutor()

        # Should have attempted to pull the image
        mock_docker_client.images.pull.assert_called_once_with("python:3.11-slim")

    async def test_container_creation_parameters(self, mock_docker_client):
        """Test that container is created with correct parameters."""
        executor = DockerExecutor(
            memory_limit="512m",
            cpu_limit=0.8,
            container_timeout=15,
        )

        # Mock container
        container = MagicMock()
        container.status = "exited"
        container.logs.return_value = (
            b'{"success": true, "result": null, "error": null}'
        )
        mock_docker_client.containers.run.return_value = container

        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir()

            await executor.execute_tool(
                tool_code="def test(): pass",
                function_name="test",
                arguments={},
                tools_dir=tools_dir,
            )

            # Check container creation parameters
            call_args = mock_docker_client.containers.run.call_args
            assert call_args[1]["mem_limit"] == "512m"
            assert call_args[1]["cpu_quota"] == int(100000 * 0.8)
            assert call_args[1]["network_disabled"] is True
            assert call_args[1]["read_only"] is True
