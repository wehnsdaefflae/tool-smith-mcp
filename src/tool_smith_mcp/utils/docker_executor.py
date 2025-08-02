"""Docker-based tool execution for sandboxing."""

import contextlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)


class DockerExecutor:
    """Executes tools in Docker containers for security isolation."""

    def __init__(
        self,
        image_name: str = "python:3.11-slim",
        container_timeout: int = 30,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
    ) -> None:
        """Initialize the Docker executor.

        Args:
            image_name: Docker image to use for execution
            container_timeout: Maximum execution time in seconds
            memory_limit: Memory limit for containers
            cpu_limit: CPU limit as fraction of single core
        """
        self.image_name = image_name
        self.container_timeout = container_timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

        # Initialize Docker client
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise

        self._ensure_image_available()

    def _ensure_image_available(self) -> None:
        """Ensure the execution image is available."""
        try:
            self.client.images.get(self.image_name)
            logger.debug(f"Docker image {self.image_name} already available")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image {self.image_name}...")
            self.client.images.pull(self.image_name)
            logger.info(f"Successfully pulled {self.image_name}")

    async def execute_tool(
        self,
        tool_code: str,
        function_name: str,
        arguments: Dict[str, Any],
        tools_dir: Path,
        initial_tools_dir: Optional[Path] = None,
    ) -> Any:
        """Execute a tool function in a Docker container.

        Args:
            tool_code: Python code containing the tool function
            function_name: Name of the function to execute
            arguments: Arguments to pass to the function
            tools_dir: Directory containing runtime tools
            initial_tools_dir: Directory containing initial tools

        Returns:
            Function execution result

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If execution fails
        """
        # Create execution script
        execution_script = self._create_execution_script(
            tool_code, function_name, arguments
        )

        # Create temporary files for input/output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write execution script
            script_file = temp_path / "execute.py"
            script_file.write_text(execution_script)

            # Copy tool dependencies
            deps_dir = temp_path / "tools"
            deps_dir.mkdir()

            # Copy runtime tools
            if tools_dir.exists():
                for tool_file in tools_dir.glob("*.py"):
                    if not tool_file.name.startswith("__"):
                        (deps_dir / tool_file.name).write_text(tool_file.read_text())

            # Copy initial tools
            if initial_tools_dir and initial_tools_dir.exists():
                initial_dir = deps_dir / "initial_tools"
                initial_dir.mkdir()
                for tool_file in initial_tools_dir.glob("*.py"):
                    if not tool_file.name.startswith("__"):
                        (initial_dir / tool_file.name).write_text(tool_file.read_text())

            # Run in container
            return await self._run_container(temp_path)

    def _create_execution_script(
        self,
        tool_code: str,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Create the execution script for the container."""
        return f'''#!/usr/bin/env python3
import json
import sys
import traceback
from pathlib import Path

# Add tools to path
sys.path.insert(0, "/workspace/tools")
sys.path.insert(0, "/workspace/tools/initial_tools")

def main():
    try:
        # Execute the tool code to define the function
        exec("""
{tool_code}
""", globals())
        
        # Get the function and execute it
        if "{function_name}" not in globals():
            raise RuntimeError(f"Function '{function_name}' not found in tool code")
        
        func = globals()["{function_name}"]
        arguments = {json.dumps(arguments)}
        
        # Execute the function
        result = func(**arguments)
        
        # Return result as JSON
        output = {{
            "success": True,
            "result": result,
            "error": None
        }}
        
    except Exception as e:
        # Return error information
        output = {{
            "success": False,
            "result": None,
            "error": {{
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }}
        }}
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
'''

    async def _run_container(self, workspace_path: Path) -> Any:
        """Run the execution script in a Docker container."""
        container: Optional[Container] = None

        try:
            container = self._create_container(workspace_path)
            self._wait_for_completion(container)
            logs = self._get_container_logs(container)
            result_data = self._parse_container_output(logs)
            return self._handle_execution_result(result_data)

        except docker.errors.ContainerError as e:
            logger.error(f"Container execution failed: {e}")
            raise RuntimeError(f"Container execution failed: {e}") from e

        finally:
            self._cleanup_container(container)

    def _create_container(self, workspace_path: Path) -> Any:
        """Create and start the Docker container."""
        return self.client.containers.run(
            image=self.image_name,
            command=["python", "/workspace/execute.py"],
            volumes={str(workspace_path): {"bind": "/workspace", "mode": "ro"}},
            working_dir="/workspace",
            mem_limit=self.memory_limit,
            cpu_period=100000,
            cpu_quota=int(100000 * self.cpu_limit),
            network_disabled=True,  # No network access
            read_only=True,  # Read-only filesystem
            remove=True,  # Auto-remove when done
            detach=True,
            stdout=True,
            stderr=True,
        )

    def _wait_for_completion(self, container: Any) -> None:
        """Wait for container completion with timeout."""
        start_time = time.time()
        while container.status != "exited":
            if time.time() - start_time > self.container_timeout:
                container.kill()
                raise TimeoutError(
                    f"Tool execution exceeded {self.container_timeout}s timeout"
                )
            time.sleep(0.1)
            container.reload()

    def _get_container_logs(self, container: Any) -> str:
        """Get container output logs."""
        return container.logs(stdout=True, stderr=False).decode("utf-8")

    def _parse_container_output(self, logs: str) -> Dict[str, Any]:
        """Parse container output as JSON."""
        try:
            return json.loads(logs.strip())
        except json.JSONDecodeError:
            logger.error(f"Failed to parse container output: {logs}")
            raise RuntimeError(f"Invalid JSON output from container: {logs}") from None

    def _handle_execution_result(self, result_data: Dict[str, Any]) -> Any:
        """Handle the execution result from container."""
        if not result_data.get("success", False):
            error_info = result_data.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            error_type = error_info.get("type", "RuntimeError")

            logger.error(f"Tool execution failed: {error_type}: {error_msg}")
            self._raise_original_exception(error_type, error_msg)

        return result_data.get("result")

    def _raise_original_exception(self, error_type: str, error_msg: str) -> None:
        """Re-raise the original exception type."""
        if error_type == "TimeoutError":
            raise TimeoutError(error_msg)
        elif error_type == "ValueError":
            raise ValueError(error_msg)
        elif error_type == "TypeError":
            raise TypeError(error_msg)
        else:
            raise RuntimeError(f"{error_type}: {error_msg}")

    def _cleanup_container(self, container: Optional[Any]) -> None:
        """Clean up the container."""
        if container:
            with contextlib.suppress(Exception):
                container.remove(force=True)

    def cleanup(self) -> None:
        """Clean up Docker resources."""
        try:
            # Remove any dangling containers from this executor
            containers = self.client.containers.list(
                all=True, filters={"ancestor": self.image_name}
            )
            for container in containers:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

            logger.info("Docker executor cleanup completed")

        except Exception as e:
            logger.warning(f"Error during Docker cleanup: {e}")
