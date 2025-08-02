"""Main MCP server implementation."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import TextContent, Tool
from pydantic import BaseModel

from ..core.tool_manager import ToolManager
from ..utils.cache import SimpleCache
from ..utils.claude_client import ClaudeClient
from ..utils.config import (
    Config,
    get_claude_api_key,
    get_initial_tools_dir,
    load_config,
    setup_logging,
)
from ..utils.docker_executor import DockerExecutor
from ..utils.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TaskRequest(BaseModel):
    """Request model for the solve_task tool."""

    task_description: str
    arguments: Dict[str, Any] = {}
    expected_outcome: Optional[str] = None


class ToolSmithMCPServer:
    """Main MCP server that provides the solve_task tool."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the Tool Smith MCP server.

        Args:
            config: Configuration object. If None, loads from config file.
        """
        if config is None:
            config = load_config()

        self.config = config
        self.server = Server(config.server.name)

        # Set up Claude client
        claude_api_key = get_claude_api_key()
        self.claude_client = ClaudeClient(
            api_key=claude_api_key, model=config.claude.model
        )

        # Set up runtime tools directory
        self.tools_dir = Path(config.tools.tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        # Set up cache if enabled
        cache = None
        if config.cache.enabled:
            cache_dir = Path(config.cache.cache_dir)
            cache = SimpleCache(cache_dir)

        # Set up Docker executor if enabled
        docker_executor = None
        if config.docker.enabled:
            try:
                docker_executor = DockerExecutor(
                    image_name=config.docker.image_name,
                    container_timeout=config.docker.container_timeout,
                    memory_limit=config.docker.memory_limit,
                    cpu_limit=config.docker.cpu_limit,
                )
                logger.info("Docker executor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker executor: {e}")
                logger.info("Continuing without Docker sandboxing")

        # Set up vector store
        vector_db_path = Path(config.vector_store.db_path)
        self.vector_store = VectorStore(
            db_path=vector_db_path,
            collection_name=config.vector_store.collection_name,
            model_name=config.vector_store.model_name,
            cache=cache,
        )

        # Set up tool manager
        self.tool_manager = ToolManager(
            tools_dir=self.tools_dir,
            vector_store=self.vector_store,
            claude_client=self.claude_client,
            similarity_threshold=config.tools.similarity_threshold,
            initial_tools_dir=get_initial_tools_dir(config),
            docker_executor=docker_executor,
            cache=cache,
        )

        self._register_tools()

    def _register_tools(self) -> None:
        """Register MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="solve_task",
                    description=(
                        "Solve a task using available tools or by creating new ones. "
                        "Provide a formal description of the task, arguments, and expected outcome."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Formal description of the task to be solved",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments required for the task",
                                "default": {},
                            },
                            "expected_outcome": {
                                "type": "string",
                                "description": "Expected outcome or return type",
                            },
                        },
                        "required": ["task_description"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            if name == "solve_task":
                try:
                    request = TaskRequest(**arguments)
                    result = await self.tool_manager.solve_task(
                        task_description=request.task_description,
                        arguments=request.arguments,
                        expected_outcome=request.expected_outcome,
                    )
                    return [TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.error(f"Error solving task: {e}")
                    return [TextContent(type="text", text=f"Error: {str(e)}")]
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def run(self) -> None:
        """Run the MCP server."""
        # Initialize tool manager
        await self.tool_manager.initialize()

        # Run the server
        await self.server.run()


async def main() -> None:
    """Main entry point."""
    # Load configuration and set up logging
    config = load_config()
    setup_logging(config)

    logger.info(f"Starting {config.server.name} v{config.server.version}")

    server = ToolSmithMCPServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
