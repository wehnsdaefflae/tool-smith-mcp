"""Tests for the MCP server."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tool_smith_mcp.core.server import TaskRequest, ToolSmithMCPServer


@pytest.fixture
def mock_tool_manager() -> Mock:
    """Create a mock ToolManager."""
    mock_manager = Mock()
    mock_manager.initialize = AsyncMock()
    mock_manager.solve_task = AsyncMock()
    return mock_manager


@pytest.mark.asyncio
async def test_task_request_validation() -> None:
    """Test TaskRequest model validation."""
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
    request_minimal = TaskRequest(task_description="Minimal task")
    assert request_minimal.arguments == {}
    assert request_minimal.expected_outcome is None


@pytest.mark.asyncio
async def test_server_initialization() -> None:
    """Test server initialization."""
    with patch("tool_smith_mcp.core.server.ToolManager") as mock_tool_manager_class:
        with patch("tool_smith_mcp.core.server.VectorStore") as mock_vector_store_class:
            with patch(
                "tool_smith_mcp.core.server.ClaudeClient"
            ) as mock_claude_client_class:
                mock_tool_manager = Mock()
                mock_tool_manager_class.return_value = mock_tool_manager

                server = ToolSmithMCPServer(claude_api_key="test_key")

                # Should have initialized components
                mock_claude_client_class.assert_called_once_with("test_key")
                mock_vector_store_class.assert_called_once()
                mock_tool_manager_class.assert_called_once()

                assert server.tools_dir.name == "tools"
                assert server.claude_client is not None
                assert server.vector_store is not None
                assert server.tool_manager is not None


@pytest.mark.asyncio
async def test_solve_task_success() -> None:
    """Test successful task solving."""
    with patch("tool_smith_mcp.core.server.ToolManager") as mock_tool_manager_class:
        with patch("tool_smith_mcp.core.server.VectorStore"):
            with patch("tool_smith_mcp.core.server.ClaudeClient"):
                mock_tool_manager = Mock()
                mock_tool_manager.solve_task = AsyncMock(return_value="Task result")
                mock_tool_manager_class.return_value = mock_tool_manager

                server = ToolSmithMCPServer(claude_api_key="test_key")

                # Create a mock tool call
                arguments = {
                    "task_description": "Test task",
                    "arguments": {"param": "value"},
                    "expected_outcome": "Test outcome",
                }

                # Get the call_tool handler from the server
                tool_handlers = []

                # Mock the server's call_tool decorator
                original_call_tool = server.server.call_tool

                def mock_call_tool():
                    def decorator(func):
                        tool_handlers.append(func)
                        return func

                    return decorator

                server.server.call_tool = mock_call_tool

                # Re-register tools to capture the handler
                server._register_tools()

                # Call the handler directly
                assert len(tool_handlers) == 1
                handler = tool_handlers[0]

                result = await handler("solve_task", arguments)

                # Should have called the tool manager
                mock_tool_manager.solve_task.assert_called_once_with(
                    task_description="Test task",
                    arguments={"param": "value"},
                    expected_outcome="Test outcome",
                )

                # Should return the result
                assert len(result) == 1
                assert result[0].text == "Task result"


@pytest.mark.asyncio
async def test_solve_task_error_handling() -> None:
    """Test error handling in task solving."""
    with patch("tool_smith_mcp.core.server.ToolManager") as mock_tool_manager_class:
        with patch("tool_smith_mcp.core.server.VectorStore"):
            with patch("tool_smith_mcp.core.server.ClaudeClient"):
                mock_tool_manager = Mock()
                mock_tool_manager.solve_task = AsyncMock(
                    side_effect=Exception("Test error")
                )
                mock_tool_manager_class.return_value = mock_tool_manager

                server = ToolSmithMCPServer(claude_api_key="test_key")

                # Create a mock tool call
                arguments = {
                    "task_description": "Test task",
                }

                # Get the call_tool handler
                tool_handlers = []

                def mock_call_tool():
                    def decorator(func):
                        tool_handlers.append(func)
                        return func

                    return decorator

                server.server.call_tool = mock_call_tool
                server._register_tools()

                handler = tool_handlers[0]
                result = await handler("solve_task", arguments)

                # Should return error message
                assert len(result) == 1
                assert "Error: Test error" in result[0].text


@pytest.mark.asyncio
async def test_unknown_tool_handling() -> None:
    """Test handling of unknown tool calls."""
    with patch("tool_smith_mcp.core.server.ToolManager"):
        with patch("tool_smith_mcp.core.server.VectorStore"):
            with patch("tool_smith_mcp.core.server.ClaudeClient"):
                server = ToolSmithMCPServer(claude_api_key="test_key")

                # Get the call_tool handler
                tool_handlers = []

                def mock_call_tool():
                    def decorator(func):
                        tool_handlers.append(func)
                        return func

                    return decorator

                server.server.call_tool = mock_call_tool
                server._register_tools()

                handler = tool_handlers[0]
                result = await handler("unknown_tool", {})

                # Should return unknown tool message
                assert len(result) == 1
                assert "Unknown tool: unknown_tool" in result[0].text


def test_main_missing_api_key() -> None:
    """Test main function with missing API key."""
    import os

    from tool_smith_mcp.core.server import main

    # Ensure API key is not set
    if "CLAUDE_API_KEY" in os.environ:
        del os.environ["CLAUDE_API_KEY"]

    with pytest.raises(
        ValueError, match="CLAUDE_API_KEY environment variable is required"
    ):
        import asyncio

        asyncio.run(main())
