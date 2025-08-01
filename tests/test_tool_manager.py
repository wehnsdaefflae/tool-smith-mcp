"""Tests for the ToolManager class."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from tool_smith_mcp.core.tool_manager import ToolManager
from tool_smith_mcp.utils.vector_store import VectorStore
from tool_smith_mcp.utils.claude_client import ClaudeClient


@pytest.fixture
def temp_tools_dir() -> Path:
    """Create a temporary directory for tools."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary directory for vector database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_claude_client() -> Mock:
    """Create a mock Claude client."""
    mock_client = Mock(spec=ClaudeClient)
    mock_client.generate_tool = AsyncMock()
    mock_client.structure_arguments = AsyncMock()
    return mock_client


@pytest.fixture
def vector_store(temp_db_path: Path) -> VectorStore:
    """Create a VectorStore instance for testing."""
    return VectorStore(temp_db_path)


@pytest.fixture
def tool_manager(temp_tools_dir: Path, vector_store: VectorStore, mock_claude_client: Mock) -> ToolManager:
    """Create a ToolManager instance for testing."""
    return ToolManager(
        tools_dir=temp_tools_dir,
        vector_store=vector_store,
        claude_client=mock_claude_client,
        similarity_threshold=0.7,
    )


@pytest.mark.asyncio
async def test_initialize_with_no_tools(tool_manager: ToolManager) -> None:
    """Test that initialization works when no tools exist."""
    await tool_manager.initialize()
    
    # Should have no tools loaded from empty directory
    assert len(tool_manager.loaded_tools) == 0


@pytest.mark.asyncio
async def test_load_existing_tools(tool_manager: ToolManager, temp_tools_dir: Path) -> None:
    """Test loading existing tools from files."""
    # Create a test tool file
    tool_code = '''def test_tool(value: str) -> str:
    """A test tool for testing.
    
    Args:
        value: Input value to process
        
    Returns:
        Processed value
    """
    return value.upper()
'''
    
    tool_file = temp_tools_dir / "test_tool.py"
    tool_file.write_text(tool_code)
    
    await tool_manager.initialize()
    
    # Should have loaded the test tool
    assert "test_tool" in tool_manager.loaded_tools
    
    # Test the loaded function
    result = tool_manager.loaded_tools["test_tool"]("hello")
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_solve_task_with_existing_tool(tool_manager: ToolManager, temp_tools_dir: Path) -> None:
    """Test solving a task with an existing tool."""
    # Create a test tool
    tool_code = '''def calculate_math(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    return float(eval(expression))
'''
    tool_file = temp_tools_dir / "calculate_math.py"
    tool_file.write_text(tool_code)
    
    await tool_manager.initialize()
    
    # Mock vector search to return a matching tool
    with patch.object(tool_manager.vector_store, 'search') as mock_search:
        mock_search.return_value = [("calculate_math", 0.8, "desc", {})]
        
        # Mock argument structuring
        tool_manager.claude_client.structure_arguments.return_value = {"expression": "2 + 3"}
        
        # Solve task
        result = await tool_manager.solve_task(
            task_description="Calculate 2 + 3",
            arguments={"expression": "2 + 3"},
        )
        
        # Should return the calculation result
        assert result == 5.0


@pytest.mark.asyncio
async def test_solve_task_creates_new_tool(tool_manager: ToolManager, mock_claude_client: Mock) -> None:
    """Test solving a task by creating a new tool."""
    # Initialize with initial tools
    await tool_manager.initialize()
    
    # Mock vector search to return no suitable tools
    with patch.object(tool_manager.vector_store, 'search') as mock_search:
        mock_search.return_value = [("some_tool", 0.3, "desc", {})]  # Low similarity
        
        # Mock tool generation
        generated_code = '''def reverse_string(text: str) -> str:
    """Reverse a string.
    
    Args:
        text: String to reverse
        
    Returns:
        Reversed string
    """
    return text[::-1]
'''
        mock_claude_client.generate_tool.return_value = generated_code
        mock_claude_client.structure_arguments.return_value = {"text": "hello"}
        
        # Solve task
        result = await tool_manager.solve_task(
            task_description="Reverse the string 'hello'",
            arguments={"text": "hello"},
        )
        
        # Should have created and used the new tool
        assert result == "olleh"
        assert "reverse_string" in tool_manager.loaded_tools
        
        # Tool should be saved to file
        tool_file = tool_manager.tools_dir / "reverse_string.py"
        assert tool_file.exists()


@pytest.mark.asyncio
async def test_get_existing_tools_context(tool_manager: ToolManager, temp_tools_dir: Path) -> None:
    """Test getting context about existing tools."""
    # Create test tools
    tool_code = '''def test_tool(value: str) -> str:
    """A test tool for testing.
    
    Args:
        value: Input value to process
        
    Returns:
        Processed value
    """
    return value.upper()
'''
    tool_file = temp_tools_dir / "test_tool.py"
    tool_file.write_text(tool_code)
    
    await tool_manager.initialize()
    
    context = tool_manager._get_existing_tools_context()
    
    # Should contain information about loaded tools
    assert "test_tool" in context
    assert "Args:" in context
    assert "Returns:" in context


def test_extract_function_name(tool_manager: ToolManager) -> None:
    """Test extracting function name from code."""
    code = '''def my_function(param: str) -> str:
    """A test function."""
    return param
'''
    
    name = tool_manager._extract_function_name(code)
    assert name == "my_function"
    
    # Test with invalid code
    invalid_code = "not valid python code"
    name = tool_manager._extract_function_name(invalid_code)
    assert name.startswith("tool_")  # Should generate hash-based name


@pytest.mark.asyncio
async def test_error_handling_in_solve_task(tool_manager: ToolManager) -> None:
    """Test error handling when task execution fails."""
    await tool_manager.initialize()
    
    # Mock vector search to return a tool
    with patch.object(tool_manager.vector_store, 'search') as mock_search:
        mock_search.return_value = [("calculate_math", 0.8, "desc", {})]
        
        # Mock argument structuring to return invalid arguments
        tool_manager.claude_client.structure_arguments.return_value = {"expression": "invalid"}
        
        # Should raise an exception
        with pytest.raises(Exception):
            await tool_manager.solve_task(
                task_description="Calculate something invalid",
                arguments={"expression": "invalid"},
            )