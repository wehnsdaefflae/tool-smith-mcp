"""Isolated unit tests for ToolManager without MCP server dependencies.

These tests demonstrate how to test the ToolManager module independently,
with all external dependencies mocked for fast, reliable unit testing.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from tool_smith_mcp.core.tool_manager import ToolManager
from tool_smith_mcp.models import ToolInfo, ToolType
from tests.conftest import (
    create_test_tool_file, 
    create_sample_tool_code,
    setup_claude_mock_responses,
    setup_vector_store_mock_responses,
    setup_docker_mock_responses,
    run_tool_manager_test
)


class TestToolManagerIsolated:
    """Test ToolManager functionality with all dependencies mocked."""

    @pytest.mark.asyncio
    async def test_initialize_empty_directories(self, tool_manager: ToolManager):
        """Test initialization with empty tool directories."""
        await tool_manager.initialize()
        
        # Should have no tools loaded
        assert len(tool_manager.loaded_tools) == 0
        assert len(tool_manager.tool_metadata) == 0
        
        # Should have called vector store initialize
        tool_manager.vector_store.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_initial_tools(
        self, 
        tool_manager: ToolManager,
        temp_initial_tools_dir: Path
    ):
        """Test loading initial tools from the initial tools directory."""
        # Create a sample initial tool
        tool_code = create_sample_tool_code(
            "sample_initial_tool",
            "Sample initial tool for testing",
            implementation='return f"initial_{value}"'
        )
        create_test_tool_file(temp_initial_tools_dir, "sample_initial", tool_code)
        
        await tool_manager.initialize()
        
        # Should have loaded the initial tool
        assert "sample_initial_tool" in tool_manager.loaded_tools
        assert "sample_initial_tool" in tool_manager.tool_metadata
        
        # Metadata should indicate it's an initial tool
        metadata = tool_manager.tool_metadata["sample_initial_tool"]
        assert metadata["type"] == "initial"
        assert "sample_initial.py" in metadata["file"]
        assert metadata["code"] is not None
        
        # Should have added to vector store
        tool_manager.vector_store.add_document.assert_called()

    @pytest.mark.asyncio
    async def test_load_generated_tools(
        self,
        tool_manager: ToolManager,
        temp_tools_dir: Path
    ):
        """Test loading previously generated tools from tools directory."""
        # Create a sample generated tool
        tool_code = create_sample_tool_code(
            "sample_generated_tool",
            "Sample generated tool for testing",
            implementation='return f"generated_{value}"'
        )
        create_test_tool_file(temp_tools_dir, "sample_generated", tool_code)
        
        await tool_manager.initialize()
        
        # Should have loaded the generated tool
        assert "sample_generated_tool" in tool_manager.loaded_tools
        
        # Should have added to vector store
        tool_manager.vector_store.add_document.assert_called()

    @pytest.mark.asyncio
    async def test_solve_task_with_existing_tool(self, tool_manager: ToolManager):
        """Test solving a task using an existing tool found by similarity search."""
        # Set up mocks
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            [("existing_tool", 0.8, "A test tool", {"type": "initial"})]
        )
        
        setup_claude_mock_responses(
            tool_manager.claude_client,
            structure_args_response={"value": "test_input"}
        )
        
        # Create a mock tool function
        mock_tool = Mock(return_value="existing_tool_result")
        mock_tool.__name__ = "existing_tool"
        mock_tool.__doc__ = "Test existing tool"
        
        tool_manager.loaded_tools["existing_tool"] = mock_tool
        tool_manager.tool_metadata["existing_tool"] = {
            "type": "initial",
            "code": "mock_code",
            "file": "existing_tool.py",
            "function": "existing_tool"
        }
        
        # Solve task
        result = await tool_manager.solve_task(
            task_description="Use existing tool for test",
            arguments={"value": "test_input"}
        )
        
        # Should have used existing tool
        assert result == "existing_tool_result"
        mock_tool.assert_called_once_with(value="test_input")
        
        # Should not have generated new tool
        tool_manager.claude_client.generate_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_solve_task_creates_new_tool(self, tool_manager: ToolManager):
        """Test solving a task by creating a new tool when no suitable tool exists."""
        # Set up mocks for no suitable existing tools
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            [("unrelated_tool", 0.3, "Unrelated tool", {})]  # Low similarity
        )
        
        # Set up Claude to generate a new tool
        generated_code = create_sample_tool_code(
            "new_generated_tool",
            "A newly generated tool",
            implementation='return f"new_{value}"'
        )
        
        setup_claude_mock_responses(
            tool_manager.claude_client,
            tool_generation_response=generated_code,
            structure_args_response={"value": "test_input"}
        )
        
        await tool_manager.initialize()
        
        # Solve task
        result = await tool_manager.solve_task(
            task_description="Create new tool for testing",
            arguments={"value": "test_input"}
        )
        
        # Should have generated and used new tool
        assert "new_generated_tool" in tool_manager.loaded_tools
        assert "new_generated_tool" in tool_manager.tool_metadata
        
        # Metadata should indicate it's a generated tool
        metadata = tool_manager.tool_metadata["new_generated_tool"]
        assert metadata["type"] == "generated"
        
        # Should have saved tool to file
        tool_file = tool_manager.tools_dir / "new_generated_tool.py"
        assert tool_file.exists()
        
        # Should have added to vector store
        tool_manager.vector_store.add_document.assert_called()

    @pytest.mark.asyncio
    async def test_execute_tool_safely_with_docker(self, tool_manager: ToolManager):
        """Test safe tool execution using Docker container."""
        # Set up Docker executor
        setup_docker_mock_responses(
            tool_manager.docker_executor,
            execution_result="docker_result"
        )
        
        # Create a generated tool (should use Docker)
        mock_tool = Mock()
        mock_tool.__name__ = "test_tool"
        
        tool_manager.tool_metadata["test_tool"] = {
            "type": "generated",
            "code": "def test_tool(): return 'test'",
            "file": "test_tool.py",
            "function": "test_tool"
        }
        
        # Execute tool
        result = await tool_manager._execute_tool_safely(
            tool_func=mock_tool,
            arguments={"value": "test"},
            task_description="Test task"
        )
        
        # Should have used Docker executor
        assert result == "docker_result"
        tool_manager.docker_executor.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_safely_locally(self, tool_manager: ToolManager):
        """Test safe tool execution locally when Docker is disabled or forced local."""
        # Create an initial tool (should run locally)
        mock_tool = Mock(return_value="local_result")
        mock_tool.__name__ = "initial_tool"
        
        tool_manager.tool_metadata["initial_tool"] = {
            "type": "initial",
            "code": "def initial_tool(): return 'test'",
            "file": "initial_tool.py",
            "function": "initial_tool"
        }
        
        # Execute tool with force_local=True
        result = await tool_manager._execute_tool_safely(
            tool_func=mock_tool,
            arguments={"value": "test"},
            task_description="Test task",
            force_local=True
        )
        
        # Should have executed locally
        assert result == "local_result"
        mock_tool.assert_called_once_with(value="test")
        
        # Should not have used Docker
        tool_manager.docker_executor.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_tool_directly(self, tool_manager: ToolManager):
        """Test direct tool execution functionality."""
        # Create a test tool
        mock_tool = Mock(return_value="direct_result")
        mock_tool.__name__ = "direct_test_tool"
        
        tool_manager.loaded_tools["direct_test_tool"] = mock_tool
        tool_manager.tool_metadata["direct_test_tool"] = {
            "type": "initial",
            "code": "def direct_test_tool(): return 'test'",
            "file": "direct_test_tool.py",
            "function": "direct_test_tool"
        }
        
        # Execute tool directly
        result = await tool_manager.execute_tool_directly(
            tool_name="direct_test_tool",
            arguments={"value": "test"},
            force_local=True
        )
        
        # Should return the result
        assert result == "direct_result"
        mock_tool.assert_called_once_with(value="test")

    @pytest.mark.asyncio
    async def test_execute_tool_directly_nonexistent(self, tool_manager: ToolManager):
        """Test executing a non-existent tool directly."""
        await tool_manager.initialize()
        
        # Should raise KeyError for non-existent tool
        with pytest.raises(KeyError, match="Tool 'nonexistent' not found"):
            await tool_manager.execute_tool_directly(
                tool_name="nonexistent",
                arguments={}
            )

    def test_list_tools(self, tool_manager: ToolManager):
        """Test listing all available tools."""
        # Add some mock tools
        mock_initial_tool = Mock()
        mock_initial_tool.__name__ = "initial_tool"
        mock_initial_tool.__doc__ = "An initial tool"
        
        mock_generated_tool = Mock()  
        mock_generated_tool.__name__ = "generated_tool"
        mock_generated_tool.__doc__ = "A generated tool"
        
        tool_manager.loaded_tools = {
            "initial_tool": mock_initial_tool,
            "generated_tool": mock_generated_tool
        }
        
        tool_manager.tool_metadata = {
            "initial_tool": {
                "type": "initial",
                "file": "initial_tool.py"
            },
            "generated_tool": {
                "type": "generated", 
                "file": "generated_tool.py"
            }
        }
        
        # List tools
        tools = tool_manager.list_tools()
        
        # Should return ToolInfo objects
        assert len(tools) == 2
        assert all(isinstance(tool, ToolInfo) for tool in tools)
        
        # Find tools by name
        initial_tool_info = next(t for t in tools if t.name == "initial_tool")
        generated_tool_info = next(t for t in tools if t.name == "generated_tool")
        
        # Verify tool info
        assert initial_tool_info.type == ToolType.INITIAL
        assert initial_tool_info.docstring == "An initial tool"
        
        assert generated_tool_info.type == ToolType.GENERATED
        assert generated_tool_info.docstring == "A generated tool"

    def test_get_existing_tools_context(self, tool_manager: ToolManager):
        """Test getting context about existing tools for Claude."""
        # Add mock tools
        mock_tool1 = Mock()
        mock_tool1.__name__ = "tool1"
        mock_tool1.__doc__ = "First test tool"
        
        mock_tool2 = Mock()
        mock_tool2.__name__ = "tool2"
        mock_tool2.__doc__ = "Second test tool"
        
        tool_manager.loaded_tools = {
            "tool1": mock_tool1,
            "tool2": mock_tool2
        }
        
        # Mock inspect.signature
        with patch("tool_smith_mcp.core.tool_manager.inspect.signature") as mock_signature:
            mock_signature.return_value = "(param: str) -> str"
            
            context = tool_manager._get_existing_tools_context()
            
            # Should contain information about both tools
            assert "tool1" in context
            assert "tool2" in context
            assert "First test tool" in context
            assert "Second test tool" in context
            assert "(param: str) -> str" in context

    def test_extract_function_name(self, tool_manager: ToolManager):
        """Test extracting function names from generated code."""
        # Test with valid Python code
        valid_code = '''def my_function(param: str) -> str:
    """A test function."""
    return param.upper()
'''
        
        name = tool_manager._extract_function_name(valid_code)
        assert name == "my_function"
        
        # Test with multiple functions (should get first one)
        multi_function_code = '''def first_function():
    pass

def second_function():
    pass
'''
        
        name = tool_manager._extract_function_name(multi_function_code)
        assert name == "first_function"
        
        # Test with invalid code (should generate hash-based name)
        invalid_code = "this is not valid python code"
        name = tool_manager._extract_function_name(invalid_code)
        assert name.startswith("tool_")
        assert len(name) == 13  # "tool_" + 8 character hash

    @pytest.mark.asyncio
    async def test_structure_arguments(self, tool_manager: ToolManager):
        """Test argument structuring for tool functions."""
        # Create mock tool function
        mock_tool = Mock()
        mock_tool.__doc__ = "Test function with parameters"
        
        # Set up Claude mock response
        expected_args = {"param1": "value1", "param2": 42}
        setup_claude_mock_responses(
            tool_manager.claude_client,
            structure_args_response=expected_args
        )
        
        # Mock inspect.signature
        with patch("tool_smith_mcp.core.tool_manager.inspect.signature") as mock_signature:
            mock_signature.return_value = "(param1: str, param2: int) -> str"
            
            # Structure arguments
            result = await tool_manager._structure_arguments(
                tool_func=mock_tool,
                task_description="Test task",
                arguments={"param1": "value1", "param2": 42}
            )
            
            # Should return structured arguments from Claude
            assert result == expected_args
            tool_manager.claude_client.structure_arguments.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_claude_api_failure(self, tool_manager: ToolManager):
        """Test error handling when Claude API fails."""
        # Set up no existing tools
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            []  # No existing tools
        )
        
        # Make Claude API fail
        tool_manager.claude_client.generate_tool.side_effect = Exception("Claude API error")
        
        await tool_manager.initialize()
        
        # Should propagate the Claude API error
        with pytest.raises(Exception, match="Claude API error"):
            await tool_manager.solve_task(
                task_description="This should fail",
                arguments={}
            )

    @pytest.mark.asyncio
    async def test_error_handling_docker_failure(self, tool_manager: ToolManager):
        """Test error handling when Docker execution fails."""
        # Set up Docker to fail
        tool_manager.docker_executor.execute_tool.side_effect = Exception("Docker error")
        
        # Create a generated tool that would use Docker
        mock_tool = Mock()
        mock_tool.__name__ = "failing_tool"
        
        tool_manager.tool_metadata["failing_tool"] = {
            "type": "generated",
            "code": "def failing_tool(): return 'test'",
            "file": "failing_tool.py",
            "function": "failing_tool"
        }
        
        # Should propagate Docker error
        with pytest.raises(Exception, match="Docker error"):
            await tool_manager._execute_tool_safely(
                tool_func=mock_tool,
                arguments={},
                task_description="Test task"
            )

    @pytest.mark.asyncio
    async def test_save_and_load_tool(self, tool_manager: ToolManager):
        """Test saving and loading a generated tool."""
        await tool_manager.initialize()
        
        # Generate tool code
        tool_code = create_sample_tool_code(
            "save_load_test_tool",
            "Tool for testing save and load",
            implementation='return f"saved_{value}"'
        )
        
        # Save and load tool
        result_func = await tool_manager._save_and_load_tool(
            tool_name="save_load_test_tool",
            tool_code=tool_code,
            description="Test tool description"
        )
        
        # Should have saved to file
        tool_file = tool_manager.tools_dir / "save_load_test_tool.py"
        assert tool_file.exists()
        assert tool_code in tool_file.read_text()
        
        # Should have loaded the function
        assert result_func is not None
        assert "save_load_test_tool" in tool_manager.loaded_tools
        
        # Should have stored metadata
        assert "save_load_test_tool" in tool_manager.tool_metadata
        metadata = tool_manager.tool_metadata["save_load_test_tool"]
        assert metadata["type"] == "generated"
        assert metadata["code"] == tool_code
        
        # Should have added to vector store
        tool_manager.vector_store.add_document.assert_called()


class TestToolManagerIntegrationScenarios:
    """Test realistic scenarios that combine multiple ToolManager features."""

    @pytest.mark.asyncio
    async def test_complete_task_workflow(self, tool_manager: ToolManager):
        """Test a complete task workflow from request to result."""
        # Set up realistic mock responses
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            [("math_calculator", 0.9, "Calculate math expressions", {})]
        )
        
        setup_claude_mock_responses(
            tool_manager.claude_client,
            structure_args_response={"expression": "2 + 3 * 4"}
        )
        
        # Create a realistic mock tool
        def mock_calculator(expression: str) -> float:
            """Calculate mathematical expressions."""
            return eval(expression)  # In real code, use safe evaluation
        
        mock_calculator.__name__ = "math_calculator"
        tool_manager.loaded_tools["math_calculator"] = mock_calculator
        tool_manager.tool_metadata["math_calculator"] = {
            "type": "initial",
            "code": "def math_calculator(expression): return eval(expression)",
            "file": "math_calculator.py",
            "function": "math_calculator"
        }
        
        # Execute complete workflow
        result = await run_tool_manager_test(
            tool_manager,
            task_description="Calculate the result of 2 + 3 * 4",
            arguments={"expression": "2 + 3 * 4"},
            expected_outcome="numerical result"
        )
        
        # Should get correct calculation result
        assert result == 14.0

    @pytest.mark.asyncio
    async def test_tool_reuse_optimization(self, tool_manager: ToolManager):
        """Test that tools are properly reused for similar tasks."""
        # Set up high similarity match
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            [("text_formatter", 0.95, "Format text strings", {})]
        )
        
        setup_claude_mock_responses(
            tool_manager.claude_client,
            structure_args_response={"text": "hello world"}
        )
        
        # Create mock formatter tool
        def mock_formatter(text: str) -> str:
            """Format text to uppercase."""
            return text.upper()
        
        mock_formatter.__name__ = "text_formatter"
        tool_manager.loaded_tools["text_formatter"] = mock_formatter
        tool_manager.tool_metadata["text_formatter"] = {
            "type": "initial",
            "code": "def text_formatter(text): return text.upper()",
            "file": "text_formatter.py",
            "function": "text_formatter"
        }
        
        await tool_manager.initialize()
        
        # Execute similar tasks
        result1 = await tool_manager.solve_task(
            task_description="Make text uppercase",
            arguments={"text": "hello world"}
        )
        
        result2 = await tool_manager.solve_task(
            task_description="Convert text to capital letters",
            arguments={"text": "hello world"}
        )
        
        # Should reuse same tool for both tasks
        assert result1 == "HELLO WORLD"
        assert result2 == "HELLO WORLD"
        
        # Should not have generated new tools
        tool_manager.claude_client.generate_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_generation(self, tool_manager: ToolManager):
        """Test fallback to tool generation when existing tools don't match."""
        # Set up low similarity matches only
        setup_vector_store_mock_responses(
            tool_manager.vector_store,
            [("unrelated_tool", 0.2, "Unrelated functionality", {})]
        )
        
        # Set up tool generation
        new_tool_code = create_sample_tool_code(
            "specialized_formatter",
            "Specialized text formatting tool",
            implementation='return value.title()'
        )
        
        setup_claude_mock_responses(
            tool_manager.claude_client,
            tool_generation_response=new_tool_code,
            structure_args_response={"value": "hello world"}
        )
        
        await tool_manager.initialize()
        
        # Request specialized functionality
        result = await tool_manager.solve_task(
            task_description="Convert text to title case",
            arguments={"value": "hello world"}
        )
        
        # Should have generated and used new tool
        assert "specialized_formatter" in tool_manager.loaded_tools
        
        # Should have added to vector store for future reuse
        tool_manager.vector_store.add_document.assert_called()
        
        # Should have saved to disk
        tool_file = tool_manager.tools_dir / "specialized_formatter.py"
        assert tool_file.exists()