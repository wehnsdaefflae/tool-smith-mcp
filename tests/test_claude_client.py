"""Tests for the ClaudeClient class."""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from tool_smith_mcp.utils.claude_client import ClaudeClient


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """Create a mock Anthropic client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_content = Mock()
    mock_content.text = "def test_function():\n    return 'test'"
    mock_response.content = [mock_content]
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def claude_client(mock_anthropic_client: Mock) -> ClaudeClient:
    """Create a ClaudeClient with mocked Anthropic client."""
    with patch('tool_smith_mcp.utils.claude_client.Anthropic', return_value=mock_anthropic_client):
        return ClaudeClient("test_api_key")


@pytest.mark.asyncio
async def test_generate_tool(claude_client: ClaudeClient, mock_anthropic_client: Mock) -> None:
    """Test tool generation."""
    task_description = "Calculate the sum of two numbers"
    arguments = {"a": 5, "b": 3}
    expected_outcome = "Return the sum as an integer"
    
    result = await claude_client.generate_tool(
        task_description=task_description,
        arguments=arguments,
        expected_outcome=expected_outcome,
    )
    
    # Verify the mock was called
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args
    
    assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
    assert call_args[1]["max_tokens"] == 4000
    assert call_args[1]["temperature"] == 0.1
    assert len(call_args[1]["messages"]) == 1
    
    # Check that the prompt contains expected elements
    prompt = call_args[1]["messages"][0]["content"]
    assert task_description in prompt
    assert json.dumps(arguments, indent=2) in prompt
    assert expected_outcome in prompt
    
    # Check return value
    assert result == "def test_function():\n    return 'test'"


@pytest.mark.asyncio
async def test_structure_arguments(claude_client: ClaudeClient, mock_anthropic_client: Mock) -> None:
    """Test argument structuring."""
    # Mock response with JSON
    mock_response = Mock()
    mock_content = Mock()
    mock_content.text = '{"param1": "value1", "param2": 42}'
    mock_response.content = [mock_content]
    mock_anthropic_client.messages.create.return_value = mock_response
    
    function_signature = "def test_func(param1: str, param2: int) -> str"
    function_docstring = "Test function docstring"
    task_description = "Test task"
    available_arguments = {"input_text": "value1", "number": 42}
    
    result = await claude_client.structure_arguments(
        function_signature=function_signature,
        function_docstring=function_docstring,
        task_description=task_description,
        available_arguments=available_arguments,
    )
    
    # Verify the mock was called
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args
    
    assert call_args[1]["temperature"] == 0.0
    
    # Check that the prompt contains expected elements
    prompt = call_args[1]["messages"][0]["content"]
    assert function_signature in prompt
    assert function_docstring in prompt
    assert task_description in prompt
    assert json.dumps(available_arguments, indent=2) in prompt
    
    # Check return value
    expected_result = {"param1": "value1", "param2": 42}
    assert result == expected_result


def test_extract_code_from_response(claude_client: ClaudeClient) -> None:
    """Test code extraction from Claude response."""
    # Test with markdown code blocks
    response_with_blocks = """Here's the function:

```python
def add_numbers(a: int, b: int) -> int:
    return a + b
```

This function adds two numbers."""
    
    result = claude_client._extract_code_from_response(response_with_blocks)
    expected = "def add_numbers(a: int, b: int) -> int:\n    return a + b"
    assert result == expected
    
    # Test without code blocks
    response_without_blocks = "def multiply(x, y):\n    return x * y"
    result = claude_client._extract_code_from_response(response_without_blocks)
    assert result == response_without_blocks


def test_extract_json_from_response(claude_client: ClaudeClient) -> None:
    """Test JSON extraction from Claude response."""
    # Test with valid JSON
    response_with_json = '{"key": "value", "number": 123}'
    result = claude_client._extract_json_from_response(response_with_json)
    assert result == {"key": "value", "number": 123}
    
    # Test with JSON in text
    response_with_text = """Here's the mapping:
    
    {"param1": "mapped_value", "param2": 42}
    
    This should work."""
    
    result = claude_client._extract_json_from_response(response_with_text)
    assert result == {"param1": "mapped_value", "param2": 42}
    
    # Test with invalid JSON
    response_invalid = "Not valid JSON at all"
    result = claude_client._extract_json_from_response(response_invalid)
    assert result == {}  # Should return empty dict as fallback


@pytest.mark.asyncio
async def test_api_error_handling(claude_client: ClaudeClient, mock_anthropic_client: Mock) -> None:
    """Test error handling when API calls fail."""
    # Mock API error
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")
    
    with pytest.raises(Exception, match="API Error"):
        await claude_client.generate_tool("test task", {})
    
    with pytest.raises(Exception, match="API Error"):
        await claude_client.structure_arguments("sig", "doc", "task", {})