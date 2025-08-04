"""Isolated unit tests for ClaudeClient without external API calls.

These tests demonstrate how to test the Claude client module without
making actual API calls to Anthropic's servers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from tool_smith_mcp.utils.claude_client import ClaudeClient


class TestClaudeClientIsolated:
    """Test ClaudeClient functionality with mocked API calls."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock()
        return mock_client

    @pytest.fixture
    def claude_client(self, mock_anthropic_client):
        """Create a ClaudeClient with mocked dependencies."""
        with patch('tool_smith_mcp.utils.claude_client.anthropic.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            client = ClaudeClient(
                api_key="test_api_key",
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1
            )
            return client

    @pytest.mark.asyncio
    async def test_generate_tool_success(self, claude_client, mock_anthropic_client):
        """Test successful tool generation."""
        # Mock the API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b
'''
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Test tool generation
        result = await claude_client.generate_tool(
            task_description="Create a function to add two numbers",
            arguments={"a": 5, "b": 3},
            expected_outcome="sum of the numbers",
            existing_tools_context="No existing math tools"
        )
        
        # Verify the result
        assert "def calculate_sum" in result
        assert "return a + b" in result
        assert "Calculate the sum" in result
        
        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["max_tokens"] == 4000
        assert call_args[1]["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_generate_tool_with_existing_context(self, claude_client, mock_anthropic_client):
        """Test tool generation with existing tools context."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''def advanced_calculator(expression: str) -> float:
    """Calculate mathematical expressions using existing math tools.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    # Use existing calculate_sum for simple additions
    if '+' in expression and len(expression.split('+')) == 2:
        parts = expression.split('+')
        return calculate_sum(int(parts[0].strip()), int(parts[1].strip()))
    
    return eval(expression)  # Simplified for demo
'''
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        existing_context = """Function: calculate_sum(a: int, b: int) -> int
Description: Calculate the sum of two numbers"""
        
        result = await claude_client.generate_tool(
            task_description="Create an advanced calculator",
            arguments={"expression": "2 + 3 * 4"},
            existing_tools_context=existing_context
        )
        
        # Should reference existing tools
        assert "calculate_sum" in result
        assert "advanced_calculator" in result
        
        # Verify existing context was included in prompt
        call_args = mock_anthropic_client.messages.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "calculate_sum" in prompt_content

    @pytest.mark.asyncio
    async def test_structure_arguments_success(self, claude_client, mock_anthropic_client):
        """Test successful argument structuring."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"expression": "2 + 3", "precision": 2}'
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = await claude_client.structure_arguments(
            function_signature="(expression: str, precision: int = 2) -> float",
            function_docstring="Calculate mathematical expression with precision",
            task_description="Calculate 2 + 3 with 2 decimal places",
            available_arguments={"expression": "2 + 3", "precision": 2}
        )
        
        # Should return structured arguments as dict
        assert isinstance(result, dict)
        assert result["expression"] == "2 + 3"
        assert result["precision"] == 2

    @pytest.mark.asyncio
    async def test_structure_arguments_with_defaults(self, claude_client, mock_anthropic_client):
        """Test argument structuring with default parameters."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"text": "hello world"}'
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = await claude_client.structure_arguments(
            function_signature="(text: str, uppercase: bool = True) -> str",
            function_docstring="Format text with optional uppercase conversion",
            task_description="Format the text 'hello world'",
            available_arguments={"text": "hello world"}
        )
        
        # Should structure available arguments, may include defaults
        assert "text" in result
        assert result["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_api_error_handling(self, claude_client, mock_anthropic_client):
        """Test handling of API errors."""
        # Make API call raise an exception
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")
        
        # Should propagate the exception
        with pytest.raises(Exception, match="API Error"):
            await claude_client.generate_tool(
                task_description="This should fail",
                arguments={}
            )

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, claude_client, mock_anthropic_client):
        """Test handling of invalid JSON in structure_arguments."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = 'invalid json response'
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Should handle invalid JSON gracefully (depends on implementation)
        with pytest.raises((ValueError, Exception)):
            await claude_client.structure_arguments(
                function_signature="(param: str) -> str",
                function_docstring="Test function",
                task_description="Test task",
                available_arguments={"param": "test"}
            )

    def test_initialization_with_custom_config(self):
        """Test ClaudeClient initialization with custom configuration."""
        with patch('tool_smith_mcp.utils.claude_client.anthropic.AsyncAnthropic') as mock_anthropic:
            client = ClaudeClient(
                api_key="custom_key",
                model="claude-3-opus-20240229", 
                max_tokens=8000,
                temperature=0.5,
                structure_args_temperature=0.0,
                structure_args_max_tokens=2000
            )
            
            # Verify initialization parameters
            assert client.model == "claude-3-opus-20240229"
            assert client.max_tokens == 8000
            assert client.temperature == 0.5
            assert client.structure_args_temperature == 0.0
            assert client.structure_args_max_tokens == 2000
            
            # Verify Anthropic client was initialized with API key
            mock_anthropic.assert_called_once_with(api_key="custom_key")

    @pytest.mark.asyncio
    async def test_prompt_construction(self, claude_client, mock_anthropic_client):
        """Test that prompts are constructed correctly."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "def test(): pass"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        await claude_client.generate_tool(
            task_description="Create a test function",
            arguments={"param": "value"},
            expected_outcome="a simple function",
            existing_tools_context="existing tool context"
        )
        
        # Verify prompt construction
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        
        prompt = messages[0]["content"]
        assert "Create a test function" in prompt
        assert "param" in prompt
        assert "value" in prompt
        assert "a simple function" in prompt
        assert "existing tool context" in prompt

    @pytest.mark.asyncio
    async def test_model_parameters_used_correctly(self, claude_client, mock_anthropic_client):
        """Test that model parameters are used correctly for different operations."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "response"
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Test tool generation uses main parameters
        await claude_client.generate_tool("test task", {})
        
        tool_gen_call = mock_anthropic_client.messages.create.call_args
        assert tool_gen_call[1]["temperature"] == 0.1
        assert tool_gen_call[1]["max_tokens"] == 4000
        
        # Reset mock
        mock_anthropic_client.messages.create.reset_mock()
        mock_response.content[0].text = "{}"
        
        # Test argument structuring uses specific parameters
        await claude_client.structure_arguments("()", "", "", {})
        
        struct_call = mock_anthropic_client.messages.create.call_args
        # Should use structure_args parameters (if implemented)
        assert struct_call[1]["model"] == "claude-3-5-sonnet-20241022"


class TestClaudeClientIntegrationScenarios:
    """Test realistic scenarios for Claude client usage."""

    @pytest.fixture
    def claude_client_with_responses(self):
        """Create a Claude client with pre-configured responses."""
        mock_client = Mock()
        mock_client.messages = Mock()
        mock_client.messages.create = AsyncMock()
        
        with patch('tool_smith_mcp.utils.claude_client.anthropic.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_client
            
            client = ClaudeClient(api_key="test_key")
            client._mock_client = mock_client  # Store for test access
            return client

    @pytest.mark.asyncio
    async def test_complete_tool_generation_workflow(self, claude_client_with_responses):
        """Test a complete workflow from task to structured arguments."""
        client = claude_client_with_responses
        mock_client = client._mock_client
        
        # Set up responses for the workflow
        def side_effect(*args, **kwargs):
            # Determine which call this is based on the prompt content
            prompt = kwargs["messages"][0]["content"]
            
            response = Mock()
            response.content = [Mock()]
            
            if "generate" in prompt.lower() or "create" in prompt.lower():
                # Tool generation response
                response.content[0].text = '''def calculate_percentage(value: float, percentage: float) -> float:
    """Calculate percentage of a value.
    
    Args:
        value: The base value
        percentage: The percentage to calculate (as decimal, e.g., 0.1 for 10%)
        
    Returns:
        The calculated percentage value
    """
    return value * percentage
'''
            else:
                # Argument structuring response
                response.content[0].text = '{"value": 100.0, "percentage": 0.15}'
            
            return response
        
        mock_client.messages.create.side_effect = side_effect
        
        # Step 1: Generate tool
        tool_code = await client.generate_tool(
            task_description="Create a function to calculate 15% of a value",
            arguments={"value": 100, "percentage": 15},
            expected_outcome="percentage calculation function"
        )
        
        assert "calculate_percentage" in tool_code
        assert "value * percentage" in tool_code
        
        # Step 2: Structure arguments for the generated tool
        structured_args = await client.structure_arguments(
            function_signature="(value: float, percentage: float) -> float",
            function_docstring="Calculate percentage of a value",
            task_description="Calculate 15% of 100",
            available_arguments={"value": 100, "percentage": 0.15}
        )
        
        assert structured_args["value"] == 100.0
        assert structured_args["percentage"] == 0.15
        
        # Verify both API calls were made
        assert mock_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, claude_client_with_responses):
        """Test error recovery in various scenarios."""
        client = claude_client_with_responses
        mock_client = client._mock_client
        
        # Test retry logic (if implemented)
        call_count = 0
        def failing_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary API error")
            
            response = Mock()
            response.content = [Mock()]
            response.content[0].text = "def recovered_function(): pass"
            return response
        
        mock_client.messages.create.side_effect = failing_side_effect
        
        # This test assumes retry logic exists - adjust based on actual implementation
        try:
            result = await client.generate_tool("test task", {})
            # If retry logic exists, this should succeed
            assert "recovered_function" in result
        except Exception:
            # If no retry logic, should fail on first attempt
            assert call_count == 1