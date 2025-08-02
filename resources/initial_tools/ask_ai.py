"""AI question answering tool for getting responses from Claude."""

import json
import os
from typing import Any, Dict, Union

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolParam, ToolChoiceToolParam


def ask_ai(question: str, output_schema: str = None) -> Union[str, Dict[str, Any]]:
    """Ask a question to Claude AI and get an answer.
    
    Args:
        question: The question or prompt to ask
        output_schema: Optional JSON schema string for structured output
        
    Returns:
        Either a text response or parsed JSON data matching the schema
        
    Structured Output:
        When output_schema is provided, returns a dictionary conforming to the provided 
        JSON schema. The schema should be a valid JSON Schema specification defining
        the expected structure, types, and required fields. The output structure
        depends entirely on the schema provided by the caller.
        
    Raises:
        ValueError: If API key is missing or API call fails
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is required")
    
    client = Anthropic(api_key=api_key)
    
    try:
        if output_schema:
            # Use tool calling for structured output (recommended by Anthropic)
            # Parse the provided JSON schema
            tool_schema = json.loads(output_schema)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                messages=[MessageParam(role="user", content=question)],
                tools=[
                    ToolParam(
                        name="provide_structured_response",
                        description="Provide a structured response",
                        input_schema=tool_schema
                    )
                ],
                tool_choice=ToolChoiceToolParam(type="tool", name="provide_structured_response")
            )
            
            # Extract the tool use result
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        tool_input = content_block.input
                        if isinstance(tool_input, dict):
                            return tool_input
                        return str(tool_input)
            
            raise ValueError("No tool use found in response")
            
        else:
            # Plain text response
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                messages=[MessageParam(role="user", content=question)]
            )
            
            return response.content[0].text
            
    except Exception as e:
        raise ValueError(f"AI request failed: {str(e)}")