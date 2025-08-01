"""Prompt execution tool with structured output support."""

import json
from typing import Any, Dict, Optional, Union


def prompt_executor(
    prompt: str,
    output_schema: Optional[str] = None,
    **kwargs: Any
) -> Union[str, Dict[str, Any]]:
    """Execute a natural language prompt using the tool LLM (Claude) with optional structured output.
    
    This tool acts as a direct interface to the Claude API, allowing any prompt to be executed
    with optional structured response formatting using Pydantic model schemas.
    
    Args:
        prompt: The natural language prompt to execute
        output_schema: Optional JSON schema string for structured output (Pydantic model schema)
        **kwargs: Additional parameters (temperature, max_tokens, model, etc.)
        
    Returns:
        Either plain text response or structured data matching the provided schema
        
    Raises:
        ValueError: For invalid parameters or API errors
    """
    # Get Claude API key
    import os
    api_key = os.getenv("CLAUDE_API_KEY") or kwargs.get('claude_api_key')
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable is required or pass claude_api_key parameter")
    
    # Import network_utils (must be done inside function for initial tools)
    from .network_utils import network_utils
    
    # Build the final prompt
    final_prompt = prompt
    
    if output_schema:
        try:
            # Validate that output_schema is valid JSON
            schema = json.loads(output_schema)
            final_prompt += f"""

IMPORTANT: You must return your response as valid JSON that conforms to this exact schema:
{json.dumps(schema, indent=2)}

Return ONLY the JSON object, no additional text or explanation.
Ensure all required fields are included and types match the schema."""
        except json.JSONDecodeError:
            raise ValueError("output_schema must be valid JSON schema")
    
    # Prepare API request
    api_data = {
        "model": kwargs.get('model', 'claude-3-5-sonnet-20241022'),
        "max_tokens": kwargs.get('max_tokens', 4000),
        "temperature": kwargs.get('temperature', 0.1),
        "messages": [
            {
                "role": "user", 
                "content": final_prompt
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    try:
        # Make the API call using network_utils
        response_data = network_utils(
            operation="post",
            url="https://api.anthropic.com/v1/messages",
            data=api_data,
            headers=headers,
            timeout=kwargs.get('timeout', 60)
        )
        
        # Extract the response content
        if isinstance(response_data, dict) and 'content' in response_data and response_data['content']:
            content = response_data['content'][0].get('text', '')
        else:
            raise ValueError("No content in API response")
        
        # If output_schema was specified, try to parse as JSON
        if output_schema:
            try:
                # Clean the response (remove any markdown formatting)
                cleaned_content = content.strip()
                
                if cleaned_content.startswith('```json'):
                    # Extract content between ```json and ```
                    start_marker = '```json'
                    end_marker = '```'
                    start_idx = cleaned_content.find(start_marker) + len(start_marker)
                    end_idx = cleaned_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        cleaned_content = cleaned_content[start_idx:end_idx].strip()
                elif cleaned_content.startswith('```'):
                    # Extract content between ``` and ```
                    parts = cleaned_content.split('```')
                    if len(parts) >= 3:
                        cleaned_content = parts[1].strip()
                
                # Parse JSON
                return json.loads(cleaned_content)
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse structured response as JSON: {e}\nResponse: {content}")
        
        # Return plain text response
        return content
        
    except Exception as e:
        if "HTTP" in str(e) or "URL" in str(e):
            raise ValueError(f"Claude API error: {str(e)}")
        else:
            raise ValueError(f"Unexpected error: {str(e)}")


# Helper function to create Pydantic model schemas
def create_pydantic_schema(model_class) -> str:
    """Create a JSON schema string from a Pydantic model class.
    
    Args:
        model_class: Pydantic model class
        
    Returns:
        JSON schema string
    """
    try:
        # Get the JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        return json.dumps(schema, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to create schema from model: {e}")


# Example usage functions for common structured outputs
def extract_entities(text: str, **kwargs) -> Dict[str, Any]:
    """Extract named entities from text using natural language processing.
    
    Args:
        text: Input text to analyze
        **kwargs: Additional parameters for the API call
        
    Returns:
        Dictionary with extracted entities
    """
    schema = {
        "type": "object",
        "properties": {
            "persons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of people mentioned"
            },
            "organizations": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "Organizations mentioned"
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Places and locations mentioned"
            },
            "dates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dates mentioned in the text"
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of the text"
            }
        },
        "required": ["persons", "organizations", "locations", "dates", "summary"]
    }
    
    return prompt_executor(
        prompt=f"Extract named entities from this text: {text}",
        output_schema=json.dumps(schema),
        **kwargs
    )


def analyze_sentiment(text: str, **kwargs) -> Dict[str, Any]:
    """Analyze sentiment of text with structured output.
    
    Args:
        text: Text to analyze
        **kwargs: Additional parameters for the API call
        
    Returns:
        Sentiment analysis results
    """
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Overall sentiment"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence score between 0 and 1"
            },
            "emotions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific emotions detected"
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"}, 
                "description": "Important phrases that indicate sentiment"
            }
        },
        "required": ["sentiment", "confidence", "emotions", "key_phrases"]
    }
    
    return prompt_executor(
        prompt=f"Analyze the sentiment of this text: {text}",
        output_schema=json.dumps(schema),
        **kwargs
    )