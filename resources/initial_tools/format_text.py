"""Text formatting and manipulation tool."""

from typing import Any


def format_text(text: str, operation: str = "upper", **kwargs: Any) -> str:
    """Format text according to specified operation.
    
    Args:
        text: Input text to format
        operation: Type of formatting ("upper", "lower", "title", "strip", "replace")
        **kwargs: Additional arguments for specific operations
        
    Returns:
        Formatted text
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "title":
        return text.title()
    elif operation == "strip":
        return text.strip()
    elif operation == "replace":
        old = kwargs.get("old", "")
        new = kwargs.get("new", "")
        return text.replace(old, new)
    else:
        raise ValueError(f"Unsupported operation: {operation}")