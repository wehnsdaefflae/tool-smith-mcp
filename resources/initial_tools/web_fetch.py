"""Simple web content fetching tool."""

from typing import Any, Dict, Union
from .network_utils import network_utils


def web_fetch(url: str, **kwargs: Any) -> Union[str, Dict[str, Any]]:
    """Fetch content from a web URL.
    
    Args:
        url: URL to fetch content from
        **kwargs: Additional parameters (headers, timeout, format)
        
    Returns:
        Web content as string or parsed data
        
    Raises:
        ValueError: For invalid URLs or fetch errors
    """
    # Validate URL
    if not network_utils("validate_url", url):
        raise ValueError(f"Invalid URL: {url}")
    
    # Set default headers
    headers = kwargs.get('headers', {
        'User-Agent': 'Mozilla/5.0 (compatible; ToolSmithMCP/1.0)'
    })
    
    # Fetch content
    timeout = kwargs.get('timeout', 30)
    content = network_utils("get", url, headers=headers, timeout=timeout)
    
    # Return based on requested format
    output_format = kwargs.get('format', 'text')
    
    if output_format == 'json' and isinstance(content, dict):
        return content
    elif output_format == 'text':
        return str(content)
    else:
        return content