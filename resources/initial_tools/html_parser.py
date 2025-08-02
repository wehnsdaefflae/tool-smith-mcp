"""Simple HTML parsing and extraction tool."""

import re
from typing import Any, Dict, List, Union


def html_parser(html: str, operation: str = "text", **kwargs: Any) -> Union[str, List[str], Dict[str, Any]]:
    """Parse and extract data from HTML content.
    
    Args:
        html: HTML content string to parse
        operation: Type of operation ("text", "links", "title", "meta", "extract")
        **kwargs: Operation-specific parameters
        
    Returns:
        Extracted content based on operation
        
    Structured Output:
        - links: List of link strings or dicts with url and text fields
        - meta: Dict containing title, description, and keywords fields
        - extract: List of strings matching the provided pattern
        
    Raises:
        ValueError: For invalid operations or parameters
    """
    if operation == "text":
        # Remove HTML tags and clean up text
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    if operation == "links":
        # Extract links
        pattern = r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>([^<]*)</a>'
        matches = re.findall(pattern, html, re.IGNORECASE)
        
        include_text = kwargs.get('include_text', False)
        if include_text:
            return [{"url": url, "text": text.strip()} for url, text in matches]
        return [url for url, text in matches]
    
    if operation == "title":
        # Extract page title
        match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    if operation == "meta":
        # Extract meta information
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\'][^>]*>', html, re.IGNORECASE)
        keywords_match = re.search(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\'][^>]*>', html, re.IGNORECASE)
        
        return {
            "title": title_match.group(1).strip() if title_match else "",
            "description": desc_match.group(1).strip() if desc_match else "",
            "keywords": keywords_match.group(1).strip() if keywords_match else ""
        }
    
    if operation == "extract":
        # Extract content using custom pattern
        pattern = kwargs.get('pattern')
        if not pattern:
            raise ValueError("Extract operation requires 'pattern' parameter")
        
        matches = re.findall(pattern, html, re.IGNORECASE)
        
        # Handle tuple results from groups
        if matches and isinstance(matches[0], tuple):
            return [' '.join(match) if isinstance(match, tuple) else match for match in matches]
        return matches

    raise ValueError(f"Unsupported operation: {operation}")