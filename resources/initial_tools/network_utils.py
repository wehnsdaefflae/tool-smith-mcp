"""Network and HTTP utilities tool."""

import urllib.request
import urllib.parse
import urllib.error
import json
import re
from typing import Any, Dict, Optional, Union


def network_utils(
    operation: str,
    url: Optional[str] = None,
    **kwargs: Any
) -> Union[str, Dict[str, Any], bool]:
    """Perform network and HTTP operations.
    
    Args:
        operation: Type of operation ("get", "post", "validate_url", "parse_url", 
                  "encode_url", "decode_url", "check_status")
        url: Target URL for operations
        **kwargs: Operation-specific parameters
        
    Returns:
        Operation result
        
    Raises:
        ValueError: For invalid operations or parameters
        urllib.error.URLError: For HTTP errors
    """
    if operation == "get":
        if not url:
            raise ValueError("GET operation requires URL")
        
        headers = kwargs.get('headers', {})
        timeout = kwargs.get('timeout', 30)
        
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode('utf-8')
                
                # Try to parse as JSON if content type suggests it
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                return content
        except urllib.error.HTTPError as e:
            raise urllib.error.URLError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"URL Error: {e.reason}")
    
    elif operation == "post":
        if not url:
            raise ValueError("POST operation requires URL")
        
        data = kwargs.get('data', {})
        headers = kwargs.get('headers', {'Content-Type': 'application/json'})
        timeout = kwargs.get('timeout', 30)
        
        # Encode data
        if isinstance(data, dict):
            if headers.get('Content-Type') == 'application/json':
                data_encoded = json.dumps(data).encode('utf-8')
            else:
                data_encoded = urllib.parse.urlencode(data).encode('utf-8')
        else:
            data_encoded = str(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data_encoded, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read().decode('utf-8')
                
                # Try to parse as JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
        except urllib.error.HTTPError as e:
            raise urllib.error.URLError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"URL Error: {e.reason}")
    
    elif operation == "validate_url":
        if not url:
            raise ValueError("Validate_url operation requires URL")
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    
    elif operation == "parse_url":
        if not url:
            raise ValueError("Parse_url operation requires URL")
        
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "params": query_params
        }
    
    elif operation == "encode_url":
        text = kwargs.get('text')
        if not text:
            raise ValueError("Encode_url operation requires 'text' parameter")
        
        safe = kwargs.get('safe', '')
        return urllib.parse.quote(text, safe=safe)
    
    elif operation == "decode_url":
        text = kwargs.get('text')
        if not text:
            raise ValueError("Decode_url operation requires 'text' parameter")
        
        return urllib.parse.unquote(text)
    
    elif operation == "check_status":
        if not url:
            raise ValueError("Check_status operation requires URL")
        
        timeout = kwargs.get('timeout', 10)
        
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return {
                    "status_code": response.code,
                    "status_message": response.msg,
                    "headers": dict(response.headers),
                    "url": response.url,
                    "accessible": True
                }
        except urllib.error.HTTPError as e:
            return {
                "status_code": e.code,
                "status_message": e.reason,
                "accessible": False,
                "error": f"HTTP {e.code}: {e.reason}"
            }
        except urllib.error.URLError as e:
            return {
                "status_code": None,
                "status_message": None,
                "accessible": False,
                "error": f"URL Error: {e.reason}"
            }
    
    elif operation == "build_url":
        base_url = kwargs.get('base_url')
        if not base_url:
            raise ValueError("Build_url operation requires 'base_url' parameter")
        
        params = kwargs.get('params', {})
        path = kwargs.get('path', '')
        
        # Combine base URL and path
        if path:
            if not base_url.endswith('/') and not path.startswith('/'):
                full_url = f"{base_url}/{path}"
            elif base_url.endswith('/') and path.startswith('/'):
                full_url = f"{base_url}{path[1:]}"
            else:
                full_url = f"{base_url}{path}"
        else:
            full_url = base_url
        
        # Add query parameters
        if params:
            query_string = urllib.parse.urlencode(params)
            separator = '&' if '?' in full_url else '?'
            full_url = f"{full_url}{separator}{query_string}"
        
        return full_url
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")