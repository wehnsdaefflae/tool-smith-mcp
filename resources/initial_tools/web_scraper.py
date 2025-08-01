"""Composite web scraping and content extraction tool."""

import re
import json
from typing import Any, Dict, List, Optional, Union

# Import the building block tools
from .network_utils import network_utils
from .format_text import format_text
from .data_processing import data_processing
from .file_operations import file_operations
from .datetime_utils import datetime_utils
from .encoding_utils import encoding_utils


def web_scraper(
    operation: str,
    url: Optional[str] = None,
    **kwargs: Any
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Comprehensive web scraping tool that combines multiple building blocks.
    
    This tool demonstrates how future tools would be created by combining
    existing building blocks to provide sophisticated web scraping capabilities.
    
    Args:
        operation: Type of operation ("fetch", "extract_links", "extract_data", 
                  "monitor", "download", "sitemap", "batch_fetch")
        url: Target URL for operations
        **kwargs: Operation-specific parameters
        
    Returns:
        Scraped content or extracted data
        
    Raises:
        ValueError: For invalid operations or parameters
    """
    if operation == "fetch":
        """Fetch and clean web page content."""
        if not url:
            raise ValueError("Fetch operation requires URL")
        
        # Validate URL first
        if not network_utils("validate_url", url):
            raise ValueError(f"Invalid URL: {url}")
        
        # Fetch content using network_utils
        headers = kwargs.get('headers', {
            'User-Agent': 'Mozilla/5.0 (compatible; ToolSmithMCP/1.0)'
        })
        
        content = network_utils("get", url, headers=headers, timeout=kwargs.get('timeout', 30))
        
        # Clean and process the content
        if kwargs.get('clean', True):
            # Remove HTML tags using text formatting
            cleaned_content = re.sub(r'<[^>]+>', ' ', content)
            # Clean up whitespace
            cleaned_content = format_text(cleaned_content, "strip")
            # Replace multiple spaces with single space
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            content = cleaned_content
        
        # Extract metadata
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "No title found"
        
        result = {
            "url": url,
            "title": title,
            "content": content,
            "content_length": len(content),
            "fetch_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
        
        # Add content hash for caching/comparison
        result["content_hash"] = encoding_utils("hash", content, algorithm="sha256")
        
        return result
    
    elif operation == "extract_links":
        """Extract all links from a web page."""
        if not url:
            raise ValueError("Extract_links operation requires URL")
        
        # Fetch page content first
        page_data = web_scraper("fetch", url, clean=False, **kwargs)
        content = page_data["content"]
        
        # Extract links using regex
        link_pattern = r'<a[^>]+href\s*=\s*["\']([^"\']+)["\'][^>]*>([^<]*)</a>'
        matches = re.findall(link_pattern, content, re.IGNORECASE)
        
        base_url = kwargs.get('base_url', url)
        parsed_base = network_utils("parse_url", base_url)
        base_domain = f"{parsed_base['scheme']}://{parsed_base['netloc']}"
        
        links = []
        for href, text in matches:
            # Clean link text
            link_text = format_text(text.strip(), "strip")
            
            # Handle relative URLs
            if href.startswith('http'):
                full_url = href
            elif href.startswith('//'):
                full_url = f"{parsed_base['scheme']}:{href}"
            elif href.startswith('/'):
                full_url = f"{base_domain}{href}"
            else:
                full_url = f"{base_url.rstrip('/')}/{href}"
            
            # Validate the constructed URL
            if network_utils("validate_url", full_url):
                links.append({
                    "url": full_url,
                    "text": link_text,
                    "href": href,
                    "type": "absolute" if href.startswith('http') else "relative"
                })
        
        # Remove duplicates and filter if requested
        unique_links = data_processing("aggregate", links, function="unique") if links else []
        
        # Filter by domain if requested
        if kwargs.get('same_domain_only', False):
            domain_filter = parsed_base['netloc']
            filtered_links = []
            for link in unique_links:
                link_domain = network_utils("parse_url", link["url"])["netloc"]
                if link_domain == domain_filter:
                    filtered_links.append(link)
            unique_links = filtered_links
        
        return {
            "source_url": url,
            "links_found": len(unique_links),
            "links": unique_links,
            "extraction_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
        }
    
    elif operation == "extract_data":
        """Extract structured data from web pages using patterns."""
        if not url:
            raise ValueError("Extract_data operation requires URL")
        
        # Fetch page content
        page_data = web_scraper("fetch", url, clean=False, **kwargs)
        content = page_data["content"]
        
        patterns = kwargs.get('patterns', {})
        if not patterns:
            # Default patterns for common data
            patterns = {
                "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "phones": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                "prices": r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
            }
        
        extracted_data = {}
        
        for data_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            # Clean and deduplicate matches
            if matches:
                if isinstance(matches[0], tuple):
                    # For patterns with groups, join them
                    cleaned_matches = [' '.join(match) for match in matches]
                else:
                    cleaned_matches = matches
                
                # Remove duplicates using data processing
                unique_matches = data_processing("aggregate", cleaned_matches, function="unique")
                extracted_data[data_type] = unique_matches
            else:
                extracted_data[data_type] = []
        
        # Extract metadata
        meta_patterns = {
            "title": r'<title[^>]*>([^<]+)</title>',
            "description": r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\'][^>]*>',
            "keywords": r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\'][^>]*>'
        }
        
        metadata = {}
        for key, pattern in meta_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            metadata[key] = match.group(1).strip() if match else None
        
        return {
            "source_url": url,
            "metadata": metadata,
            "extracted_data": extracted_data,
            "extraction_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S"),
            "total_matches": sum(len(matches) for matches in extracted_data.values())
        }
    
    elif operation == "monitor":
        """Monitor a web page for changes."""
        if not url:
            raise ValueError("Monitor operation requires URL")
        
        # Get current content
        current_data = web_scraper("fetch", url, **kwargs)
        current_hash = current_data["content_hash"]
        
        # Check if we have previous data to compare
        cache_file = kwargs.get('cache_file', f"./monitor_{encoding_utils('hash', url, algorithm='md5')}.json")
        
        try:
            # Try to load previous data
            previous_data = file_operations("read", cache_file)
            if isinstance(previous_data, str):
                previous_data = data_processing("parse", previous_data, format="json")
            
            # Compare hashes
            previous_hash = previous_data.get("content_hash")
            has_changed = current_hash != previous_hash
            
            if has_changed:
                # Calculate time since last change
                last_check = datetime_utils("parse", previous_data.get("fetch_time", ""))
                time_diff = datetime_utils("diff", current_data["fetch_time"], other=last_check, unit="minutes")
            else:
                time_diff = 0
            
        except:
            # No previous data
            previous_data = None
            has_changed = True
            time_diff = 0
        
        # Save current data for future comparisons
        file_operations("write", cache_file, content=current_data)
        
        result = {
            "url": url,
            "has_changed": has_changed,
            "current_hash": current_hash,
            "check_time": current_data["fetch_time"],
            "content_length": current_data["content_length"]
        }
        
        if previous_data:
            result.update({
                "previous_hash": previous_data.get("content_hash"),
                "last_check": previous_data.get("fetch_time"),
                "minutes_since_change": time_diff
            })
        
        return result
    
    elif operation == "download":
        """Download web content to file."""
        if not url:
            raise ValueError("Download operation requires URL")
        
        file_path = kwargs.get('file_path')
        if not file_path:
            # Generate filename from URL
            parsed_url = network_utils("parse_url", url)
            filename = parsed_url['path'].split('/')[-1] or 'index.html'
            file_path = f"./downloads/{filename}"
        
        # Fetch content
        content_data = web_scraper("fetch", url, **kwargs)
        
        # Save to file
        file_operations("write", file_path, content=content_data["content"])
        
        return {
            "url": url,
            "file_path": file_path,
            "downloaded_size": len(content_data["content"]),
            "download_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S"),
            "content_hash": content_data["content_hash"]
        }
    
    elif operation == "batch_fetch":
        """Fetch multiple URLs in batch."""
        urls = kwargs.get('urls', [])
        if not urls:
            raise ValueError("Batch_fetch operation requires 'urls' parameter")
        
        results = []
        for target_url in urls:
            try:
                result = web_scraper("fetch", target_url, **kwargs)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "url": target_url,
                    "status": "error",
                    "error": str(e),
                    "fetch_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
                })
        
        # Generate summary using data processing
        successful = data_processing("filter", results, key="status", value="success")
        failed = data_processing("filter", results, key="status", value="error")
        
        return {
            "total_urls": len(urls),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / len(urls) * 100, 2),
            "results": results,
            "batch_time": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
        }
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")