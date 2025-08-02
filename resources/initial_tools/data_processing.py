"""Data processing and manipulation tool."""

import json
import csv
import io
from typing import Any, Dict, List, Union, Optional
from collections import Counter


def data_processing(
    operation: str,
    data: Any,
    **kwargs: Any
) -> Any:
    """Process and manipulate data in various formats.
    
    Args:
        operation: Type of operation ("filter", "sort", "group", "transform", "aggregate", 
                  "parse", "format", "validate", "merge", "split")
        data: Input data (list, dict, string, etc.)
        **kwargs: Operation-specific parameters
        
    Returns:
        Processed data
        
    Structured Output:
        - filter/sort/transform: List containing processed items
        - group: Dict where keys are group identifiers and values are lists of grouped items
        - aggregate: Number (count, sum, avg, min, max) or list (unique values)
        - parse: Dict (from JSON) or list of dicts (from CSV)
        - validate: Boolean indicating whether data conforms to schema
        - merge: Combined dict or list depending on input data types
        - split: List of string segments (from string) or list of sublists (from list)
        
    Raises:
        ValueError: For invalid operations or parameters
        TypeError: For incompatible data types
    """
    if operation == "filter":
        if not isinstance(data, list):
            raise TypeError("Filter operation requires a list")
        key = kwargs.get('key')
        value = kwargs.get('value')
        condition = kwargs.get('condition', 'equals')
        
        def matches(item):
            item_value = item.get(key) if isinstance(item, dict) else item
            if condition == 'equals':
                return item_value == value
            if condition == 'contains':
                return str(value).lower() in str(item_value).lower()
            if condition == 'greater':
                return item_value > value
            if condition == 'less':
                return item_value < value
            raise ValueError(f"Unknown condition: {condition}")
        
        return [item for item in data if matches(item)]
    
    if operation == "sort":
        if not isinstance(data, list):
            raise TypeError("Sort operation requires a list")
        key = kwargs.get('key')
        reverse = kwargs.get('reverse', False)
        
        if key:
            return sorted(data, key=lambda x: x.get(key) if isinstance(x, dict) else x, reverse=reverse)
        return sorted(data, reverse=reverse)
    
    if operation == "group":
        if not isinstance(data, list):
            raise TypeError("Group operation requires a list")
        key = kwargs.get('key')
        if not key:
            raise ValueError("Group operation requires a 'key' parameter")
        
        groups = {}
        for item in data:
            group_key = item.get(key) if isinstance(item, dict) else str(item)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups
    
    if operation == "transform":
        if not isinstance(data, list):
            raise TypeError("Transform operation requires a list")
        transform_func = kwargs.get('function')
        if not transform_func:
            raise ValueError("Transform operation requires a 'function' parameter")
        
        # Simple transformations
        if transform_func == 'uppercase':
            return [str(item).upper() for item in data]
        if transform_func == 'lowercase':
            return [str(item).lower() for item in data]
        if transform_func == 'length':
            return [len(str(item)) for item in data]
        if transform_func == 'reverse':
            return [str(item)[::-1] for item in data]
        raise ValueError(f"Unknown transform function: {transform_func}")
    
    if operation == "aggregate":
        if not isinstance(data, list):
            raise TypeError("Aggregate operation requires a list")
        func = kwargs.get('function', 'count')
        key = kwargs.get('key')
        
        # Extract values if key is specified
        values = []
        for item in data:
            if isinstance(item, dict) and key:
                values.append(item.get(key, 0))
            else:
                values.append(item)
        
        if func == 'count':
            return len(values)
        if func == 'sum':
            return sum(float(v) for v in values if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit())
        if func == 'avg':
            numeric_values = [float(v) for v in values if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit()]
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0
        if func == 'min':
            return min(values)
        if func == 'max':
            return max(values)
        if func == 'unique':
            return list(set(values))
        raise ValueError(f"Unknown aggregate function: {func}")
    
    if operation == "parse":
        format_type = kwargs.get('format', 'json')
        if format_type == 'json':
            return json.loads(data) if isinstance(data, str) else data
        if format_type == 'csv':
            if isinstance(data, str):
                reader = csv.DictReader(io.StringIO(data))
                return list(reader)
            raise TypeError("CSV parsing requires string input")
        raise ValueError(f"Unknown format: {format_type}")
    
    if operation == "format":
        format_type = kwargs.get('format', 'json')
        if format_type == 'json':
            return json.dumps(data, indent=kwargs.get('indent', 2))
        if format_type == 'csv':
            if isinstance(data, list) and data and isinstance(data[0], dict):
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()
            raise TypeError("CSV formatting requires list of dictionaries")
        raise ValueError(f"Unknown format: {format_type}")
    
    if operation == "validate":
        schema = kwargs.get('schema')
        if not schema:
            raise ValueError("Validate operation requires a 'schema' parameter")
        
        # Simple validation
        if isinstance(schema, dict):
            if not isinstance(data, dict):
                return False
            for key, expected_type in schema.items():
                if key not in data:
                    return False
                if expected_type == 'string' and not isinstance(data[key], str):
                    return False
                if expected_type == 'number' and not isinstance(data[key], (int, float)):
                    return False
                if expected_type == 'list' and not isinstance(data[key], list):
                    return False
        return True
    
    if operation == "merge":
        other_data = kwargs.get('other')
        if not other_data:
            raise ValueError("Merge operation requires 'other' parameter")
        
        if isinstance(data, dict) and isinstance(other_data, dict):
            result = data.copy()
            result.update(other_data)
            return result
        if isinstance(data, list) and isinstance(other_data, list):
            return data + other_data
        raise TypeError("Merge requires compatible data types")
    
    if operation == "split":
        if isinstance(data, str):
            delimiter = kwargs.get('delimiter', ',')
            return [item.strip() for item in data.split(delimiter)]
        if isinstance(data, list):
            chunk_size = kwargs.get('chunk_size', 1)
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        raise TypeError("Split operation requires string or list")
    
    raise ValueError(f"Unsupported operation: {operation}")