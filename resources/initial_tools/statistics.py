"""Simple statistical calculations tool."""

from typing import Any, Dict, List, Union


def statistics(data: List[Union[int, float]], operation: str = "mean", **kwargs: Any) -> Union[float, Dict[str, float]]:
    """Calculate statistical measures for numeric data.
    
    Args:
        data: List of numeric values
        operation: Type of calculation ("mean", "median", "mode", "std", "summary")
        **kwargs: Additional parameters
        
    Returns:
        Statistical result as number or summary dict
        
    Structured Output:
        - summary: Dict containing statistical measures with fields:
          * count: number of data points
          * mean: arithmetic average
          * median: middle value
          * min: minimum value  
          * max: maximum value
          * std: standard deviation
          * variance: variance value
        
    Raises:
        ValueError: For invalid operations or empty data
        TypeError: For non-numeric data
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Validate numeric data
    numeric_data = []
    for item in data:
        if isinstance(item, (int, float)):
            numeric_data.append(float(item))
        else:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                raise TypeError(f"All data must be numeric, got: {type(item)}")
    
    if operation == "mean":
        return sum(numeric_data) / len(numeric_data)
    
    elif operation == "median":
        sorted_data = sorted(numeric_data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            return sorted_data[n//2]
    
    elif operation == "mode":
        # Find most frequent value
        from collections import Counter
        counts = Counter(numeric_data)
        max_count = max(counts.values())
        modes = [value for value, count in counts.items() if count == max_count]
        return modes[0] if len(modes) == 1 else float('nan')
    
    elif operation == "std":
        # Standard deviation
        mean_val = sum(numeric_data) / len(numeric_data)
        variance = sum((x - mean_val) ** 2 for x in numeric_data) / len(numeric_data)
        return variance ** 0.5
    
    elif operation == "summary":
        # Complete statistical summary
        sorted_data = sorted(numeric_data)
        n = len(sorted_data)
        mean_val = sum(numeric_data) / n
        
        # Median
        if n % 2 == 0:
            median_val = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median_val = sorted_data[n//2]
        
        # Variance and standard deviation
        variance = sum((x - mean_val) ** 2 for x in numeric_data) / n
        std_val = variance ** 0.5
        
        return {
            "count": n,
            "mean": mean_val,
            "median": median_val,
            "min": min(numeric_data),
            "max": max(numeric_data),
            "std": std_val,
            "variance": variance
        }
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")