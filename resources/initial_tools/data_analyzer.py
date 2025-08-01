"""Composite data analysis tool that combines multiple building blocks."""

import json
from typing import Any, Dict, List, Optional, Union

# Import the building block tools
from .data_processing import data_processing
from .calculate_math import calculate_math
from .datetime_utils import datetime_utils
from .file_operations import file_operations
from .format_text import format_text


def data_analyzer(
    operation: str,
    data: Any,
    **kwargs: Any
) -> Union[Dict[str, Any], List[Any], str, float]:
    """Comprehensive data analysis tool that combines multiple building blocks.
    
    This tool demonstrates how future tools would be created by combining
    existing building blocks to provide higher-level functionality.
    
    Args:
        operation: Type of analysis ("summary", "trends", "outliers", "report", 
                  "export", "import", "correlate")
        data: Input data to analyze
        **kwargs: Operation-specific parameters
        
    Returns:
        Analysis results in various formats
        
    Raises:
        ValueError: For invalid operations or parameters
    """
    if operation == "summary":
        """Generate comprehensive data summary using multiple tools."""
        if not isinstance(data, list):
            raise TypeError("Summary operation requires a list of data")
        
        # Use data_processing for basic aggregations
        total_count = data_processing("aggregate", data, function="count")
        
        # Calculate numeric statistics if data contains numbers
        numeric_data = []
        for item in data:
            if isinstance(item, dict):
                # Extract numeric values from dict
                for value in item.values():
                    if isinstance(value, (int, float)):
                        numeric_data.append(value)
            elif isinstance(value, (int, float)):
                numeric_data.append(item)
        
        summary = {
            "total_records": total_count,
            "data_types": list(set(type(item).__name__ for item in data)),
            "timestamp": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
        }
        
        if numeric_data:
            summary.update({
                "numeric_count": len(numeric_data),
                "sum": data_processing("aggregate", numeric_data, function="sum"),
                "average": data_processing("aggregate", numeric_data, function="avg"),
                "min": data_processing("aggregate", numeric_data, function="min"),
                "max": data_processing("aggregate", numeric_data, function="max"),
                "unique_values": len(data_processing("aggregate", numeric_data, function="unique"))
            })
        
        return summary
    
    elif operation == "trends":
        """Analyze trends in time-series data."""
        if not isinstance(data, list):
            raise TypeError("Trends operation requires a list of data")
        
        date_field = kwargs.get('date_field', 'date')
        value_field = kwargs.get('value_field', 'value')
        
        # Sort data by date using data_processing and datetime_utils
        if data and isinstance(data[0], dict) and date_field in data[0]:
            # Parse and sort by dates
            for item in data:
                if date_field in item:
                    # Convert date string to datetime for sorting
                    try:
                        parsed_date = datetime_utils("parse", item[date_field])
                        item['_parsed_date'] = parsed_date
                    except:
                        item['_parsed_date'] = item[date_field]
            
            # Sort by parsed date
            sorted_data = data_processing("sort", data, key="_parsed_date")
            
            # Calculate trend metrics
            values = [item.get(value_field, 0) for item in sorted_data if value_field in item]
            
            if len(values) >= 2:
                # Calculate simple trend
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                first_avg = data_processing("aggregate", first_half, function="avg")
                second_avg = data_processing("aggregate", second_half, function="avg")
                
                trend_direction = "increasing" if second_avg > first_avg else "decreasing" if second_avg < first_avg else "stable"
                trend_magnitude = abs(second_avg - first_avg) / first_avg * 100 if first_avg != 0 else 0
                
                return {
                    "trend_direction": trend_direction,
                    "trend_magnitude_percent": round(trend_magnitude, 2),
                    "first_period_avg": first_avg,
                    "second_period_avg": second_avg,
                    "total_data_points": len(values),
                    "analysis_date": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
                }
        
        return {"error": "Unable to analyze trends - insufficient or invalid data"}
    
    elif operation == "outliers":
        """Detect outliers in numeric data using statistical methods."""
        if not isinstance(data, list):
            raise TypeError("Outliers operation requires a list of data")
        
        # Extract numeric values
        numeric_values = []
        for item in data:
            if isinstance(item, (int, float)):
                numeric_values.append(item)
            elif isinstance(item, dict):
                value_field = kwargs.get('value_field', 'value')
                if value_field in item and isinstance(item[value_field], (int, float)):
                    numeric_values.append(item[value_field])
        
        if len(numeric_values) < 3:
            return {"error": "Insufficient numeric data for outlier detection"}
        
        # Calculate statistics using existing tools
        mean = data_processing("aggregate", numeric_values, function="avg")
        sorted_values = data_processing("sort", numeric_values)
        
        # Calculate standard deviation using math operations
        variance_sum = sum(calculate_math(f"({value} - {mean}) ** 2") 
                          for value in numeric_values)
        variance = variance_sum / len(numeric_values)
        std_dev = calculate_math(f"{variance} ** 0.5")
        
        # Identify outliers (values beyond 2 standard deviations)
        threshold = 2 * std_dev
        outliers = []
        
        for i, value in enumerate(numeric_values):
            deviation = abs(value - mean)
            if deviation > threshold:
                outliers.append({
                    "index": i,
                    "value": value,
                    "deviation": deviation,
                    "z_score": deviation / std_dev if std_dev > 0 else 0
                })
        
        return {
            "outliers": outliers,
            "outlier_count": len(outliers),
            "total_count": len(numeric_values),
            "outlier_percentage": round(len(outliers) / len(numeric_values) * 100, 2),
            "statistics": {
                "mean": mean,
                "std_dev": std_dev,
                "min": min(numeric_values),
                "max": max(numeric_values)
            }
        }
    
    elif operation == "report":
        """Generate a comprehensive analysis report."""
        # Combine multiple analysis operations
        summary = data_analyzer("summary", data)
        
        report_sections = {
            "metadata": {
                "report_title": kwargs.get('title', 'Data Analysis Report'),
                "generated_at": datetime_utils("now", format="%Y-%m-%d %H:%M:%S"),
                "data_source": kwargs.get('source', 'Unknown')
            },
            "summary": summary
        }
        
        # Add trends analysis if data supports it
        try:
            trends = data_analyzer("trends", data, **kwargs)
            if "error" not in trends:
                report_sections["trends"] = trends
        except:
            pass
        
        # Add outliers analysis for numeric data
        try:
            outliers = data_analyzer("outliers", data, **kwargs)
            if "error" not in outliers:
                report_sections["outliers"] = outliers
        except:
            pass
        
        # Format as readable text if requested
        if kwargs.get('format') == 'text':
            text_parts = []
            text_parts.append(f"# {report_sections['metadata']['report_title']}")
            text_parts.append(f"Generated: {report_sections['metadata']['generated_at']}")
            text_parts.append("")
            
            # Summary section
            text_parts.append("## Summary")
            summary = report_sections['summary']
            text_parts.append(f"Total Records: {summary.get('total_records', 'N/A')}")
            if 'average' in summary:
                text_parts.append(f"Average: {summary['average']}")
                text_parts.append(f"Min: {summary['min']}, Max: {summary['max']}")
            text_parts.append("")
            
            return format_text("\n".join(text_parts), "strip")
        
        return report_sections
    
    elif operation == "export":
        """Export analysis results to file."""
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("Export operation requires 'file_path' parameter")
        
        # Perform analysis first
        analysis_type = kwargs.get('analysis', 'summary')
        if analysis_type == 'report':
            analysis_result = data_analyzer("report", data, **kwargs)
        else:
            analysis_result = data_analyzer(analysis_type, data, **kwargs)
        
        # Export using file_operations
        file_operations("write", file_path, content=analysis_result)
        
        return {
            "exported": True,
            "file_path": file_path,
            "analysis_type": analysis_type,
            "timestamp": datetime_utils("now", format="%Y-%m-%d %H:%M:%S")
        }
    
    elif operation == "import":
        """Import data from file and perform analysis."""
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("Import operation requires 'file_path' parameter")
        
        # Import data using file_operations
        imported_data = file_operations("read", file_path)
        
        # If it's JSON string, parse it
        if isinstance(imported_data, str):
            try:
                imported_data = data_processing("parse", imported_data, format="json")
            except:
                # Try CSV format
                try:
                    imported_data = data_processing("parse", imported_data, format="csv")
                except:
                    raise ValueError("Unable to parse imported data as JSON or CSV")
        
        # Perform requested analysis
        analysis_type = kwargs.get('analysis', 'summary')
        analysis_result = data_analyzer(analysis_type, imported_data, **kwargs)
        
        return {
            "imported_records": len(imported_data) if isinstance(imported_data, list) else 1,
            "analysis": analysis_result,
            "source_file": file_path
        }
    
    elif operation == "correlate":
        """Analyze correlation between two data fields."""
        if not isinstance(data, list) or not data:
            raise TypeError("Correlate operation requires a non-empty list of data")
        
        field_x = kwargs.get('field_x')
        field_y = kwargs.get('field_y')
        
        if not field_x or not field_y:
            raise ValueError("Correlate operation requires 'field_x' and 'field_y' parameters")
        
        # Extract values for both fields
        x_values = []
        y_values = []
        
        for item in data:
            if isinstance(item, dict) and field_x in item and field_y in item:
                x_val = item[field_x]
                y_val = item[field_y]
                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                    x_values.append(x_val)
                    y_values.append(y_val)
        
        if len(x_values) < 2:
            return {"error": "Insufficient numeric data for correlation analysis"}
        
        # Calculate correlation coefficient using mathematical operations
        n = len(x_values)
        sum_x = data_processing("aggregate", x_values, function="sum")
        sum_y = data_processing("aggregate", y_values, function="sum")
        
        sum_xy = sum(calculate_math(f"{x} * {y}") for x, y in zip(x_values, y_values))
        sum_x2 = sum(calculate_math(f"{x} ** 2") for x in x_values)
        sum_y2 = sum(calculate_math(f"{y} ** 2") for y in y_values)
        
        # Pearson correlation coefficient
        numerator = calculate_math(f"{n} * {sum_xy} - {sum_x} * {sum_y}")
        denominator_x = calculate_math(f"({n} * {sum_x2} - {sum_x} ** 2) ** 0.5")
        denominator_y = calculate_math(f"({n} * {sum_y2} - {sum_y} ** 2) ** 0.5")
        
        if denominator_x == 0 or denominator_y == 0:
            correlation = 0
        else:
            correlation = calculate_math(f"{numerator} / ({denominator_x} * {denominator_y})")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "none"
        
        return {
            "correlation_coefficient": round(correlation, 4),
            "strength": strength,
            "direction": direction,
            "data_points": n,
            "field_x": field_x,
            "field_y": field_y,
            "x_stats": {
                "mean": data_processing("aggregate", x_values, function="avg"),
                "min": min(x_values),
                "max": max(x_values)
            },
            "y_stats": {
                "mean": data_processing("aggregate", y_values, function="avg"),
                "min": min(y_values),
                "max": max(y_values)
            }
        }
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")