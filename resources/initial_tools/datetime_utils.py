"""Date and time utilities tool."""

import datetime
from typing import Optional, Union, Dict, Any
import calendar


def datetime_utils(
    operation: str,
    date_input: Optional[Union[str, datetime.datetime]] = None,
    **kwargs: Any
) -> Union[str, int, datetime.datetime, Dict[str, Any], bool]:
    """Perform date and time operations.
    
    Args:
        operation: Type of operation ("now", "parse", "format", "add", "subtract", 
                  "diff", "weekday", "is_weekend", "age", "timestamp")
        date_input: Input date (string or datetime object)
        **kwargs: Operation-specific parameters
        
    Returns:
        Result based on operation type
        
    Raises:
        ValueError: For invalid operations or date formats
    """
    if operation == "now":
        format_str = kwargs.get('format')
        now = datetime.datetime.now()
        if format_str:
            return now.strftime(format_str)
        return now
    
    elif operation == "parse":
        if not date_input:
            raise ValueError("Parse operation requires date_input")
        
        format_str = kwargs.get('format', '%Y-%m-%d')
        if isinstance(date_input, str):
            # Try common formats if no format specified
            if format_str == '%Y-%m-%d':
                formats_to_try = [
                    '%Y-%m-%d',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ'
                ]
                for fmt in formats_to_try:
                    try:
                        return datetime.datetime.strptime(date_input, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unable to parse date: {date_input}")
            else:
                return datetime.datetime.strptime(date_input, format_str)
        return date_input
    
    elif operation == "format":
        if not date_input:
            raise ValueError("Format operation requires date_input")
        
        if isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        format_str = kwargs.get('format', '%Y-%m-%d')
        return date_obj.strftime(format_str)
    
    elif operation == "add":
        if not date_input:
            raise ValueError("Add operation requires date_input")
        
        if isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        days = kwargs.get('days', 0)
        hours = kwargs.get('hours', 0)
        minutes = kwargs.get('minutes', 0)
        seconds = kwargs.get('seconds', 0)
        weeks = kwargs.get('weeks', 0)
        
        delta = datetime.timedelta(
            days=days, hours=hours, minutes=minutes, 
            seconds=seconds, weeks=weeks
        )
        return date_obj + delta
    
    elif operation == "subtract":
        if not date_input:
            raise ValueError("Subtract operation requires date_input")
        
        if isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        days = kwargs.get('days', 0)
        hours = kwargs.get('hours', 0)
        minutes = kwargs.get('minutes', 0)
        seconds = kwargs.get('seconds', 0)
        weeks = kwargs.get('weeks', 0)
        
        delta = datetime.timedelta(
            days=days, hours=hours, minutes=minutes, 
            seconds=seconds, weeks=weeks
        )
        return date_obj - delta
    
    elif operation == "diff":
        if not date_input:
            raise ValueError("Diff operation requires date_input")
        
        other_date = kwargs.get('other')
        if not other_date:
            raise ValueError("Diff operation requires 'other' parameter")
        
        if isinstance(date_input, str):
            date_obj1 = datetime_utils("parse", date_input)
        else:
            date_obj1 = date_input
        
        if isinstance(other_date, str):
            date_obj2 = datetime_utils("parse", other_date)
        else:
            date_obj2 = other_date
        
        diff = abs((date_obj1 - date_obj2).total_seconds())
        unit = kwargs.get('unit', 'days')
        
        if unit == 'seconds':
            return int(diff)
        elif unit == 'minutes':
            return int(diff / 60)
        elif unit == 'hours':
            return int(diff / 3600)
        elif unit == 'days':
            return int(diff / 86400)
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    elif operation == "weekday":
        if not date_input:
            raise ValueError("Weekday operation requires date_input")
        
        if isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        return calendar.day_name[date_obj.weekday()]
    
    elif operation == "is_weekend":
        if not date_input:
            raise ValueError("Is_weekend operation requires date_input")
        
        if isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    elif operation == "age":
        if not date_input:
            raise ValueError("Age operation requires date_input")
        
        if isinstance(date_input, str):
            birth_date = datetime_utils("parse", date_input)
        else:
            birth_date = date_input
        
        today = datetime.datetime.now()
        age = today.year - birth_date.year
        
        # Adjust if birthday hasn't occurred this year
        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1
        
        return age
    
    elif operation == "timestamp":
        if not date_input:
            date_obj = datetime.datetime.now()
        elif isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        return int(date_obj.timestamp())
    
    elif operation == "from_timestamp":
        timestamp = kwargs.get('timestamp')
        if timestamp is None:
            raise ValueError("From_timestamp operation requires 'timestamp' parameter")
        
        return datetime.datetime.fromtimestamp(timestamp)
    
    elif operation == "info":
        if not date_input:
            date_obj = datetime.datetime.now()
        elif isinstance(date_input, str):
            date_obj = datetime_utils("parse", date_input)
        else:
            date_obj = date_input
        
        return {
            "date": date_obj.strftime('%Y-%m-%d'),
            "time": date_obj.strftime('%H:%M:%S'),
            "datetime": date_obj.strftime('%Y-%m-%d %H:%M:%S'),
            "weekday": calendar.day_name[date_obj.weekday()],
            "is_weekend": date_obj.weekday() >= 5,
            "day_of_year": date_obj.timetuple().tm_yday,
            "week_of_year": date_obj.isocalendar()[1],
            "timestamp": int(date_obj.timestamp())
        }
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")