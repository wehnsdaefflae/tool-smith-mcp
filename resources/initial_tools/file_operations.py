"""File and directory operations tool."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def file_operations(
    operation: str,
    path: str,
    content: Optional[str] = None,
    encoding: str = "utf-8",
    **kwargs: Any
) -> Union[str, bool, List[str], Dict[str, Any]]:
    """Perform file and directory operations safely.
    
    Args:
        operation: Type of operation ("read", "write", "exists", "list", "mkdir", "delete", "size", "info")
        path: File or directory path
        content: Content to write (for write operations)
        encoding: Text encoding (default: utf-8)
        **kwargs: Additional operation-specific arguments
        
    Returns:
        Operation result (content, boolean, list, or dict)
        
    Raises:
        ValueError: For invalid operations
        OSError: For file system errors
    """
    path_obj = Path(path)
    
    if operation == "read":
        if not path_obj.exists():
            raise OSError(f"File not found: {path}")
        if path_obj.suffix.lower() == '.json':
            with open(path_obj, 'r', encoding=encoding) as f:
                return json.load(f)
        else:
            with open(path_obj, 'r', encoding=encoding) as f:
                return f.read()
    
    elif operation == "write":
        if content is None:
            raise ValueError("Content is required for write operation")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.suffix.lower() == '.json' and isinstance(content, (dict, list)):
            with open(path_obj, 'w', encoding=encoding) as f:
                json.dump(content, f, indent=kwargs.get('indent', 2))
        else:
            with open(path_obj, 'w', encoding=encoding) as f:
                f.write(str(content))
        return True
    
    elif operation == "exists":
        return path_obj.exists()
    
    elif operation == "list":
        if not path_obj.exists():
            raise OSError(f"Directory not found: {path}")
        if not path_obj.is_dir():
            raise OSError(f"Not a directory: {path}")
        pattern = kwargs.get('pattern', '*')
        return [str(p.name) for p in path_obj.glob(pattern)]
    
    elif operation == "mkdir":
        path_obj.mkdir(parents=kwargs.get('parents', True), exist_ok=kwargs.get('exist_ok', True))
        return True
    
    elif operation == "delete":
        if not path_obj.exists():
            return False
        if path_obj.is_file():
            path_obj.unlink()
        elif path_obj.is_dir():
            if kwargs.get('recursive', False):
                import shutil
                shutil.rmtree(path_obj)
            else:
                path_obj.rmdir()
        return True
    
    elif operation == "size":
        if not path_obj.exists():
            raise OSError(f"Path not found: {path}")
        if path_obj.is_file():
            return path_obj.stat().st_size
        else:
            # Directory size
            total = 0
            for file_path in path_obj.rglob('*'):
                if file_path.is_file():
                    total += file_path.stat().st_size
            return total
    
    elif operation == "info":
        if not path_obj.exists():
            raise OSError(f"Path not found: {path}")
        stat = path_obj.stat()
        return {
            "name": path_obj.name,
            "path": str(path_obj.absolute()),
            "size": stat.st_size,
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:]
        }
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")