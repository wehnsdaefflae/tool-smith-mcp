"""Tool manager for creating and managing second-layer tools."""

import ast
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

from ..utils.vector_store import VectorStore
from ..utils.claude_client import ClaudeClient

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages second-layer tools and their creation."""
    
    def __init__(
        self,
        tools_dir: Path,
        vector_store: VectorStore,
        claude_client: ClaudeClient,
        similarity_threshold: float = 0.7,
        initial_tools_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the tool manager.
        
        Args:
            tools_dir: Directory where runtime tools are stored
            vector_store: Vector store for tool indexing
            claude_client: Claude client for tool generation
            similarity_threshold: Minimum similarity for tool matching
            initial_tools_dir: Directory containing initial/built-in tools
        """
        self.tools_dir = tools_dir
        self.vector_store = vector_store
        self.claude_client = claude_client
        self.similarity_threshold = similarity_threshold
        self.initial_tools_dir = initial_tools_dir
        self.loaded_tools: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the tool manager by loading existing tools."""
        # Load initial/built-in tools first
        if self.initial_tools_dir and self.initial_tools_dir.exists():
            await self._load_initial_tools()
        
        # Load runtime tools
        await self._load_existing_tools()
    
    async def solve_task(
        self,
        task_description: str,
        arguments: Dict[str, Any],
        expected_outcome: Optional[str] = None,
    ) -> Any:
        """Solve a task using existing or newly created tools.
        
        Args:
            task_description: Description of the task to solve
            arguments: Arguments for the task
            expected_outcome: Expected outcome description
            
        Returns:
            Result of the task execution
        """
        logger.info(f"Solving task: {task_description}")
        
        # Search for existing tools
        similar_tools = await self.vector_store.search(
            query=task_description,
            top_k=3,
        )
        
        # Check if any tool is similar enough
        best_tool = None
        if similar_tools and similar_tools[0][1] >= self.similarity_threshold:
            tool_name = similar_tools[0][0]
            best_tool = self.loaded_tools.get(tool_name)
            logger.info(f"Found suitable tool: {tool_name} (similarity: {similar_tools[0][1]:.3f})")
        
        if best_tool is None:
            # Create a new tool
            logger.info("No suitable tool found, creating new tool")
            tool_name, tool_func = await self._create_new_tool(
                task_description, arguments, expected_outcome
            )
            best_tool = tool_func
        
        # Execute the tool
        try:
            # Use Claude to structure the arguments for the tool
            structured_args = await self._structure_arguments(
                tool_func=best_tool,
                task_description=task_description,
                arguments=arguments,
            )
            
            result = best_tool(**structured_args)
            logger.info(f"Task completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            raise
    
    async def _load_existing_tools(self) -> None:
        """Load existing tools from the tools directory."""
        logger.info("Loading existing tools")
        
        for tool_file in self.tools_dir.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(
                    tool_file.stem, tool_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find functions in the module
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if not name.startswith("_") and obj.__doc__:
                            self.loaded_tools[name] = obj
                            
                            # Add to vector store
                            await self.vector_store.add_document(
                                doc_id=name,
                                content=obj.__doc__,
                                metadata={"file": str(tool_file), "function": name},
                            )
                            
                logger.info(f"Loaded tools from {tool_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading tools from {tool_file}: {e}")
    
    async def _load_initial_tools(self) -> None:
        """Load initial/built-in tools from the initial tools directory."""
        logger.info("Loading initial tools")
        
        if not self.initial_tools_dir:
            return
        
        for tool_file in self.initial_tools_dir.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(
                    tool_file.stem, tool_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find functions in the module
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if not name.startswith("_") and obj.__doc__:
                            self.loaded_tools[name] = obj
                            
                            # Add to vector store
                            await self.vector_store.add_document(
                                doc_id=name,
                                content=obj.__doc__,
                                metadata={"file": str(tool_file), "function": name, "type": "initial"},
                            )
                            
                logger.info(f"Loaded initial tools from {tool_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading initial tools from {tool_file}: {e}")
    
    async def _create_new_tool(
        self,
        task_description: str,
        arguments: Dict[str, Any],
        expected_outcome: Optional[str] = None,
    ) -> Tuple[str, Any]:
        """Create a new tool using Claude.
        
        Args:
            task_description: Description of the task
            arguments: Available arguments
            expected_outcome: Expected outcome description
            
        Returns:
            Tuple of (tool_name, tool_function)
        """
        # Get context from existing tools
        existing_tools_context = self._get_existing_tools_context()
        
        # Generate tool using Claude
        tool_code = await self.claude_client.generate_tool(
            task_description=task_description,
            arguments=arguments,
            expected_outcome=expected_outcome,
            existing_tools_context=existing_tools_context,
        )
        
        # Extract tool name and function from generated code
        tool_name = self._extract_function_name(tool_code)
        
        # Save and load the tool
        tool_func = await self._save_and_load_tool(
            tool_name=tool_name,
            tool_code=tool_code,
            description=task_description,
        )
        
        return tool_name, tool_func
    
    def _get_existing_tools_context(self) -> str:
        """Get context about existing tools for Claude."""
        context_parts = []
        
        for tool_name, tool_func in self.loaded_tools.items():
            signature = inspect.signature(tool_func)
            docstring = tool_func.__doc__ or "No description"
            
            context_parts.append(
                f"Function: {tool_name}{signature}\n"
                f"Description: {docstring}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from generated code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        
        # Fallback: generate a hash-based name
        return f"tool_{hashlib.md5(code.encode()).hexdigest()[:8]}"
    
    async def _save_and_load_tool(
        self, tool_name: str, tool_code: str, description: str
    ) -> Any:
        """Save tool to file and load it."""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        # Write tool to file
        tool_file.write_text(tool_code)
        
        # Load the tool
        spec = importlib.util.spec_from_file_location(tool_name, tool_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the function
            tool_func = getattr(module, tool_name)
            self.loaded_tools[tool_name] = tool_func
            
            # Add to vector store
            await self.vector_store.add_document(
                doc_id=tool_name,
                content=description,
                metadata={"file": str(tool_file), "function": tool_name},
            )
            
            logger.info(f"Created and loaded new tool: {tool_name}")
            return tool_func
        
        raise RuntimeError(f"Failed to load tool: {tool_name}")
    
    async def _structure_arguments(
        self,
        tool_func: Any,
        task_description: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use Claude to structure arguments for the tool function."""
        signature = inspect.signature(tool_func)
        docstring = tool_func.__doc__ or "No description"
        
        return await self.claude_client.structure_arguments(
            function_signature=str(signature),
            function_docstring=docstring,
            task_description=task_description,
            available_arguments=arguments,
        )