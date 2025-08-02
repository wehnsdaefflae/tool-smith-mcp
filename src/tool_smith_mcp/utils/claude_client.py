"""Claude client for tool generation and argument structuring."""

import json
import logging
from typing import Any, Dict, Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for interacting with Claude API for tool generation."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Claude client with model: {model}")

    async def generate_tool(
        self,
        task_description: str,
        arguments: Dict[str, Any],
        expected_outcome: Optional[str] = None,
        existing_tools_context: str = "",
    ) -> str:
        """Generate a new tool function using Claude.

        Args:
            task_description: Description of the task to solve
            arguments: Available arguments
            expected_outcome: Expected outcome description
            existing_tools_context: Context about existing tools

        Returns:
            Generated Python function code
        """
        prompt = self._build_tool_generation_prompt(
            task_description=task_description,
            arguments=arguments,
            expected_outcome=expected_outcome,
            existing_tools_context=existing_tools_context,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            # Extract the generated code
            content = response.content[0].text if response.content else ""
            generated_code = self._extract_code_from_response(content)

            logger.info(f"Generated tool code for task: {task_description[:50]}...")
            return generated_code

        except Exception as e:
            logger.error(f"Error generating tool with Claude: {e}")
            raise

    async def structure_arguments(
        self,
        function_signature: str,
        function_docstring: str,
        task_description: str,
        available_arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Structure arguments to match a function signature using Claude.

        Args:
            function_signature: The function signature
            function_docstring: The function's docstring
            task_description: Description of the task
            available_arguments: Available arguments to map

        Returns:
            Structured arguments dictionary
        """
        prompt = self._build_argument_structuring_prompt(
            function_signature=function_signature,
            function_docstring=function_docstring,
            task_description=task_description,
            available_arguments=available_arguments,
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            # Extract the structured arguments
            content = response.content[0].text if response.content else ""
            structured_args = self._extract_json_from_response(content)

            logger.debug(f"Structured arguments for function: {structured_args}")
            return structured_args

        except Exception as e:
            logger.error(f"Error structuring arguments with Claude: {e}")
            raise

    def _build_tool_generation_prompt(
        self,
        task_description: str,
        arguments: Dict[str, Any],
        expected_outcome: Optional[str],
        existing_tools_context: str,
    ) -> str:
        """Build prompt for tool generation."""
        prompt_parts = [
            "You are a Python function generator that creates tools to solve specific tasks.",
            "Create a single Python function that solves the given task.",
            "",
            "Requirements:",
            "- The function must be complete and self-contained",
            "- Include proper type hints for all parameters and return value",
            "- Include a comprehensive docstring with Args and Returns sections",
            "- Use only standard library modules or the existing tools provided",
            "- Handle errors gracefully with appropriate exceptions",
            "- Follow Python best practices and PEP 8 style",
            "",
            f"Task to solve: {task_description}",
        ]

        if arguments:
            prompt_parts.extend(
                [
                    "",
                    f"Available arguments: {json.dumps(arguments, indent=2)}",
                ]
            )

        if expected_outcome:
            prompt_parts.extend(
                [
                    "",
                    f"Expected outcome: {expected_outcome}",
                ]
            )

        if existing_tools_context:
            prompt_parts.extend(
                [
                    "",
                    "Existing tools you can use (call them as regular functions):",
                    existing_tools_context,
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Return only the Python function code, no additional explanation.",
                "Make sure the function name is descriptive and follows snake_case convention.",
            ]
        )

        return "\n".join(prompt_parts)

    def _build_argument_structuring_prompt(
        self,
        function_signature: str,
        function_docstring: str,
        task_description: str,
        available_arguments: Dict[str, Any],
    ) -> str:
        """Build prompt for argument structuring."""
        return f"""You need to map available arguments to a function's parameters.

Function signature: {function_signature}
Function docstring: {function_docstring}

Task description: {task_description}
Available arguments: {json.dumps(available_arguments, indent=2)}

Map the available arguments to the function parameters based on:
1. Parameter names and types
2. Function docstring descriptions
3. Task context

Return a JSON object with the mapped arguments. Only include parameters that can be mapped.
If an argument cannot be mapped or a required parameter is missing, use reasonable defaults.

Return only the JSON object, no additional text."""

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from Claude's response."""
        # Remove markdown code blocks if present
        lines = response.strip().split("\n")

        # Find code block boundaries
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.strip().startswith("```python") or line.strip().startswith("```"):
                start_idx = i + 1
                break

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break

        # Extract the code
        code_lines = lines[start_idx:end_idx]
        code = "\n".join(code_lines).strip()

        # If no code blocks found, return the entire response
        if not code or "def " not in code:
            code = response.strip()

        return code

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response."""
        # Try to parse the entire response as JSON first
        if parsed_json := self._try_parse_full_response(response):
            return parsed_json

        # Look for JSON within the response
        return self._extract_json_from_lines(response)

    def _try_parse_full_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Try to parse the full response as JSON."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return None

    def _extract_json_from_lines(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response lines."""
        lines = response.strip().split("\n")
        json_start = self._find_json_start(lines)

        if json_start >= 0:
            json_end = self._find_json_end(lines, json_start)
            if json_end >= 0:
                return self._parse_json_block(lines, json_start, json_end)

        # Fallback: return empty dict
        logger.warning(f"Could not extract JSON from response: {response[:100]}...")
        return {}

    def _find_json_start(self, lines: list[str]) -> int:
        """Find the starting line of JSON block."""
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                return i
        return -1

    def _find_json_end(self, lines: list[str], start: int) -> int:
        """Find the ending line of JSON block."""
        brace_count = 0
        for i in range(start, len(lines)):
            for char in lines[i]:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return i
        return -1

    def _parse_json_block(
        self, lines: list[str], start: int, end: int
    ) -> Dict[str, Any]:
        """Parse JSON block from lines."""
        json_text = "\n".join(lines[start : end + 1])
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return {}
