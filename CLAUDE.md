# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tool Smith MCP is an MCP (Model Context Protocol) server that dynamically creates and manages tools using LLM assistance. The server provides a single `solve_task` tool that can either use existing second-layer tools or create new ones on-demand using Claude API.

## Key Architecture

The system has two layers of tools:
1. **Top-level MCP tool**: `solve_task` - the main interface
2. **Second-layer tools**: Python functions that can be initial (built-in) or runtime-generated

### Core Components

- **ToolManager**: Manages second-layer tools, handles creation and execution
- **VectorStore**: Uses ChromaDB and sentence-transformers for semantic search of tool descriptions
- **ClaudeClient**: Interfaces with Claude API for tool generation and argument structuring
- **MCP Server**: Provides the `solve_task` tool interface
- **Config System**: TOML-based configuration management

### Tool Creation Flow

1. Receive task via `solve_task` 
2. Perform semantic similarity search over existing tool descriptions
3. If suitable tool found (similarity > threshold): use it
4. If no suitable tool: use Claude to generate new Python function
5. New tools are saved to filesystem and indexed in vector database

## Configuration

The server uses `tool-smith.toml` for configuration. Key settings include:

- **Claude API**: Model selection, temperature, token limits
- **Tools**: Similarity threshold, search parameters, directories
- **Vector Store**: Database path, collection settings, embedding model
- **Logging**: Level and format configuration

## Development Commands

```bash
# Install project and dependencies
pip install -e ".[dev]"

# Run the server (with configuration)
tool-smith-mcp

# Run tests
pytest

# Run tests with coverage
pytest --cov=tool_smith_mcp --cov-report=term-missing

# Code formatting
black src/tool_smith_mcp tests
isort src/tool_smith_mcp tests

# Type checking
mypy src/tool_smith_mcp

# Linting
ruff check src/tool_smith_mcp tests

# Run all quality checks
black src/tool_smith_mcp tests && isort src/tool_smith_mcp tests && mypy src/tool_smith_mcp && ruff check src/tool_smith_mcp tests && pytest
```

## Environment Setup

The server requires a Claude API key:

```bash
export CLAUDE_API_KEY="your_api_key_here"
```

## Project Structure (src/ layout)

```
src/
└── tool_smith_mcp/
    ├── core/
    │   ├── server.py          # Main MCP server implementation
    │   └── tool_manager.py    # Tool creation and management
    ├── utils/
    │   ├── vector_store.py    # Vector database for tool indexing
    │   ├── claude_client.py   # Claude API client
    │   └── config.py          # Configuration management
    └── __init__.py

resources/
└── initial_tools/             # Built-in initial tools
    ├── calculate_math.py      # Mathematical calculations
    └── format_text.py         # Text formatting

tests/                         # Comprehensive test suite
├── test_server.py
├── test_tool_manager.py
├── test_vector_store.py
├── test_claude_client.py
└── test_config.py

tool-smith.toml               # Configuration file
```

## Dependencies

- **mcp**: MCP protocol implementation
- **anthropic**: Claude API client
- **chromadb**: Vector database
- **sentence-transformers**: Text embeddings
- **pydantic**: Data validation
- **toml**: Configuration file parsing

## Tool Storage

- **Initial tools**: Stored in `resources/initial_tools/` as part of the package
- **Runtime tools**: Stored in `./tool-smith-mcp/tools/` (configurable)
- **Vector index**: ChromaDB database for semantic search
- **Metadata**: Tool source tracking (initial vs generated)