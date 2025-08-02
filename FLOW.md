# Tool Smith MCP - Control Flow Documentation

This document describes the detailed flow of information and control when the MCP server's `solve_task` tool is called, outlining all possible decisions made until the tool returns control to the agent.

## Overview

Tool Smith MCP operates as a two-layer system where the top-level MCP tool (`solve_task`) orchestrates the execution of second-layer Python functions (initial tools and runtime-generated tools).

## Detailed Control Flow

### 1. Initial Request Reception

```
MCP Client -> MCP Server -> solve_task()
```

**Input Parameters:**
- `task_description` (string, required): Formal description of the task
- `arguments` (object, optional): Arguments required for the task  
- `expected_outcome` (string, optional): Expected outcome or return type

**Initial Processing:**
1. **Parameter Validation**: Validate required `task_description` parameter
2. **Logging**: Log incoming request with task description
3. **Configuration Loading**: Ensure configuration is loaded and accessible

**Decision Point 1**: Are parameters valid?
- ❌ **Invalid**: Return error response immediately
- ✅ **Valid**: Continue to semantic search phase

---

### 2. Semantic Search Phase

**Location**: `ToolManager.find_tool()`

**Process**:
1. **Vector Store Initialization**: Ensure ChromaDB vector store is available
2. **Embedding Generation**: Generate embedding for task description using sentence-transformers
3. **Similarity Search**: Query vector database for similar tool descriptions

**Configuration Parameters Used**:
- `tools.search_top_k`: Number of similar tools to retrieve (default: 3)
- `tools.similarity_threshold`: Minimum similarity score (default: 0.7)

**Decision Point 2**: Are similar tools found above threshold?

#### Path A: Existing Tool Found (similarity ≥ threshold)

**Process**:
1. **Tool Selection**: Select highest scoring tool above threshold
2. **Tool Loading**: Load Python function from filesystem
3. **Argument Preparation**: Structure arguments for the selected tool
4. **Logging**: Log tool selection and similarity score

**Decision Point 3**: Does tool exist on filesystem?
- ❌ **Missing**: Fall back to tool generation (Path B)
- ✅ **Available**: Continue to tool execution (Section 4)

**Decision Point 4**: Are tool arguments compatible?
- ❌ **Incompatible**: Attempt argument transformation or fall back to generation
- ✅ **Compatible**: Continue to tool execution (Section 4)

#### Path B: No Suitable Tool Found (similarity < threshold)

**Process**: Continue to tool generation (Section 3)

---

### 3. Tool Generation Phase

**Location**: `ToolManager.create_tool()` + `ClaudeClient`

**Process**:
1. **Claude API Preparation**: Initialize Claude client with configuration
2. **Prompt Construction**: Build tool generation prompt with:
   - Task description
   - Available initial tools for import/reuse in the new tool
   - Expected function signature
   - Code quality requirements

**Configuration Parameters Used**:
- `claude.model`: Claude model to use (default: claude-3-5-sonnet-20241022)
- `claude.max_tokens`: Maximum response tokens (default: 4000)
- `claude.temperature`: Generation temperature (default: 0.1)

**Decision Point 5**: Is Claude API available and configured?
- ❌ **Unavailable**: Return error response
- ✅ **Available**: Continue generation

**API Call Process**:
1. **Tool Generation Request**: Send prompt to Claude API
2. **Response Processing**: Extract generated Python function code
3. **Code Validation**: Basic syntax and structure validation

**Decision Point 6**: Is generated code valid?
- ❌ **Invalid**: Retry generation (up to configured limit) or return error
- ✅ **Valid**: Continue to tool persistence

**Tool Persistence**:
1. **Filename Generation**: Create unique filename for new tool
2. **File Writing**: Save Python function to `tools_dir`
3. **Vector Indexing**: Add tool description to vector database
4. **Metadata Storage**: Store tool source information (generated vs initial)

**Decision Point 7**: Does tool persistence succeed?
- ❌ **Failed**: Return error response
- ✅ **Success**: Continue to tool execution

---

### 4. Tool Execution Phase

**Location**: `ToolManager.execute_tool()`

**Process**:
1. **Dynamic Import**: Import the tool function (initial or generated)
2. **Argument Binding**: Bind provided arguments to function parameters
3. **Execution Context**: Set up execution environment

**Decision Point 8**: Does tool import succeed?
- ❌ **Import Error**: Log error and return failure response
- ✅ **Success**: Continue execution

**Decision Point 9**: Are function arguments valid?
- ❌ **Invalid**: Attempt argument coercion or return argument error
- ✅ **Valid**: Execute function

**Function Execution**:
1. **Secure Execution**: Execute function in controlled environment
2. **Result Capture**: Capture function return value
3. **Error Handling**: Catch and process any execution exceptions

**Decision Point 10**: Does function execution succeed?
- ❌ **Exception**: Log exception and return error response
- ✅ **Success**: Continue to result processing

---

### 5. Result Processing Phase

**Process**:
1. **Result Validation**: Ensure result matches expected outcome (if specified)
2. **Serialization**: Convert result to JSON-serializable format
3. **Logging**: Log successful execution and result summary

**Decision Point 11**: Is result serializable?
- ❌ **Not Serializable**: Convert to string representation or return error
- ✅ **Serializable**: Continue to response formatting

**Response Formatting**:
1. **Success Response**: Format successful result
2. **Metadata Addition**: Add execution metadata (tool used, execution time, etc.)
3. **Final Validation**: Ensure response conforms to MCP protocol

---

### 6. Response Return

**Final Process**:
1. **Response Packaging**: Package result according to MCP protocol
2. **Cleanup**: Clean up any temporary resources
3. **Metrics**: Update usage metrics and logs
4. **Return**: Return control to MCP client

## Error Handling Paths

### Configuration Errors
- **Missing Configuration**: Return configuration error
- **Invalid Claude API Key**: Return authentication error
- **Database Connection Issues**: Return storage error

### Tool Management Errors
- **Tool Loading Failures**: Attempt graceful degradation
- **Generation Failures**: Provide detailed error messages
- **Execution Timeouts**: Return timeout error with partial results

### Resource Management
- **Memory Limits**: Monitor and prevent excessive memory usage
- **Disk Space**: Check available space before tool persistence
- **Network Issues**: Handle Claude API connectivity problems

## Performance Considerations

### Caching Strategies
- **Vector Embeddings**: Cache embeddings for repeated task descriptions
- **Tool Compilation**: Cache compiled tool functions
- **Configuration**: Cache parsed configuration

### Optimization Points
- **Parallel Processing**: Consider parallel tool search and generation
- **Background Tasks**: Tool indexing and cleanup in background
- **Connection Pooling**: Reuse Claude API connections

## Configuration Impact on Flow

### `tools.similarity_threshold`
- **Higher Values** (0.8-1.0): More tool generation, fewer reused tools
- **Lower Values** (0.3-0.6): More tool reuse, potentially less accurate matches

### `tools.search_top_k`
- **Higher Values**: More comprehensive search, slower response
- **Lower Values**: Faster search, potentially missed matches

### `claude.temperature`
- **Higher Values** (0.3-1.0): More creative tool generation, less predictable
- **Lower Values** (0.0-0.2): More deterministic generation, consistent results

## Monitoring and Observability

### Key Metrics
- **Tool Reuse Rate**: Percentage of requests using existing tools
- **Generation Success Rate**: Percentage of successful tool generations
- **Execution Time**: Average time for complete request processing
- **Error Rates**: Breakdown of error types and frequencies

### Logging Points
- Request reception and parameter validation
- Tool search results and similarity scores
- Tool generation attempts and outcomes
- Tool execution results and performance
- Error conditions and recovery actions

This flow ensures reliable, efficient tool management while providing flexibility for dynamic tool creation based on user requirements.