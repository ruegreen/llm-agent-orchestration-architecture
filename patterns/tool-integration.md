# Pattern: Tool Integration

## Built-in Tool Registry

Built-in tools provide core capabilities without external dependencies. Each tool is a decorated function with schema validation and configuration-driven availability.

```python
# Tools are conditionally registered based on settings
tools = []

if settings.enable_web_search:
    tools.append(web_search_tool)

if settings.enable_file_operations:
    tools.append(read_file_tool)
    tools.append(write_file_tool)
    tools.append(list_files_tool)

if settings.enable_code_execution:
    tools.append(run_shell_command_tool)

# Always available (no security implications)
tools.append(calculator_tool)
tools.append(get_system_info_tool)
```

### Tool Definition Pattern

```python
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    results = search_engine.search(query, max_results=max_results)
    return format_results(results)
```

## MCP Client Integration

Enterprise tools are discovered dynamically at startup via the Model Context Protocol. The MCP client connects to configured servers, discovers available tools, and converts them to LangChain-compatible `StructuredTool` instances.

### Connection Management

```python
class MCPManager:
    """Manages connections to multiple MCP servers."""

    async def connect_all(self, server_configs: list[MCPServerConfig]):
        """Connect to all configured MCP servers."""
        for config in server_configs:
            client = MCPHttpClient(
                url=config.url,
                auth_token=config.token,
                name=config.name
            )
            await client.initialize()  # Handshake + session
            self.clients[config.name] = client

    async def discover_tools(self) -> list[StructuredTool]:
        """Discover all tools from all connected servers."""
        all_tools = []
        for name, client in self.clients.items():
            tools = await client.list_tools()  # JSON-RPC: tools/list
            for tool_def in tools:
                structured = convert_to_langchain_tool(tool_def, client)
                all_tools.append(structured)
        return all_tools
```

### MCP HTTP/SSE Protocol

```python
class MCPHttpClient:
    """JSON-RPC 2.0 over HTTP with SSE session management."""

    async def initialize(self):
        """Establish MCP session."""
        response = await self._rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {}
        })
        self.session_id = response.headers.get("mcp-session-id")

    async def list_tools(self) -> list[dict]:
        """Discover available tools."""
        result = await self._rpc("tools/list", {})
        return result["tools"]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        result = await self._rpc("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return result["content"][0]["text"]

    async def _rpc(self, method: str, params: dict) -> dict:
        """Send JSON-RPC 2.0 request with session tracking."""
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = {
            "jsonrpc": "2.0",
            "id": next_id(),
            "method": method,
            "params": params
        }
        response = await http_post(self.url, json=payload, headers=headers)
        return response.json()["result"]
```

### MCP-to-LangChain Conversion

```python
def convert_to_langchain_tool(tool_def: dict, client: MCPHttpClient) -> StructuredTool:
    """Convert an MCP tool definition to a LangChain StructuredTool."""

    async def execute(**kwargs):
        return await client.call_tool(tool_def["name"], kwargs)

    return StructuredTool(
        name=tool_def["name"],
        description=tool_def["description"],
        func=execute,
        args_schema=json_schema_to_pydantic(tool_def["inputSchema"])
    )
```

## Configuration Modes

MCP server configuration supports two modes — simple (shared auth) and advanced (per-server auth):

```bash
# Simple mode: comma-separated URLs with optional shared token
MCP_SERVERS=http://localhost:3001,http://localhost:3002
MCP_AUTH_TOKEN=shared-bearer-token

# Advanced mode: JSON array with per-server configuration
MCP_SERVERS_CONFIG='[
  {"url": "http://itsm-mcp:3001", "token": "itsm-key", "name": "itsm"},
  {"url": "http://crm-mcp:3002", "token": "crm-key", "name": "crm"}
]'
```

The configuration parser auto-detects which mode is active based on which environment variables are set.
