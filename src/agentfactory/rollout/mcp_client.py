from __future__ import annotations
import json
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from contextlib import AsyncExitStack
from loguru import logger
import anyio
from fastmcp import Client
from fastmcp.client.progress import ProgressHandler
from mcp.types import TextContent, ImageContent, ContentBlock
from .core.config import McpConfig

# Type aliases
ToolResponse = List[Dict[str, Any]]
StructuredContent = Dict[str, Any]


class MultiServerClient:
    """Base class for multiplexing calls to several MCP servers."""
    
    SYS_PYTHON_COMMAND = "sys_python"
    
    def __init__(self, config: McpConfig):
        self.server_configs = config.server_configs
        self.n_retries = config.connection_max_retries
        self.retry_interval = config.connection_retry_delay
        self.exponential_backoff = config.connection_exponential_backoff
        self.tool_name_separator = config.tool_name_separator
        
        self.name_to_client: dict[str, Client] = {}
        self.stack: AsyncExitStack | None = None

    def should_connect_server(self, server_name: str) -> bool:
        """Override this method to filter which servers to connect to."""
        return True

    async def __aenter__(self):
        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        try:
            for config in self.server_configs:
                server_name = next(iter(config['mcpServers']))
                if self.should_connect_server(server_name):
                    client = await self._create_client_with_retry(config)
                    if client is not None:
                        self.name_to_client[server_name] = client
            return self
        except Exception:
            await self.stack.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.stack is not None:
            return await self.stack.__aexit__(exc_type, exc_val, exc_tb)
        return False

    async def _create_client_with_retry(self, config) -> Client | None:
        """Create a single Client with automatic retries."""
        last_exception = None
        
        for attempt in range(self.n_retries):
            try:
                client_config = self._prepare_client_config(config)
                client = Client(client_config)
                client.transport.transport.keep_alive = False
                assert self.stack is not None, "Stack should be initialized"
                return await self.stack.enter_async_context(client)
            except Exception as e:
                logger.warning(f"Client {list(config['mcpServers'])[0]} creation failed, retrying... {attempt+1} times")
                last_exception = e
                if attempt < self.n_retries - 1:
                    wait_time = self.retry_interval
                    if self.exponential_backoff:
                        wait_time = self.retry_interval * (2 ** attempt)
                    await anyio.sleep(wait_time)
        
        raise last_exception or Exception("Failed to create client after retries")

    def _prepare_client_config(self, config):
        """Prepare client configuration, handling sys_python command."""
        config = config.copy()
        
        for server_name, server_config in config['mcpServers'].items():
            if server_config.get('command') == self.SYS_PYTHON_COMMAND:
                server_config = server_config.copy()
                server_config['command'] = sys.executable
                
                if server_config['args'] and server_config['args'][0].endswith('.py'):
                    server_config['args'][0] = str(Path(server_config['args'][0]).resolve())
                
                config['mcpServers'][server_name] = server_config
        
        return config

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        timeout: datetime.timedelta | float | int | None = None,
        progress_handler: ProgressHandler | None = None,
    ) -> Tuple[ToolResponse, StructuredContent]:
        """Call a tool and return raw result."""
        parsed_arguments = self._parse_string_arguments(arguments)
        server_name, tool_name = name.split(self.tool_name_separator, 1)
        
        tool_outputs = await self.name_to_client[server_name].call_tool(
            tool_name, parsed_arguments, timeout, progress_handler
        )
        content_list = tool_outputs.content
        structured_content = tool_outputs.structured_content or {}
        
        response = self._convert_tool_outputs_to_dict(content_list)
        return response, structured_content

    def _parse_string_arguments(self, arguments: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """Try to parse string arguments as JSON, keep as string if failed."""
        if not arguments:
            return arguments
        
        parsed = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                try:
                    parsed[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed[key] = value
            else:
                parsed[key] = value
        return parsed

    async def list_tools(self):
        """List all available tools from all servers."""
        tools = []
        for server_name, client in self.name_to_client.items():
            server_tools = await client.list_tools()
            for tool in server_tools:
                tool.name = f"{server_name}{self.tool_name_separator}{tool.name}"
            tools.extend(server_tools)
        return tools

    async def get_prompt(self, name: str, arguments: Dict[str, Any] | None = None):
        """Get prompt from server."""
        server_name, prompt_name = name.split(self.tool_name_separator, 1)
        client = self.name_to_client[server_name]
        prompt = await client.get_prompt(prompt_name, arguments)
        
        content_list = [msg.content for msg in prompt.messages]
        return self._convert_tool_outputs_to_dict(content_list)

    @staticmethod
    def _convert_tool_outputs_to_dict(tool_outputs: List[ContentBlock]) -> ToolResponse:
        """Convert tool outputs to dictionary format."""
        contents = []
        for output in tool_outputs:
            if isinstance(output, TextContent):
                contents.append({"type": "text", "text": output.text})
            elif isinstance(output, ImageContent):
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{output.mimeType};base64,{output.data}",
                    },
                })
            else:
                raise ValueError(f"Unsupported tool output type: {type(output)}")
        return contents


@dataclass
class _ToolCounter:
    current: int = 0
    limit: int | None = None
    
    def can_increment(self) -> bool:
        """Check if counter can be incremented without exceeding limit."""
        return self.limit is None or self.current < self.limit
    
    def increment(self) -> bool:
        """Increment counter and return True if still within quota."""
        if not self.can_increment():
            return False
        self.current += 1
        return True
    
    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.current)


class MultiServerClientWithLimit(MultiServerClient):
    """Multi-server client with tool usage limits."""
    
    def __init__(self, config: McpConfig):
        super().__init__(config)
        self.tool_limits = config.tool_limits
        
        # Initialize tool counters
        self._tool_counter = {
            tool_name: _ToolCounter(limit=limit)
            for tool_name, limit in self.tool_limits.items()
        }
        
        # Initialize available servers based on tool limits
        self._available_servers = {
            tool_name.split(self.tool_name_separator, 1)[0]
            for tool_name, counter in self._tool_counter.items()
            if counter.limit is None or counter.limit > 0
        }
        logger.debug(f"Created MultiServerClientWithLimit with available servers: {self._available_servers}")

    def should_connect_server(self, server_name: str) -> bool:
        """Only connect servers that have tools with configured limits."""
        return server_name in self._available_servers

    def is_tool_name_available(self, tool_name: str) -> bool:
        """Check if tool name is available."""
        return tool_name in self._tool_counter

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        timeout: datetime.timedelta | float | int | None = None,
        progress_handler: ProgressHandler | None = None,
        show_usage: bool = True,
    ) -> Tuple[ToolResponse, StructuredContent]:
        """Call a tool with usage limits checking."""
        # Check tool availability
        if name not in self._tool_counter:
            msg = f"Tool `{name}` is not available"
            return [{"type": "text", "text": msg}], {}
        
        # Check and update quota
        counter = self._tool_counter[name]
        if not counter.increment():
            msg = f"Exceeded maximum calls for `{name}`"
            return [{"type": "text", "text": msg}], {}
        
        # Call parent method
        response, structured_content = await super().call_tool(
            name, arguments, timeout, progress_handler
        )
        
        # Add usage info if requested
        if show_usage and counter.limit is not None:
            note = f"\nRemaining calls for `{name}`: {counter.remaining} / {counter.limit}"
            response.append({"type": "text", "text": note})
        
        return response, structured_content

    async def list_tools(self):
        """List all available tools from all servers with limit filtering."""
        tools = []
        for server_name, client in self.name_to_client.items():
            server_tools = await client.list_tools()
            for tool in server_tools:
                tool.name = f"{server_name}{self.tool_name_separator}{tool.name}"
            
            # Filter tools that have counters
            server_tools = [t for t in server_tools if t.name in self._tool_counter]
            tools.extend(server_tools)
        
        return tools

    async def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get definitions for all tools including usage information."""
        definitions = []
        for tool in await self.list_tools():
            counter = self._tool_counter[tool.name]
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "usage": {
                    "current": counter.current,
                    "max_calls": counter.limit or "unlimited"
                },
            })
        return definitions