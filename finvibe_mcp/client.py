"""FinVibe synchronous in-process MCP client.

Bypasses async transport entirely by invoking tool callables directly
through FastMCP's ToolManager.  Safe in Jupyter / Kaggle kernels where
the event loop is already owned by the runtime.

Import:
    from finvibe_mcp import SyncMCPClient
    from finvibe_mcp import mcp_server
    client = SyncMCPClient(mcp_server)
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP


class SyncMCPClient:
    """Synchronous in-process client for a FastMCP server instance.

    Resolves each tool by name through the server's internal ToolManager
    and invokes its underlying callable directly — no asyncio, no subprocess,
    no inter-process transport.

    Args:
        server: An initialised FastMCP instance with tools already registered.

    Raises:
        AttributeError: If the server does not expose `_tool_manager`
                        (indicates an incompatible mcp version; requires >=1.28.1).
    """

    def __init__(self, server: FastMCP) -> None:
        if not hasattr(server, "_tool_manager"):
            raise AttributeError(
                "FastMCP instance has no '_tool_manager' attribute. "
                "This client targets mcp>=1.28.1 — verify your installed version."
            )
        self._tm = server._tool_manager

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return [t.name for t in self._tm.list_tools()]

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool by name with keyword arguments.

        Args:
            name:    The tool name as registered on the server.
            **kwargs: Keyword arguments forwarded to the tool callable.

        Returns:
            Whatever the tool callable returns (typically str for FinVibe tools).

        Raises:
            ValueError: If no tool with the given name is registered.
        """
        tool = self._tm.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' is not registered on this server.")
        return tool.fn(**kwargs)
