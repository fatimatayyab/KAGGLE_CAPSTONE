"""FinVibe MCP Data Server.

Registers the two market-data skills as MCP tools on a FastMCP server
instance.  The server is imported and shared across the application —
never instantiated more than once.

Import:
    from finvibe_mcp import mcp_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from skills.stock_data_skill import get_stock_data
from skills.news_sentiment_skill import get_stock_news

# Singleton server instance — shared by client.py and agent tool bridges
mcp_server = FastMCP("FinVibe-DataServer")

# Register skills as MCP tools.
# FastMCP's .tool() decorator returns the original function unchanged,
# so get_stock_data and get_stock_news remain directly callable after this.
mcp_server.tool()(get_stock_data)
mcp_server.tool()(get_stock_news)
