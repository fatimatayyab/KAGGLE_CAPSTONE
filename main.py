"""FinVibe: Multi-Agent Market Analyst — Standalone CLI

Usage:
    python main.py

Requires GOOGLE_API_KEY in a .env file in the same directory.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)
if not os.environ.get("GOOGLE_API_KEY"):
    sys.exit("ERROR: GOOGLE_API_KEY not found. Add it to your .env file.")

import yfinance as yf
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.genai import types
from mcp.server.fastmcp import FastMCP

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL        = "gemini-2.0-flash"
_USER_ID     = "finvibe-user"
_SESSION_ID  = "finvibe-secure-session"
_RETRY_WAIT  = 35
_MAX_RETRIES = 2

_GUARDRAIL_PHRASES: list[str] = [
    "life savings",
    "all my money",
    "should I buy",
    "should I sell",
    "invest everything",
]
_GUARDRAIL_MESSAGE = (
    "⚠️  GUARDRAIL TRIGGERED: This agent provides analysis for educational "
    "purposes only and is not licensed financial advice. It cannot recommend "
    "specific investment actions regarding your life savings."
)

# ── MCP Data Server ────────────────────────────────────────────────────────────
mcp_server = FastMCP("FinVibe-DataServer")


@mcp_server.tool()
def get_stock_data(ticker: str) -> str:
    """Fetch the latest close price and 5-day moving average for a stock ticker.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.

    Returns:
        A formatted string: 'Latest Close: $X.XX, 5-Day MA: $Y.YY'.
    """
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1mo")
    if hist.empty:
        return f"No price data found for ticker '{ticker}'."
    if len(hist) < 5:
        return f"Insufficient history for MA5 (only {len(hist)} day(s) for '{ticker}')."
    latest_close = hist["Close"].iloc[-1]
    ma5          = hist["Close"].rolling(window=5).mean().iloc[-1]
    return f"Latest Close: ${latest_close:.2f}, 5-Day MA: ${ma5:.2f}"


@mcp_server.tool()
def get_stock_news(ticker: str) -> str:
    """Fetch the top-3 recent news headlines for a stock ticker.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.

    Returns:
        A formatted string with three recent headlines and their sources.
    """
    stock    = yf.Ticker(ticker)
    raw_news = stock.news
    if not raw_news:
        return f"No recent news found for ticker '{ticker}'."
    lines: list[str] = []
    for item in raw_news[:3]:
        content  = item.get("content", {})
        title    = content.get("title", "No title")
        provider = (
            content.get("provider", {}).get("displayName", "Unknown")
            if isinstance(content.get("provider"), dict)
            else "Unknown"
        )
        lines.append(f"  • {title}  [{provider}]")
    return f"Top headlines for {ticker.upper()}:\n" + "\n".join(lines)


# ── Agents ─────────────────────────────────────────────────────────────────────
quant_agent = Agent(
    name="quant_agent",
    model=MODEL,
    description="Quantitative analyst. Retrieves price data and technical metrics.",
    instruction=(
        "You are a quantitative data analyst. Your sole focus is numerical market data. "
        "When given a stock ticker, call get_stock_data to fetch the latest close price "
        "and 5-day moving average. Report the exact figures and state whether the stock "
        "is trading above or below its MA5 — that gap signals near-term momentum direction."
    ),
    tools=[get_stock_data],
)

sentiment_agent = Agent(
    name="sentiment_agent",
    model=MODEL,
    description="Sentiment analyst. Reads headlines and determines investor mood.",
    instruction=(
        "You are a market sentiment analyst. Your sole focus is news and market psychology. "
        "When given a stock ticker, call get_stock_news to fetch the top-3 recent headlines. "
        "Analyze tone and output:\n"
        "  1. A one-word verdict: Bullish, Bearish, or Neutral.\n"
        "  2. A single-sentence rationale citing at least one specific headline."
    ),
    tools=[get_stock_news],
)

supervisor_agent = Agent(
    name="supervisor_agent",
    model=MODEL,
    description="Senior Market Analyst. Routes tasks and synthesizes a 3-bullet summary.",
    instruction=(
        "You are a Senior Market Analyst. When asked to analyze a stock:\n"
        "1. Call quant_agent with the ticker to obtain price and technical metrics.\n"
        "2. Call sentiment_agent with the same ticker to obtain news sentiment.\n"
        "3. Synthesize both into an Executive Market Summary with exactly three bullets:\n"
        "   • Price Snapshot   — quote close price and MA5; note trend direction.\n"
        "   • Market Sentiment — state verdict (Bullish/Bearish/Neutral) and cite a headline.\n"
        "   • Analyst Take     — your integrated conclusion on the overall signal.\n"
        "Be concise, professional, and grounded in the data you received."
    ),
    tools=[AgentTool(agent=quant_agent), AgentTool(agent=sentiment_agent)],
)

# ── Runner & session memory ────────────────────────────────────────────────────
runner = InMemoryRunner(agent=supervisor_agent, app_name="FinVibe-Analyst")
runner.auto_create_session = True

session_memory: list[dict[str, str]] = []


# ── Core pipeline ──────────────────────────────────────────────────────────────
def _stream_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    state_delta: dict[str, Any] | None = None,
) -> str:
    """Drive runner.run() and return the supervisor's final response text."""
    message = types.Content(role="user", parts=[types.Part(text=query)])
    supervisor_response = ""
    last_final_response = ""

    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
        state_delta=state_delta,
    ):
        if event.partial:
            continue

        author = event.author or "unknown"

        fcs = event.get_function_calls()
        if fcs:
            for fc in fcs:
                args_str = ", ".join(f"{k}={v!r}" for k, v in (fc.args or {}).items())
                print(f"  [{author}]  ▶ {fc.name}({args_str})")
            continue

        frs = event.get_function_responses()
        if frs:
            for fr in frs:
                raw     = fr.response or {}
                payload = raw.get("result", raw) if isinstance(raw, dict) else raw
                snippet = str(payload).replace("\n", " ")[:90]
                print(f"  [tool:{fr.name}]  ◀ {snippet}…")
            continue

        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text).strip()
            if text:
                snippet = text[:115] + ("…" if len(text) > 115 else "")
                print(f"  [{author}]  {snippet}")

        if event.is_final_response() and event.content and event.content.parts:
            candidate = "".join(p.text for p in event.content.parts if p.text).strip()
            if candidate:
                last_final_response = candidate
                if author == "supervisor_agent":
                    supervisor_response = candidate

    return supervisor_response or last_final_response


def input_guardrail(user_input: str) -> str | None:
    """Return the guardrail warning if a banned phrase is detected, else None."""
    lowered = user_input.lower()
    for phrase in _GUARDRAIL_PHRASES:
        if phrase in lowered:
            return _GUARDRAIL_MESSAGE
    return None


def run_secure_market_pipeline(user_query: str) -> str | None:
    """Guardrail-protected, context-aware entry point for the analysis pipeline.

    Returns the supervisor's summary, or None if the guardrail fired.
    """
    warning = input_guardrail(user_query)
    if warning:
        print(f"\n{'═' * 62}")
        print("  FinVibe  |  Input Guardrail")
        print(f"{'═' * 62}")
        print(f"  {warning}")
        print(f"{'═' * 62}\n")
        return None

    recent_context = session_memory[-4:]
    state_delta    = {"recent_history": recent_context} if recent_context else None

    print(f"\n{'═' * 62}")
    print("  FinVibe  |  Multi-Agent Market Analysis")
    print(f"{'═' * 62}")
    print(f"  Query   : {user_query}")
    print(f"  Session : {_SESSION_ID}  (turn {len(session_memory) // 2 + 1})")
    if recent_context:
        print(f"  Context : {len(recent_context)} prior message(s) injected")
    print(f"{'─' * 62}")

    final = _stream_pipeline(_SESSION_ID, _USER_ID, user_query, state_delta)

    print(f"{'─' * 62}")
    print("  EXECUTIVE MARKET SUMMARY")
    print(f"{'─' * 62}")
    print(final or "[No final response captured — review event log above]")
    print(f"{'═' * 62}\n")

    if final:
        session_memory.append({"role": "user",      "content": user_query})
        session_memory.append({"role": "assistant",  "content": final})

    return final


def _run_with_retry(query: str) -> str | None:
    """Call run_secure_market_pipeline with automatic 429 retry."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = run_secure_market_pipeline(query)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                result = ""
            else:
                print(f"\n  [error] {e}")
                return None

        if result is None:
            return None
        if result:
            return result

        if attempt < _MAX_RETRIES:
            print(
                f"\n  [429] No response captured — waiting {_RETRY_WAIT}s "
                f"before retry (attempt {attempt}/{_MAX_RETRIES})..."
            )
            time.sleep(_RETRY_WAIT)
            print("  [429] Retrying.\n")
        else:
            print(
                f"\n  [429] Still no response after {_MAX_RETRIES} attempts. "
                "Daily quota may be exhausted — try again tomorrow or enable billing."
            )
    return None


# ── Entry point ────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n{'━' * 62}")
    print("  FinVibe  |  Multi-Agent Market Analyst")
    print(f"{'━' * 62}")
    print(f"  Model   : {MODEL}")
    print(f"  Agents  : supervisor_agent → quant_agent + sentiment_agent")
    print(f"  Tools   : get_stock_data  ·  get_stock_news  (via MCP)")
    print(f"{'━' * 62}")
    print("  Ask any market question. Type 'quit' to exit.")
    print(f"{'━' * 62}\n")

    while True:
        try:
            user_input = input("  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [exit] Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n  [exit] Session ended. {len(session_memory) // 2} turn(s) recorded.")
            break

        _run_with_retry(user_input)


if __name__ == "__main__":
    asyncio.run(main())
