"""News sentiment skill: top-3 headlines via yfinance."""
from __future__ import annotations

import yfinance as yf


def get_stock_news(ticker: str) -> str:
    """Fetch the top-3 recent news headlines for a stock ticker.

    Queries Yahoo Finance for the latest news items associated with the
    given ticker and formats the top three as a bulleted list with source
    attribution for downstream sentiment analysis.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.

    Returns:
        A formatted multi-line string with up to three headline bullets.
        Returns a descriptive error string (not an exception) when no
        news is available.
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
