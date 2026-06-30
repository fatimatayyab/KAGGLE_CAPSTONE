"""Quantitative data skill: close price and 5-day moving average via yfinance."""
from __future__ import annotations

import yfinance as yf


def get_stock_data(ticker: str) -> str:
    """Fetch the latest close price and 5-day moving average for a stock ticker.

    Retrieves one month of OHLCV history from Yahoo Finance, computes the
    5-day simple moving average on closing prices, and returns both figures
    as a formatted string for downstream agent consumption.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.

    Returns:
        A formatted string: 'Latest Close: $X.XX, 5-Day MA: $Y.YY'.
        Returns a descriptive error string (not an exception) when data is
        unavailable or insufficient for MA5 computation.
    """
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1mo")

    if hist.empty:
        return f"No price data found for ticker '{ticker}'."

    if len(hist) < 5:
        return (
            f"Insufficient history for MA5 "
            f"(only {len(hist)} trading day(s) returned for '{ticker}')."
        )

    latest_close = hist["Close"].iloc[-1]
    ma5          = hist["Close"].rolling(window=5).mean().iloc[-1]

    return f"Latest Close: ${latest_close:.2f}, 5-Day MA: ${ma5:.2f}"
