"""Sentiment worker agent — reads headlines and assigns a market mood verdict."""
from __future__ import annotations

from pathlib import Path

from google.adk import Agent

from skills.news_sentiment_skill import get_stock_news

_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "sentiment_prompt.txt"
).read_text(encoding="utf-8").strip()

MODEL = "gemini-2.5-flash"

sentiment_agent = Agent(
    name="sentiment_agent",
    model=MODEL,
    description=(
        "Sentiment analyst. Reads recent news headlines and determines "
        "investor mood as Bullish, Bearish, or Neutral."
    ),
    instruction=_PROMPT,
    tools=[get_stock_news],
)
