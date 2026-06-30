"""Quantitative worker agent — fetches price data and interprets MA5 signals."""
from __future__ import annotations

from pathlib import Path

from google.adk import Agent

from skills.stock_data_skill import get_stock_data

_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "quant_prompt.txt"
).read_text(encoding="utf-8").strip()

MODEL = "gemini-2.0-flash"

quant_agent = Agent(
    name="quant_agent",
    model=MODEL,
    description=(
        "Quantitative analyst. Retrieves and interprets price data and "
        "technical metrics for a given stock ticker."
    ),
    instruction=_PROMPT,
    tools=[get_stock_data],
)
