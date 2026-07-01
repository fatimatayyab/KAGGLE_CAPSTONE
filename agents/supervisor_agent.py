"""Supervisor agent — orchestrates quant and sentiment workers into a unified summary."""
from __future__ import annotations

from pathlib import Path

from google.adk import Agent
from google.adk.tools import AgentTool

from agents.quant_agent import quant_agent
from agents.sentiment_agent import sentiment_agent

_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "supervisor_prompt.txt"
).read_text(encoding="utf-8").strip()

MODEL = "gemini-2.5-flash"

supervisor_agent = Agent(
    name="supervisor_agent",
    model=MODEL,
    description=(
        "Senior Market Analyst. Routes analytical tasks to specialist workers "
        "and synthesizes their reports into a 3-bullet Executive Market Summary."
    ),
    instruction=_PROMPT,
    tools=[
        AgentTool(agent=quant_agent),
        AgentTool(agent=sentiment_agent),
    ],
)
