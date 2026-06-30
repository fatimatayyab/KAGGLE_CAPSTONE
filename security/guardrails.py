"""Input guardrail layer for the FinVibe pipeline.

Performs keyword-based pre-screening before any Gemini API call is made.
A blocked query returns the standard disclaimer and costs zero API tokens.
"""
from __future__ import annotations

GUARDRAIL_PHRASES: list[str] = [
    "life savings",
    "all my money",
    "should i buy",
    "should i sell",
    "invest everything",
]

GUARDRAIL_MESSAGE: str = (
    "⚠️  GUARDRAIL TRIGGERED: This agent provides analysis for educational "
    "purposes only and is not licensed financial advice. It cannot recommend "
    "specific investment actions regarding your life savings."
)


def check_input(user_input: str) -> str | None:
    """Screen a user query against the banned-phrase list.

    Comparison is case-insensitive. Returns on the first match found.

    Args:
        user_input: Raw query string submitted by the user.

    Returns:
        GUARDRAIL_MESSAGE if a banned phrase is detected, otherwise None.

    Example:
        >>> check_input("Should I put my life savings into NVDA?")
        '⚠️  GUARDRAIL TRIGGERED: ...'
        >>> check_input("What is the vibe on AAPL?") is None
        True
    """
    lowered = user_input.lower()
    for phrase in GUARDRAIL_PHRASES:
        if phrase in lowered:
            return GUARDRAIL_MESSAGE
    return None


def is_safe(user_input: str) -> bool:
    """Return True if the query passes all guardrail checks."""
    return check_input(user_input) is None
