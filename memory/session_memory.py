"""Active session memory for the FinVibe conversational pipeline.

Stores user/assistant turn pairs and surfaces a sliding context window
for injection into ADK session state via state_delta['recent_history'].
"""
from __future__ import annotations

from typing import Iterator

_Turn = dict[str, str]  # {"role": "user" | "assistant", "content": str}


class SessionMemory:
    """Maintains a chronological log of conversation turns.

    Each completed exchange (one user message + one assistant response)
    is stored as two consecutive entries.  The recent_context() method
    returns the last N pairs as a flat list suitable for ADK state_delta
    injection, giving the supervisor cross-turn awareness.

    Args:
        max_context_pairs: Number of most-recent (user, assistant) pairs
                           to include in the state_delta window. Default 2
                           covers the last 4 messages, matching notebook behaviour.

    Example:
        >>> mem = SessionMemory()
        >>> mem.add_turn("What is the vibe on AAPL?", "• Price Snapshot ...")
        >>> mem.turn_count
        1
        >>> mem.recent_context()
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """

    def __init__(self, max_context_pairs: int = 2) -> None:
        self._turns: list[_Turn] = []
        self._max_context_pairs = max_context_pairs

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_turn(self, user_query: str, assistant_response: str) -> None:
        """Append a completed (user, assistant) exchange to the log."""
        self._turns.append({"role": "user",      "content": user_query})
        self._turns.append({"role": "assistant",  "content": assistant_response})

    def clear(self) -> None:
        """Reset the memory log for a fresh session."""
        self._turns.clear()

    # ── Context window ────────────────────────────────────────────────────────

    def recent_context(self) -> list[_Turn]:
        """Return the last N turn-pairs as a flat message list.

        Designed for direct use as state_delta['recent_history'] in
        _stream_pipeline() calls so the supervisor agent retains cross-turn
        context without re-reading the full event history.

        Returns:
            List of up to (max_context_pairs * 2) most-recent message dicts,
            or an empty list if no turns have been recorded yet.
        """
        max_messages = self._max_context_pairs * 2
        return self._turns[-max_messages:] if self._turns else []

    def state_delta(self) -> dict[str, list[_Turn]] | None:
        """Return an ADK-ready state_delta dict, or None if memory is empty.

        Pass the return value directly to _stream_pipeline(state_delta=...).
        """
        ctx = self.recent_context()
        return {"recent_history": ctx} if ctx else None

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        """Number of completed (user + assistant) exchanges recorded."""
        return len(self._turns) // 2

    @property
    def message_count(self) -> int:
        """Total number of individual messages stored."""
        return len(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[_Turn]:
        return iter(self._turns)

    def __repr__(self) -> str:
        return (
            f"SessionMemory(turns={self.turn_count}, "
            f"max_context_pairs={self._max_context_pairs})"
        )
