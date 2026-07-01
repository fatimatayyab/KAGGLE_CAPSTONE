"""FinVibe: Multi-Agent Market Analyst - Standalone CLI

Usage:
    python main.py

Requires GOOGLE_API_KEY in a .env file in the same directory.
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)
if not os.environ.get("GOOGLE_API_KEY"):
    sys.exit("ERROR: GOOGLE_API_KEY not found. Add it to your .env file.")

from memory import SessionMemory
from orchestration import _run_with_retry

_MODEL = "gemini-2.5-flash"


async def main() -> None:
    memory = SessionMemory()

    print(f"\n{'=' * 62}")
    print("  FinVibe  |  Multi-Agent Market Analyst")
    print(f"{'=' * 62}")
    print(f"  Model   : {_MODEL}")
    print(f"  Agents  : supervisor_agent -> quant_agent + sentiment_agent")
    print(f"  Tools   : get_stock_data  .  get_stock_news  (via MCP)")
    print(f"{'=' * 62}")
    print("  Ask any market question. Type 'quit' to exit.")
    print(f"{'=' * 62}\n")

    while True:
        try:
            user_input = input("  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [exit] Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n  [exit] Session ended. {memory.turn_count} turn(s) recorded.")
            break

        _run_with_retry(user_input, memory)


if __name__ == "__main__":
    asyncio.run(main())
