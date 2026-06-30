"""FinVibe pipeline: event streaming, guardrail gating, and 429 retry logic.

Public API
----------
run_secure_market_pipeline(user_query, memory)
    Guardrail-protected, context-aware entry point.  Pass a SessionMemory
    instance to accumulate cross-turn context.

_run_with_retry(query, memory)
    Wraps run_secure_market_pipeline with automatic 429 backoff/retry.

_stream_pipeline(session_id, user_id, query, state_delta)
    Low-level event processor; use run_secure_market_pipeline for guarded access.
"""
from __future__ import annotations

import time
from typing import Any

from google.genai import types

from memory.session_memory import SessionMemory
from orchestration.runner import runner
from security.guardrails import check_input, GUARDRAIL_MESSAGE

# ── Session constants ──────────────────────────────────────────────────────────
_USER_ID    = "finvibe-user"
_SESSION_ID = "finvibe-secure-session"

# ── Retry configuration ────────────────────────────────────────────────────────
_RETRY_WAIT  = 35   # seconds — matches Gemini free-tier retry window
_MAX_RETRIES = 2


# ── Core event processor ───────────────────────────────────────────────────────

def _stream_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    state_delta: dict[str, Any] | None = None,
) -> str:
    """Drive runner.run() and return the supervisor's final response text.

    Prints every completed event (tool calls, tool responses, agent text)
    as they arrive, then captures and returns the supervisor's synthesis.

    Args:
        session_id:  ADK session identifier.
        user_id:     Stable user identifier passed to the runner.
        query:       Natural-language query to send as a new user message.
        state_delta: Optional key/value pairs merged into ADK session state
                     before the run (used to inject recent_history context).

    Returns:
        The supervisor's Executive Market Summary, or the last agent text
        seen if the supervisor's author label is absent.  Empty string if
        no final response was captured (e.g. a 429 swallowed in the thread).
    """
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
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in (fc.args or {}).items()
                )
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
            candidate = "".join(
                p.text for p in event.content.parts if p.text
            ).strip()
            if candidate:
                last_final_response = candidate
                if author == "supervisor_agent":
                    supervisor_response = candidate

    return supervisor_response or last_final_response


# ── Guarded pipeline entry point ───────────────────────────────────────────────

def run_secure_market_pipeline(
    user_query: str,
    memory: SessionMemory,
) -> str | None:
    """Guardrail-protected, context-aware pipeline entry point.

    Execution contract:
      1. Guardrail  — banned phrases intercepted before any API call is made.
      2. Context    — last N memory entries injected via state_delta so the
                      supervisor retains cross-turn awareness.
      3. Run        — delegates to _stream_pipeline(); no event logic duplicated.
      4. Persist    — successful (query, response) pair appended to memory.

    Args:
        user_query: Natural-language market analysis request.
        memory:     Active SessionMemory instance for this session.

    Returns:
        Supervisor's Executive Market Summary string, or None if the
        guardrail fired (zero Gemini API calls are made in that case).
    """
    # 1. Guardrail check — fast path, zero API cost
    warning = check_input(user_query)
    if warning:
        print(f"\n{'═' * 62}")
        print("  FinVibe  |  Input Guardrail")
        print(f"{'═' * 62}")
        print(f"  {GUARDRAIL_MESSAGE}")
        print(f"{'═' * 62}\n")
        return None

    # 2. Build context payload
    state_delta = memory.state_delta()

    print(f"\n{'═' * 62}")
    print("  FinVibe  |  Multi-Agent Market Analysis")
    print(f"{'═' * 62}")
    print(f"  Query   : {user_query}")
    print(f"  Session : {_SESSION_ID}  (turn {memory.turn_count + 1})")
    if state_delta:
        print(f"  Context : {len(state_delta['recent_history'])} prior message(s) injected")
    print(f"{'─' * 62}")

    # 3. Run pipeline
    final = _stream_pipeline(_SESSION_ID, _USER_ID, user_query, state_delta)

    # 4. Print summary
    print(f"{'─' * 62}")
    print("  EXECUTIVE MARKET SUMMARY")
    print(f"{'─' * 62}")
    print(final or "[No final response captured — review event log above]")
    print(f"{'═' * 62}\n")

    # 5. Persist turn
    if final:
        memory.add_turn(user_query, final)

    return final


# ── Retry wrapper ──────────────────────────────────────────────────────────────

def _run_with_retry(query: str, memory: SessionMemory) -> str | None:
    """Call run_secure_market_pipeline with automatic 429 backoff/retry.

    Handles two 429 surface patterns:
      - Propagated exception: caught via try/except, checks for '429' in str(e).
      - Thread-swallowed exception: detected via empty string return from pipeline.

    Args:
        query:  User query string.
        memory: Active SessionMemory instance for this session.

    Returns:
        Pipeline result, or None if guardrail fired or all retries failed.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = run_secure_market_pipeline(query, memory)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                result = ""
            else:
                print(f"\n  [error] {e}")
                return None

        if result is None:
            return None     # guardrail triggered — hard stop, no retry

        if result:
            return result   # success

        # Empty string — 429 swallowed by ADK background thread
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
