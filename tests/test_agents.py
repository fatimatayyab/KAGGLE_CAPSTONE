"""Workflow integration tests — agents and guardrails wired together.

Tests here mock _stream_pipeline so no live Gemini API calls are made.
Run with: pytest tests/test_agents.py -v
"""
import pytest
from unittest.mock import patch

from memory.session_memory import SessionMemory
from orchestration.workflow import run_secure_market_pipeline, _run_with_retry


# ── run_secure_market_pipeline ─────────────────────────────────────────────────

class TestGuardrailGating:
    def test_banned_phrase_returns_none(self):
        mem = SessionMemory()
        result = run_secure_market_pipeline(
            "Should I put my life savings into NVDA?", mem
        )
        assert result is None

    def test_guardrail_does_not_add_to_memory(self):
        mem = SessionMemory()
        run_secure_market_pipeline("should I buy everything?", mem)
        assert mem.turn_count == 0

    def test_multiple_banned_phrases_all_blocked(self):
        mem = SessionMemory()
        for query in [
            "life savings in NVDA",
            "should I sell my shares",
            "invest everything now",
        ]:
            assert run_secure_market_pipeline(query, mem) is None
        assert mem.turn_count == 0


class TestPipelineWithMockedStream:
    @patch("orchestration.workflow._stream_pipeline")
    def test_successful_response_returned(self, mock_stream):
        mock_stream.return_value = "• Price Snapshot — AAPL at $180"
        mem = SessionMemory()
        result = run_secure_market_pipeline("What is the vibe on AAPL?", mem)
        assert result == "• Price Snapshot — AAPL at $180"

    @patch("orchestration.workflow._stream_pipeline")
    def test_successful_response_updates_memory(self, mock_stream):
        mock_stream.return_value = "• Price Snapshot — AAPL at $180"
        mem = SessionMemory()
        run_secure_market_pipeline("What is the vibe on AAPL?", mem)
        assert mem.turn_count == 1

    @patch("orchestration.workflow._stream_pipeline")
    def test_empty_response_does_not_update_memory(self, mock_stream):
        mock_stream.return_value = ""
        mem = SessionMemory()
        run_secure_market_pipeline("What is the vibe on AAPL?", mem)
        assert mem.turn_count == 0

    @patch("orchestration.workflow._stream_pipeline")
    def test_context_injected_on_second_turn(self, mock_stream):
        mock_stream.return_value = "• Analyst Take — Bullish"
        mem = SessionMemory()
        run_secure_market_pipeline("What is the vibe on AAPL?", mem)
        run_secure_market_pipeline("How does that compare to TSLA?", mem)
        # Second call must pass state_delta with prior context
        _, kwargs = mock_stream.call_args
        assert kwargs.get("state_delta") is not None or mock_stream.call_args[0][3] is not None

    @patch("orchestration.workflow._stream_pipeline")
    def test_stream_pipeline_called_with_correct_session(self, mock_stream):
        mock_stream.return_value = "• Price Snapshot"
        mem = SessionMemory()
        run_secure_market_pipeline("What is the vibe on MSFT?", mem)
        call_args = mock_stream.call_args[0]
        assert call_args[0] == "finvibe-secure-session"
        assert call_args[1] == "finvibe-user"


# ── _run_with_retry ────────────────────────────────────────────────────────────

class TestRunWithRetry:
    def test_guardrail_returns_none_without_retry(self):
        mem = SessionMemory()
        result = _run_with_retry("Should I sell my life savings?", mem)
        assert result is None

    @patch("orchestration.workflow._stream_pipeline")
    def test_success_first_attempt(self, mock_stream):
        mock_stream.return_value = "• Analyst Take — Bullish"
        mem = SessionMemory()
        result = _run_with_retry("What is the vibe on TSLA?", mem)
        assert result == "• Analyst Take — Bullish"
        assert mock_stream.call_count == 1

    @patch("orchestration.workflow.time.sleep")
    @patch("orchestration.workflow._stream_pipeline")
    def test_retries_once_on_empty_response(self, mock_stream, mock_sleep):
        mock_stream.side_effect = ["", "• Price Snapshot — TSLA at $250"]
        mem = SessionMemory()
        result = _run_with_retry("What is the vibe on TSLA?", mem)
        assert result == "• Price Snapshot — TSLA at $250"
        assert mock_stream.call_count == 2
        mock_sleep.assert_called_once_with(35)

    @patch("orchestration.workflow.time.sleep")
    @patch("orchestration.workflow._stream_pipeline")
    def test_gives_up_after_max_retries(self, mock_stream, mock_sleep):
        mock_stream.return_value = ""
        mem = SessionMemory()
        result = _run_with_retry("What is the vibe on MSFT?", mem)
        assert result is None
        assert mock_stream.call_count == 2  # _MAX_RETRIES = 2

    @patch("orchestration.workflow._stream_pipeline")
    def test_non_429_exception_propagates_as_none(self, mock_stream):
        mock_stream.side_effect = ValueError("unexpected tool failure")
        mem = SessionMemory()
        result = _run_with_retry("What is the vibe on AAPL?", mem)
        assert result is None
