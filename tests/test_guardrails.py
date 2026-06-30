"""Unit tests for security.guardrails — no API calls required."""
import pytest
from security.guardrails import (
    check_input,
    is_safe,
    GUARDRAIL_MESSAGE,
    GUARDRAIL_PHRASES,
)


class TestCheckInput:
    def test_life_savings_triggers(self):
        assert check_input("Should I put my life savings into NVDA?") == GUARDRAIL_MESSAGE

    def test_all_my_money_triggers(self):
        assert check_input("I want to invest all my money in AAPL") == GUARDRAIL_MESSAGE

    def test_should_i_buy_triggers(self):
        assert check_input("should I buy TSLA right now?") == GUARDRAIL_MESSAGE

    def test_should_i_sell_triggers(self):
        assert check_input("Should I sell my MSFT shares?") == GUARDRAIL_MESSAGE

    def test_invest_everything_triggers(self):
        assert check_input("I want to invest everything in crypto") == GUARDRAIL_MESSAGE

    def test_safe_market_query_returns_none(self):
        assert check_input("What is the vibe on AAPL right now?") is None

    def test_comparison_query_is_safe(self):
        assert check_input("How does TSLA compare to AAPL?") is None

    def test_case_insensitive_upper(self):
        assert check_input("SHOULD I BUY TSLA?") == GUARDRAIL_MESSAGE

    def test_case_insensitive_mixed(self):
        assert check_input("Life Savings at risk?") == GUARDRAIL_MESSAGE

    def test_phrase_embedded_in_sentence(self):
        assert check_input("I'm worried about my life savings shrinking") == GUARDRAIL_MESSAGE

    def test_returns_message_not_bool(self):
        result = check_input("should I sell everything")
        assert isinstance(result, str)
        assert "GUARDRAIL" in result

    def test_all_phrases_covered(self):
        # Every phrase in the list must trigger the guardrail
        for phrase in GUARDRAIL_PHRASES:
            assert check_input(f"query about {phrase} here") == GUARDRAIL_MESSAGE


class TestIsSafe:
    def test_safe_query_returns_true(self):
        assert is_safe("What is the vibe on MSFT?") is True

    def test_banned_query_returns_false(self):
        assert is_safe("Should I sell my life savings?") is False

    def test_empty_string_is_safe(self):
        assert is_safe("") is True

    def test_ticker_only_is_safe(self):
        assert is_safe("AAPL") is True
