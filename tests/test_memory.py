"""Unit tests for memory.session_memory — no API calls required."""
import pytest
from memory.session_memory import SessionMemory


class TestInitialState:
    def test_turn_count_is_zero(self):
        assert SessionMemory().turn_count == 0

    def test_message_count_is_zero(self):
        assert SessionMemory().message_count == 0

    def test_recent_context_is_empty(self):
        assert SessionMemory().recent_context() == []

    def test_state_delta_is_none_when_empty(self):
        assert SessionMemory().state_delta() is None

    def test_len_is_zero(self):
        assert len(SessionMemory()) == 0


class TestAddTurn:
    def test_turn_count_increments(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        assert mem.turn_count == 1

    def test_message_count_is_double_turns(self):
        mem = SessionMemory()
        mem.add_turn("q1", "a1")
        mem.add_turn("q2", "a2")
        assert mem.message_count == 4

    def test_roles_are_set_correctly(self):
        mem = SessionMemory()
        mem.add_turn("my question", "my answer")
        turns = list(mem)
        assert turns[0] == {"role": "user",      "content": "my question"}
        assert turns[1] == {"role": "assistant",  "content": "my answer"}

    def test_len_equals_message_count(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        assert len(mem) == mem.message_count


class TestRecentContext:
    def test_returns_all_messages_when_under_limit(self):
        mem = SessionMemory(max_context_pairs=2)
        mem.add_turn("q1", "a1")
        ctx = mem.recent_context()
        assert len(ctx) == 2

    def test_respects_max_context_pairs(self):
        mem = SessionMemory(max_context_pairs=2)
        mem.add_turn("q1", "a1")
        mem.add_turn("q2", "a2")
        mem.add_turn("q3", "a3")  # 3 turns, window is 2
        ctx = mem.recent_context()
        assert len(ctx) == 4  # 2 pairs × 2 messages
        assert ctx[0]["content"] == "q2"
        assert ctx[-1]["content"] == "a3"

    def test_returns_most_recent_messages(self):
        mem = SessionMemory(max_context_pairs=1)
        mem.add_turn("early q", "early a")
        mem.add_turn("recent q", "recent a")
        ctx = mem.recent_context()
        assert len(ctx) == 2
        assert ctx[0]["content"] == "recent q"


class TestStateDelta:
    def test_returns_none_when_empty(self):
        mem = SessionMemory()
        assert mem.state_delta() is None

    def test_returns_dict_with_recent_history_key(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        delta = mem.state_delta()
        assert delta is not None
        assert "recent_history" in delta

    def test_recent_history_matches_recent_context(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        assert mem.state_delta()["recent_history"] == mem.recent_context()


class TestClear:
    def test_resets_turn_count(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        mem.clear()
        assert mem.turn_count == 0

    def test_recent_context_empty_after_clear(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        mem.clear()
        assert mem.recent_context() == []

    def test_state_delta_none_after_clear(self):
        mem = SessionMemory()
        mem.add_turn("q", "a")
        mem.clear()
        assert mem.state_delta() is None
