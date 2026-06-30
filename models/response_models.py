"""Structured response models for the FinVibe pipeline.

All models are plain dataclasses (no third-party deps) and provide
a .to_dict() method for JSON serialisation to the output/ folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


# ── Guardrail ──────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Outcome of a pre-flight input guardrail check."""
    triggered: bool
    message: str | None = None

    @classmethod
    def safe(cls) -> GuardrailResult:
        return cls(triggered=False)

    @classmethod
    def blocked(cls, message: str) -> GuardrailResult:
        return cls(triggered=True, message=message)


# ── Executive summary ──────────────────────────────────────────────────────────

@dataclass
class ExecutiveSummary:
    """Structured 3-bullet output from the supervisor_agent.

    The supervisor is instructed to produce exactly three labelled bullets.
    This model stores them as individual fields for downstream use
    (JSON export, display, evaluation).  raw_text preserves the original
    unstructured supervisor response in case parsing is imperfect.
    """
    price_snapshot: str
    market_sentiment: str
    analyst_take: str
    raw_text: str = ""

    @classmethod
    def from_raw(cls, raw: str) -> ExecutiveSummary:
        """Best-effort parse of a raw supervisor response into structured fields.

        Looks for bullet labels: 'Price Snapshot', 'Market Sentiment',
        'Analyst Take'.  Falls back to storing the full text in raw_text
        with empty structured fields if parsing fails.
        """
        lines = [ln.strip().lstrip("•–-").strip() for ln in raw.splitlines() if ln.strip()]
        fields: dict[str, str] = {
            "price_snapshot":   "",
            "market_sentiment": "",
            "analyst_take":     "",
        }
        mapping = {
            "price snapshot":   "price_snapshot",
            "market sentiment": "market_sentiment",
            "analyst take":     "analyst_take",
        }
        for line in lines:
            lower = line.lower()
            for label, key in mapping.items():
                if lower.startswith(label):
                    fields[key] = line.split("—", 1)[-1].strip() if "—" in line else line
                    break

        return cls(**fields, raw_text=raw)

    def to_dict(self) -> dict[str, str]:
        return {
            "price_snapshot":   self.price_snapshot,
            "market_sentiment": self.market_sentiment,
            "analyst_take":     self.analyst_take,
            "raw_text":         self.raw_text,
        }


# ── Full pipeline result ───────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full result envelope for a single run_secure_market_pipeline call.

    Suitable for persisting to output/ as JSON or for programmatic
    downstream processing.
    """
    query: str
    session_id: str
    turn: int
    guardrail_triggered: bool
    summary: ExecutiveSummary | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @classmethod
    def blocked(cls, query: str, session_id: str, turn: int) -> PipelineResult:
        """Construct a result representing a guardrail block."""
        return cls(
            query=query,
            session_id=session_id,
            turn=turn,
            guardrail_triggered=True,
            summary=None,
        )

    @classmethod
    def from_response(
        cls,
        query: str,
        session_id: str,
        turn: int,
        raw_response: str,
    ) -> PipelineResult:
        """Construct a result from a raw supervisor response string."""
        return cls(
            query=query,
            session_id=session_id,
            turn=turn,
            guardrail_triggered=False,
            summary=ExecutiveSummary.from_raw(raw_response),
        )

    def to_dict(self) -> dict:
        return {
            "query":               self.query,
            "session_id":          self.session_id,
            "turn":                self.turn,
            "timestamp":           self.timestamp,
            "guardrail_triggered": self.guardrail_triggered,
            "summary":             self.summary.to_dict() if self.summary else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
