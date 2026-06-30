"""InMemoryRunner singleton for the FinVibe pipeline.

Instantiated once at import time and shared across the application.
InMemoryRunner.run() is a synchronous generator that executes async ADK
logic in a dedicated background thread — safe in both CLI and Jupyter contexts.

Import:
    from orchestration.runner import runner
"""
from __future__ import annotations

from google.adk.runners import InMemoryRunner

from agents.supervisor_agent import supervisor_agent

runner = InMemoryRunner(agent=supervisor_agent, app_name="FinVibe-Analyst")
runner.auto_create_session = True
