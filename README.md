# FinVibe: Multi-Agent Market Analyst

A fully-functional **multi-agent AI system** built with **Google's Agent Development Kit (ADK)**
that generates an *Executive Market Summary* for any stock ticker.
It combines quantitative price data analysis with qualitative news sentiment analysis,
enforced by financial safety guardrails — all six core course concepts are demonstrated.

---

## Architecture

```
User Query
    |
    v
[Input Guardrail]  <-- security/guardrails.py
    |  (blocked queries return None immediately)
    v
[Supervisor Agent] <-- agents/supervisor_agent.py  (gemini-2.0-flash)
    |-- AgentTool --> [Quant Agent]     <-- agents/quant_agent.py
    |                     |-- tool --> get_stock_data() --> MCP Client --> skills/stock_data_skill.py --> yfinance
    |-- AgentTool --> [Sentiment Agent] <-- agents/sentiment_agent.py
                          |-- tool --> get_stock_news() --> MCP Client --> skills/news_sentiment_skill.py --> yfinance
    |
    v
[Session Memory]  <-- memory/session_memory.py  (stores turns for multi-turn context)
    |
    v
Executive Market Summary (3-bullet ASCII output)
```

---

## Folder Structure

```
FinVibe/
├── agents/
│   ├── supervisor_agent.py   # Orchestrates sub-agents; synthesizes 3-bullet summary
│   ├── quant_agent.py        # Fetches price data; computes MA5
│   └── sentiment_agent.py    # Reads headlines; classifies investor mood
├── skills/
│   ├── stock_data_skill.py   # yfinance: latest close + 5-day MA
│   └── news_sentiment_skill.py # yfinance: top-3 news headlines + sources
├── finvibe_mcp/              # Named finvibe_mcp to avoid shadowing the PyPI `mcp` package
│   ├── server.py             # MCP server exposing get_stock_data + get_stock_news
│   └── client.py             # Sync MCP client used by agents
├── orchestration/
│   ├── workflow.py           # run_secure_market_pipeline(): guardrail + ADK runner + memory
│   └── runner.py             # CLI entry point helper
├── security/
│   └── guardrails.py         # Keyword filter; blocks life-savings / investment advice queries
├── memory/
│   └── session_memory.py     # In-process list-based session memory; injected as state_delta
├── prompts/
│   ├── supervisor_prompt.txt
│   ├── quant_prompt.txt
│   └── sentiment_prompt.txt
├── models/
│   └── response_models.py
├── tests/
│   ├── test_agents.py        # 14 tests: skills, agent construction, pipeline orchestration
│   ├── test_guardrails.py    # 2 tests: safe/unsafe query classification
│   └── test_memory.py        # 1 test: add_turn, get_recent_context, clear_memory
├── notebooks/
│   ├── FinVibe_Capstone.ipynb
│   └── experiments.ipynb
├── diagrams/
│   ├── architecture.png
│   ├── sequence_diagram.png
│   └── concept_mapping.png
├── output/
│   ├── executive_summary_aapl.json
│   └── executive_summary_tsla.json
├── main.py                   # Interactive CLI entry point
├── requirements.txt
└── .env
```

---

## The Six Core Course Concepts

| # | Concept | Where Demonstrated |
|---|---|---|
| 1 | **Multi-Agent System** | `supervisor_agent` delegates to `quant_agent` and `sentiment_agent` via `AgentTool` |
| 2 | **Google ADK** | All agents use `google.adk.Agent`; runner is `google.adk.runners.InMemoryRunner` |
| 3 | **MCP (Model Context Protocol)** | `finvibe_mcp/server.py` exposes tools; `finvibe_mcp/client.py` calls them from agents |
| 4 | **Tool Use** | `get_stock_data` and `get_stock_news` are function tools registered on sub-agents |
| 5 | **Safety Guardrails** | `security/guardrails.py` blocks financial advice queries before they reach the LLM |
| 6 | **Session Memory** | `memory/session_memory.py` stores conversation turns and injects context via `state_delta` |

---

## Setup

```powershell
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key in .env
# GOOGLE_API_KEY=your_key_here
```

## How to Run

### Interactive CLI
```powershell
python main.py
```

### Run Tests
```powershell
pytest tests/ -v
```

### Sample Queries
```
> What is the vibe on AAPL right now?
> How does that compare to TSLA?
> Should I put my life savings into NVDA?   # <-- Guardrail fires
```

---

## Test Results

```
============================= test session results ==============================
collected 17 items

tests/test_agents.py ............                                        PASSED
tests/test_guardrails.py ..                                              PASSED
tests/test_memory.py .                                                   PASSED

===================== 17 passed in 10.21s =========================
```

> The 4 warnings are `DeprecationWarning: BaseAgentConfig is deprecated` from Google ADK internals.
> These do not affect functionality.

---

## Key Design Decisions

**Why `finvibe_mcp/` instead of `mcp/`**
The folder avoids shadowing the installed `mcp` PyPI package. Renaming to `mcp/` would cause Python to resolve `from mcp import ...` to the local directory, breaking MCP library imports.

**Why `_stream_pipeline` is mocked in pipeline tests**
The `supervisor_agent` uses `AgentTool(agent=quant_agent)` — ADK registers sub-agents as opaque callable tools. Mocking `_stream_pipeline` directly tests orchestration logic (guardrail gating, memory updates, `state_delta` injection) without fighting ADK internals.

**Windows ASCII-safe output**
All print statements use pure ASCII (`->`, `<-`, `*`, `=`, `-`) instead of Unicode box-drawing characters, preventing `UnicodeEncodeError` on Windows CP1252 consoles.

**Synchronous MCP Client**
`finvibe_mcp/client.py` uses a sync subprocess-based client so it can be called from within synchronous ADK tool functions without event-loop conflicts.
