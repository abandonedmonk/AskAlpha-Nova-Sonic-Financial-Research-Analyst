# Voice AI Agent — Nova Sonic Financial Research Terminal

> AWS Bedrock Nova Sonic Hackathon project.  
> Speak a market question → live data, SEC filings, Monte Carlo simulation, vault logging — all spoken back.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [File-by-File Reference](#4-file-by-file-reference)
   - [Root level](#root-level)
   - [nova_sonic/](#nova_sonic)
   - [event_router/](#event_router)
   - [tools/](#tools)
   - [compute/](#compute)
   - [data/](#data)
   - [tests/](#tests)
   - [vault/](#vault)
5. [The Four Financial Tools](#5-the-four-financial-tools)
6. [Session State Machine](#6-session-state-machine)
7. [Setup & Running](#7-setup--running)
8. [Environment Variables](#8-environment-variables)
9. [Execution Paths & Fallbacks](#9-execution-paths--fallbacks)
10. [Running Tests](#10-running-tests)
11. [Demo Script](#11-demo-script)

---

## 1. What This Project Does

This is a **voice-native financial research agent** built on top of AWS Bedrock Nova Sonic.

You speak → Nova Sonic transcribes (STT), reasons (LLM), and speaks back (TTS).  
When Nova Sonic needs real data, it fires a **tool call** into the Python backend (the Event Router).  
The Event Router dispatches to one of four financial backends and returns the result to Nova Sonic, which reads it aloud.

You can interrupt it mid-sentence. Nova Sonic's built-in VAD handles that.

**Everything AWS manages:**

- Speech-to-text (STT)
- Language model reasoning (LLM)
- Text-to-speech (TTS)
- Voice Activity Detection + interruption (VAD)

**Everything this project adds:**

- The four financial tool backends
- The Event Router that connects Nova Sonic to those backends
- The bidirectional WebSocket bridge between the browser and Nova Sonic

---

## 2. Architecture

```
Browser (microphone)
        │  raw PCM-16 @ 16 kHz
        ▼
┌──────────────────────────────────────┐
│  FastAPI WebSocket  /ws/voice        │  ← main.py + event_router/router.py
│  (NovaSonicSession manages the pipe) │
└──────────────────┬───────────────────┘
                   │ bidirectional boto3 stream
                   ▼
        ┌──────────────────────┐
        │  AWS Bedrock         │
        │  Nova Sonic v1       │  STT → LLM → TTS (all managed by AWS)
        │  (amazon.nova-       │  VAD + interruption (built-in)
        │   sonic-v1:0)        │
        └──────────┬───────────┘
                   │ tool call JSON event
                   ▼
        ┌──────────────────────────────────────────┐
        │  Event Router  (event_router/router.py)  │
        │  dispatch(tool_name, tool_input) → dict  │
        └──┬──────────┬──────────────┬─────────────┘
           │          │              │              │
           ▼          ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Tool 1   │  │ Tool 2   │  │ Tool 3   │  │ Tool 4   │
  │ market   │  │ sec_rag  │  │ quant_   │  │ vault_   │
  │ _data.py │  │ .py      │  │ model.py │  │ logger.py│
  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
     Finnhub +     Bedrock KB      ironclad-       ./vault/
     Polygon       (or local       runtime or      *.md files
     fallback      FAISS index)    native Python
```

---

## 3. Directory Structure

```
Voice_AI_Agent/
│
├── main.py                      # FastAPI app factory & entry point
├── config.py                    # Pydantic settings (reads .env)
├── requirements.txt             # Python dependencies
├── .env.example                 # Template — copy to .env and fill in keys
├── .gitignore
│
├── nova_sonic/                  # AWS Bedrock / Nova Sonic layer
│   ├── __init__.py
│   ├── client.py                # boto3 stream wrapper + event builders
│   ├── session.py               # per-connection state machine
│   └── tool_schemas.py          # JSON tool schemas injected at session start
│
├── event_router/                # Dispatch layer between Nova Sonic and tools
│   ├── __init__.py
│   ├── router.py                # WebSocket endpoint + REST endpoint + dispatch()
│   └── schemas.py               # Pydantic request/response models
│
├── tools/                       # Four financial tool backends
│   ├── __init__.py
│   ├── market_data.py           # Tool 1 — Finnhub primary + Polygon EOD fallback
│   ├── sec_rag.py               # Tool 2 — SEC filings RAG (Bedrock KB or FAISS)
│   ├── quant_model.py           # Tool 3 — Monte Carlo via ironclad or Python
│   └── vault_logger.py          # Tool 4 — Markdown note writer
│
├── compute/                     # Numerical computation
│   ├── __init__.py
│   └── monte_carlo.py           # GBM simulator (NumPy + pure-Python fallback)
│
├── data/
│   ├── build_local_index.py     # One-shot FAISS index builder from PDFs
│   └── sec_filings/             # Drop NVDA/AMD/INTC 10-K PDFs here
│
├── tests/
│   ├── __init__.py
│   ├── test_market_data.py
│   ├── test_monte_carlo.py
│   ├── test_vault_logger.py
│   └── test_event_router.py
│
├── vault/                       # Markdown notes saved by Tool 4
└── docs/
    └── PROJECT_OVERVIEW.md      # ← you are here
```

---

## 4. File-by-File Reference

### Root level

#### `main.py`

The FastAPI application entry point.

- Creates the `FastAPI` app with CORS middleware (permissive during development).
- Registers the Event Router (`event_router/router.py`) under the root path.
- Uses FastAPI's `lifespan` context manager to log startup diagnostics: AWS region, Nova Sonic model ID, whether a Bedrock Knowledge Base is configured, whether the ironclad-runtime binary is present, and the vault path.
- Exposes `if __name__ == "__main__"` to run via `python main.py` (uses `uvicorn` internally).

**Start the server:**

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

#### `config.py`

Centralised settings using **Pydantic Settings** (`pydantic-settings`).

All environment variables are validated at process startup — the app fails fast rather than mid-request.

| Setting                 | Env var                 | Default                       | Purpose                        |
| ----------------------- | ----------------------- | ----------------------------- | ------------------------------ |
| `aws_access_key_id`     | `AWS_ACCESS_KEY_ID`     | required                      | AWS auth                       |
| `aws_secret_access_key` | `AWS_SECRET_ACCESS_KEY` | required                      | AWS auth                       |
| `aws_region`            | `AWS_REGION`            | `us-east-1`                   | Bedrock region                 |
| `nova_sonic_model_id`   | `NOVA_SONIC_MODEL_ID`   | `amazon.nova-sonic-v1:0`      | Nova Sonic model               |
| `bedrock_kb_id`         | `BEDROCK_KB_ID`         | `""`                          | Bedrock Knowledge Base ID      |
| `polygon_api_key`       | `POLYGON_API_KEY`       | required                      | Polygon.io key                 |
| `finnhub_api_key`       | `FINNHUB_API_KEY`       | `""`                          | Finnhub key (primary provider) |
| `vault_path`            | `VAULT_PATH`            | `./vault`                     | Where notes are saved          |
| `ironclad_runtime_path` | `IRONCLAD_RUNTIME_PATH` | `./ironclad/ironclad-runtime` | Wasm sandbox binary            |
| `app_host`              | `APP_HOST`              | `0.0.0.0`                     | Server bind address            |
| `app_port`              | `APP_PORT`              | `8000`                        | Server port                    |
| `log_level`             | `LOG_LEVEL`             | `INFO`                        | Python log level               |

**Computed properties:**

- `settings.ironclad_available` → `True` if the ironclad binary exists on disk.
- `settings.bedrock_kb_configured` → `True` if `BEDROCK_KB_ID` is non-empty.

A singleton `settings` object is imported everywhere: `from config import settings`.

---

#### `requirements.txt`

Python package dependencies.

| Package                          | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `fastapi`                        | Web framework + WebSocket support        |
| `uvicorn[standard]`              | ASGI server                              |
| `python-dotenv`                  | `.env` file loading                      |
| `pydantic` / `pydantic-settings` | Data validation & settings               |
| `boto3` / `botocore`             | AWS SDK (Bedrock, Bedrock Agent Runtime) |
| `httpx`                          | Async HTTP client for Finnhub + Polygon  |
| `numpy`                          | Monte Carlo vectorised computation       |
| `python-multipart`               | FastAPI file/form upload support         |
| `aiofiles`                       | Async file writes for the vault logger   |

Optional (commented out — needed only for local FAISS fallback):

- `llama-index`, `faiss-cpu`, `pypdf`

---

#### `.env.example`

Template showing every supported environment variable with placeholder values. Copy to `.env` and fill in real credentials before starting the server. The `.env` file is git-ignored; `.env.example` is tracked.

---

### `nova_sonic/`

#### `nova_sonic/tool_schemas.py`

Defines the **four Bedrock-compatible JSON tool schemas** that are injected into the Nova Sonic session at startup. Nova Sonic reads these schemas to know which tools exist, what their parameters are, and when to call them.

Each schema follows the `toolSpec` format required by the Bedrock Converse/InvokeModelWithBidirectionalStream API.

| Constant                     | Tool name                    | Trigger phrase                                         |
| ---------------------------- | ---------------------------- | ------------------------------------------------------ |
| `QUERY_LIVE_MARKET_DATA`     | `query_live_market_data`     | "What is the current price of [ticker]?"               |
| `ANALYZE_SEC_FILINGS_RAG`    | `analyze_sec_filings_rag`    | "What did [company] say about [topic] in their 10-K?"  |
| `EXECUTE_QUANTITATIVE_MODEL` | `execute_quantitative_model` | "Run a Monte Carlo on [ticker] for [N] days."          |
| `LOG_RESEARCH_INSIGHT`       | `log_research_insight`       | "Save this / log this / add to vault, tag it [topic]." |

`ALL_TOOLS` is a list of all four, passed directly to `sessionStart.toolConfiguration.tools`.

---

#### `nova_sonic/client.py`

Low-level wrapper around **boto3's Bedrock Runtime client**.

**Class: `NovaSonicClient`**

| Method                                                      | What it does                                                                                           |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `__init__()`                                                | Creates a `bedrock-runtime` boto3 client using credentials from `config.settings`.                     |
| `build_session_start_event(system_prompt)`                  | Returns the `sessionStart` JSON event with inference config, system prompt, and all four tool schemas. |
| `build_audio_input_start_event(prompt_id, content_id)`      | Returns the `promptStart` event that tells Nova Sonic to expect PCM-16 audio input at 16 kHz.          |
| `build_audio_chunk_event(prompt_id, content_id, audio_b64)` | Wraps a base64-encoded audio chunk in an `audioInput` event.                                           |
| `build_audio_content_end_event(prompt_id, content_id)`      | Signals the end of an audio content block.                                                             |
| `build_tool_result_event(prompt_id, tool_use_id, result)`   | Wraps a tool execution result back into the stream as a `toolResult` event.                            |
| `open_stream()`                                             | Calls `invoke_model_with_bidirectional_stream` and returns the raw boto3 stream handler.               |

Audio format for input: `audio/lpcm`, 16 kHz, 16-bit, mono, base64-encoded.  
Audio format for output: `audio/lpcm`, 24 kHz, 16-bit, mono (Nova Sonic TTS output).

---

#### `nova_sonic/session.py`

Per-connection **state machine** for one active voice conversation.

**Enum: `SessionState`**

```
IDLE → LISTENING → MODEL_THINKING → TOOL_EXECUTING → SPEAKING → LISTENING
                                                                        └─→ CLOSED
```

| State            | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `IDLE`           | Session not yet started                                             |
| `LISTENING`      | Accepting audio from the browser, forwarding to Nova Sonic          |
| `MODEL_THINKING` | Nova Sonic is processing (internal; set on tool dispatch)           |
| `TOOL_EXECUTING` | A tool call is in-flight — audio input is dropped during this state |
| `SPEAKING`       | Nova Sonic is streaming TTS audio back                              |
| `CLOSED`         | Stream has ended                                                    |

**Class: `NovaSonicSession`**

| Method / Attribute             | What it does                                                                                                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__(tool_handlers)`      | Takes a `dispatch` callable from the Event Router. Creates a `NovaSonicClient` and an `asyncio.Queue` for output audio.                                                                      |
| `start()`                      | Opens the boto3 bidirectional stream, sends `sessionStart`, transitions to `LISTENING`, starts the `_consume_output` background task.                                                        |
| `close()`                      | Closes the boto3 stream body and sets state to `CLOSED`.                                                                                                                                     |
| `send_audio_chunk(pcm_bytes)`  | Base64-encodes and forwards a PCM chunk to Nova Sonic. Drops chunks while in `TOOL_EXECUTING` state.                                                                                         |
| `audio_output_queue`           | `asyncio.Queue[bytes]` — TTS audio chunks enqueued by `_consume_output`, drained by the WebSocket send loop.                                                                                 |
| `_consume_output()`            | Background task. Reads events from the Nova Sonic output stream. Routes `audioOutput` to the queue, routes `toolUse` to `_handle_tool_use`, handles `contentBlockStop`/`generationComplete`. |
| `_handle_tool_use(tool_event)` | Extracts `name`, `toolUseId`, and `input` from the event. Calls `self._tool_handlers(name, input)`. Returns the result via `build_tool_result_event`.                                        |
| `state`                        | Read-only property returning the current `SessionState`.                                                                                                                                     |

---

### `event_router/`

#### `event_router/schemas.py`

Pydantic models for **all data that crosses the tool boundary**.

| Model                                      | Used by                                 |
| ------------------------------------------ | --------------------------------------- |
| `SessionStatus`                            | WebSocket session status responses      |
| `MarketDataRequest` / `MarketDataResponse` | Tool 1 I/O                              |
| `SecRagRequest` / `SecRagResponse`         | Tool 2 I/O                              |
| `QuantModelRequest` / `QuantModelResponse` | Tool 3 I/O                              |
| `VaultLogRequest` / `VaultLogResponse`     | Tool 4 I/O + `/vault/log` REST endpoint |
| `ToolResult`                               | Generic envelope for any tool result    |

---

#### `event_router/router.py`

The central dispatch hub. Exposes three things:

**1. `dispatch(tool_name, tool_input)` — async function**  
Routes a Nova Sonic tool-call event to the correct backend.

```
"query_live_market_data"     → tools.market_data.get_market_snapshot()
"analyze_sec_filings_rag"    → tools.sec_rag.query_sec_filings()
"execute_quantitative_model" → tools.quant_model.run_monte_carlo()
"log_research_insight"       → tools.vault_logger.log_insight()
```

If the tool name is unknown, returns `{"error": "Unknown tool: ..."}`.  
If the backend raises, catches the exception and returns `{"error": "<message>"}` — Nova Sonic always gets a response.

**2. `GET /ws/voice` — WebSocket endpoint**  
Accepts a browser connection, creates a `NovaSonicSession`, then runs two concurrent coroutines:

- `receive_loop` — reads binary frames from the browser, calls `session.send_audio_chunk()`
- `send_loop` — drains `session.audio_output_queue` and sends PCM bytes back to the browser

The session is torn down cleanly on disconnect.

**3. `POST /vault/log` — REST endpoint**  
Direct HTTP access to the vault logger. Accepts a `VaultLogRequest` JSON body. Useful for testing Tool 4 without going through the voice pipeline, or for programmatic note creation.

**4. `GET /health`**  
Returns `{"status": "ok", "tools": [...]}`. Simple liveness check.

---

### `tools/`

#### `tools/market_data.py` — Tool 1

**Function: `get_market_snapshot(ticker: str) -> dict`**

Uses a dual-provider path:

1. **Primary:** Finnhub daily candle endpoint.
2. **Fallback:** Polygon previous-day aggregate endpoint.

Endpoints:

```
GET https://finnhub.io/api/v1/stock/candle
GET https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true
```

Extracts and returns:

| Field            | Source                                              |
| ---------------- | --------------------------------------------------- |
| `ticker`         | Uppercased input                                    |
| `price`          | Latest close (`c`)                                  |
| `open`           | Latest open (`o`)                                   |
| `high`           | Latest high (`h`)                                   |
| `low`            | Latest low (`l`)                                    |
| `volume`         | Latest volume (`v`)                                 |
| `change_pct`     | `(price - open) / open * 100` formatted as `+2.30%` |
| `summary`        | Human-readable sentence for Nova Sonic to speak     |
| `data_source`    | `Finnhub` or `Polygon fallback`                     |
| `data_freshness` | `daily latest` or `EOD (previous trading day)`      |

If Finnhub fails and Polygon fallback is used, the summary explicitly states that the answer is **EOD fallback data**.

---

#### `tools/sec_rag.py` — Tool 2

**Function: `query_sec_filings(company, topic, filing_type) -> dict`**

Two execution paths selected at runtime:

**Primary — AWS Bedrock Knowledge Base** (used when `BEDROCK_KB_ID` is set):  
Calls `bedrock-agent-runtime.retrieve()` with a natural-language query built from `"{company} {topic} {filing_type}"`. Returns the top-3 most relevant passage texts from the vector store.

**Fallback — Local FAISS index** (used when `BEDROCK_KB_ID` is empty):  
Loads a LlamaIndex `VectorStoreIndex` from `data/faiss_index/`. Queries with `similarity_top_k=3`. Requires the index to have been built first by running `data/build_local_index.py`.

In both cases, returns:

```json
{
  "company": "Nvidia",
  "topic": "supply chain",
  "passages": ["...", "...", "..."],
  "summary": "From Nvidia's filing on supply chain: ..."
}
```

---

#### `tools/quant_model.py` — Tool 3

**Function: `run_monte_carlo(ticker, days, simulations=10_000) -> dict`**

Two sub-steps:

**Step 1 — Fetch live price + volatility**  
Calls `get_market_snapshot()` for the current price. Then calls Finnhub's `/api/v1/stock/candle` daily endpoint and computes realised annualised volatility from the last available closes (`std(log_returns) * sqrt(252)`). Falls back to `σ = 0.30` if the data request fails.

**Step 2 — Run Monte Carlo**  
Two execution paths:

| Path                 | When used                             | How                                                                                                              |
| -------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **ironclad-runtime** | `IRONCLAD_RUNTIME_PATH` binary exists | Writes a self-contained GBM Python script to a temp dir, executes via `ironclad-runtime`, reads JSON from stdout |
| **Native Python**    | ironclad binary not found             | Imports and calls `compute.monte_carlo.simulate()` directly in-process                                           |

Returns:

```json
{
  "ticker": "NVDA",
  "days": 30,
  "simulations": 10000,
  "current_price": 875.5,
  "p10": 820.12,
  "p50": 878.44,
  "p90": 941.33,
  "mean": 879.11,
  "summary": "Monte Carlo on NVDA over 30 trading days ..."
}
```

---

#### `tools/vault_logger.py` — Tool 4

**Function: `log_insight(content, tags, title) -> dict`**

Writes an Obsidian-compatible Markdown note with YAML front matter to `settings.vault_path`.

**Generated note format:**

```markdown
---
tags: [semiconductors, nvidia]
date: 2025-03-01T14:32:00
source: Nova Sonic Research Terminal
---

# Note title

[content body]
```

Filename derives from the title (sanitised, max 60 chars) or an ISO timestamp if no title is given.  
Uses `aiofiles` for non-blocking async writes.

Returns `{"saved": true, "filepath": "...", "message": "..."}`.

---

### `compute/`

#### `compute/monte_carlo.py`

Standalone Geometric Brownian Motion simulator. Can be used three ways:

**1. Imported by `tools/quant_model.py`** (native Python path):

```python
from compute.monte_carlo import simulate
result = simulate(current_price=162.45, volatility=0.35, days=30)
```

**2. Executed as a subprocess by ironclad-runtime** (Wasm sandbox path):

```bash
ironclad-runtime /tmp/sandbox/monte_carlo_run.py
```

**3. Run from the CLI for testing:**

```bash
python compute/monte_carlo.py --price 162.45 --volatility 0.35 --days 30 --simulations 10000
```

**GBM formula (per step, risk-neutral drift):**

$$S_{t+1} = S_t \cdot \exp\!\left(-\tfrac{1}{2}\sigma^2 \,\Delta t + \sigma\sqrt{\Delta t}\, Z\right), \quad Z\sim\mathcal{N}(0,1), \quad \Delta t = \tfrac{1}{252}$$

**Two internal paths:**

| Path                      | Trigger             | Notes                                                         |
| ------------------------- | ------------------- | ------------------------------------------------------------- |
| `_simulate_numpy()`       | NumPy importable    | Vectorised — `(simulations × days)` matrix in one shot. Fast. |
| `_simulate_pure_python()` | NumPy not available | Stdlib-only loop. Works inside restricted Wasm environments.  |

**Returns:** `{"p10": float, "p50": float, "p90": float, "mean": float}`

---

### `data/`

#### `data/build_local_index.py`

One-shot script that ingests all PDF files from `data/sec_filings/` into a local FAISS vector index stored at `data/faiss_index/`. Run this once before starting the server if you are not using the Bedrock Knowledge Base.

```bash
pip install llama-index faiss-cpu pypdf
python data/build_local_index.py
```

Requires at least one PDF in `data/sec_filings/`. Recommended files: `nvidia-2024-10k.pdf`, `amd-2024-10k.pdf`, `intc-2024-10k.pdf` (from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar)).

#### `data/sec_filings/`

Empty directory (`.gitkeep` placeholder). Place downloaded 10-K/10-Q PDFs here before running the index builder or Bedrock ingestion.

---

### `tests/`

#### `tests/test_market_data.py`

Unit tests for `tools/market_data.py`.

| Test                                                                  | What it checks                                                             |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `test_get_market_snapshot_returns_expected_keys`                      | Finnhub primary success returns required keys                              |
| `test_get_market_snapshot_falls_back_to_polygon_with_eod_notice`      | Polygon fallback is used when Finnhub fails, and EOD disclosure is present |
| `test_get_market_snapshot_ticker_normalised`                          | Input ticker is uppercased before provider request                         |
| `test_get_market_snapshot_both_providers_fail_returns_combined_error` | Combined error includes both provider failure reasons                      |

#### `tests/test_monte_carlo.py`

Unit tests for `compute/monte_carlo.py`.

| Test                                   | What it checks                                                 |
| -------------------------------------- | -------------------------------------------------------------- |
| `test_simulate_returns_required_keys`  | All four output keys present                                   |
| `test_percentile_ordering`             | P10 ≤ P50 ≤ P90 always holds                                   |
| `test_mean_within_plausible_range`     | Mean stays within ±20% of start price over 5 days (zero drift) |
| `test_pure_python_fallback_consistent` | Pure-Python path produces valid ordered percentiles            |
| `test_high_volatility_widens_spread`   | Higher σ always widens the P10-P90 spread                      |
| `test_single_day_simulation`           | Edge case: 1-day simulation does not crash                     |

#### `tests/test_vault_logger.py`

Unit tests for `tools/vault_logger.py`.

| Test                                  | What it checks                                         |
| ------------------------------------- | ------------------------------------------------------ |
| `test_log_insight_creates_file`       | A `.md` file is created and contains the right content |
| `test_log_insight_front_matter`       | YAML front matter is present and correctly formatted   |
| `test_log_insight_no_tags`            | Works without tags                                     |
| `test_log_insight_filename_sanitised` | Special characters in title do not break the filename  |

#### `tests/test_event_router.py`

Integration tests for `event_router/router.py`.

| Test                                         | What it checks                                         |
| -------------------------------------------- | ------------------------------------------------------ |
| `test_dispatch_known_tool_delegates`         | `dispatch()` calls the right backend                   |
| `test_dispatch_unknown_tool_returns_error`   | Unknown tool name returns `{"error": ...}`             |
| `test_dispatch_tool_exception_returns_error` | Backend exception is caught and returned as error dict |
| `test_health_endpoint`                       | `GET /health` returns 200 with the tool list           |

---

### `vault/`

Runtime directory. All Markdown notes created by Tool 4 (`log_research_insight`) are written here. Contains only a `.gitkeep` in the repository. If `VAULT_PATH` in `.env` points to your Obsidian vault folder, notes will appear there instead.

---

## 5. The Four Financial Tools

| #   | Tool name                    | Backend file            | External service           | Fallback                                        |
| --- | ---------------------------- | ----------------------- | -------------------------- | ----------------------------------------------- |
| 1   | `query_live_market_data`     | `tools/market_data.py`  | Finnhub daily candles      | Polygon previous-day aggregate (EOD disclosure) |
| 2   | `analyze_sec_filings_rag`    | `tools/sec_rag.py`      | AWS Bedrock Knowledge Base | Local FAISS + LlamaIndex                        |
| 3   | `execute_quantitative_model` | `tools/quant_model.py`  | ironclad-runtime (Wasm)    | Native Python in-process                        |
| 4   | `log_research_insight`       | `tools/vault_logger.py` | Local filesystem           | None needed                                     |

---

## 6. Session State Machine

```
          ┌────────────────────────────────────┐
          │                                    │
  ┌──── IDLE ──── start() ──── LISTENING ────┐ │
  │                               │          │ │
  │                        audio chunk       │ │
  │                         arrives          │ │
  │                               │          │ │
  │                        Nova Sonic        │ │
  │                       decides to         │ │
  │                        call a tool       │ │
  │                               ▼          │ │
  │                       TOOL_EXECUTING     │ │
  │                               │          │ │
  │                         tool result      │ │
  │                         returned         │ │
  │                               ▼          │ │
  │                           LISTENING ─────┘ │
  │                               │            │
  │                     Nova Sonic starts      │
  │                      speaking TTS          │
  │                               ▼            │
  │                           SPEAKING         │
  │                               │            │
  │                    contentBlockStop or     │
  │                    generationComplete      │
  │                               ▼            │
  └─────────────────────────── CLOSED ◄────────┘
                           (on disconnect)
```

Audio chunks sent from the browser are **silently dropped** while in `TOOL_EXECUTING` state — this prevents partial utterances from confusing the model while it is processing a tool result.

---

## 7. Setup & Running

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access in `us-east-1`
- Nova Sonic model access approved in the Bedrock console
- Polygon.io API key (fallback provider)
- Finnhub API key (recommended primary provider)

### Install

```bash
cd Voice_AI_Agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Configure

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

Edit `.env` — at minimum set:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
POLYGON_API_KEY=...
```

Recommended for primary market data path:

```
FINNHUB_API_KEY=...
```

### Run

```bash
python main.py
# Server starts at http://localhost:8000
```

Check: `http://localhost:8000/health`

### Connect audio

The WebSocket endpoint is `ws://localhost:8000/ws/voice`. Send raw PCM-16 at 16 kHz/mono in binary frames; receive raw PCM-16 at 24 kHz/mono back.

---

## 8. Environment Variables

See [`.env.example`](../.env.example) for the full annotated list.

**Required (no default):**

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `POLYGON_API_KEY`

**Optional (have defaults):**

- `AWS_REGION` → `us-east-1`
- `NOVA_SONIC_MODEL_ID` → `amazon.nova-sonic-v1:0`
- `BEDROCK_KB_ID` → `""` (empty = use local FAISS fallback)
- `FINNHUB_API_KEY` → `""` (empty = skip primary and use Polygon fallback)
- `VAULT_PATH` → `./vault`
- `IRONCLAD_RUNTIME_PATH` → `./ironclad/ironclad-runtime`
- `APP_HOST` → `0.0.0.0`
- `APP_PORT` → `8000`
- `LOG_LEVEL` → `INFO`

---

## 9. Execution Paths & Fallbacks

### Market Data (Tool 1)

```
FINNHUB_API_KEY configured and Finnhub request succeeds?
   YES → Return Finnhub daily latest bar
   NO  → Try Polygon /v2/aggs/ticker/{ticker}/prev
           If Polygon succeeds, response summary includes explicit EOD fallback note
           If Polygon fails, return combined error with both provider messages
```

### SEC RAG (Tool 2)

```
BEDROCK_KB_ID set?
    YES → bedrock-agent-runtime.retrieve() → Bedrock Knowledge Base
    NO  → LlamaIndex + FAISS local index at data/faiss_index/
              (must run data/build_local_index.py first)
```

### Monte Carlo (Tool 3)

```
ironclad-runtime binary exists at IRONCLAD_RUNTIME_PATH?
    YES → write temp script → subprocess call → read JSON stdout
    NO  → import compute.monte_carlo.simulate() directly

Inside simulate():
    NumPy importable?
        YES → _simulate_numpy()    (vectorised, fast)
        NO  → _simulate_pure_python() (stdlib-only, slower, Wasm-safe)
```

---

## 10. Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests use `unittest.mock` — no real AWS credentials or Polygon API key needed.
For market-data tests, Finnhub and Polygon responses are mocked.

---

## 11. Demo Script

The intended 90-second demo flow (rehearse until reliable):

1. **"What's the current trading volume and price action for AMD?"**  
   → Tool 1 fires → Finnhub primary data (or Polygon fallback) → Nova Sonic reads price, volume, change %. If fallback is used, Nova Sonic states it is EOD data.

2. **"What did Nvidia say about their datacenter supply chain in their latest 10-K?"**  
   → Tool 2 fires → Bedrock KB (or local FAISS) → Nova Sonic quotes from the actual filing.

3. **"Run a Monte Carlo simulation on Nvidia for the next 30 days."**  
   → Tool 3 fires → ironclad-runtime (or native Python) → Nova Sonic begins reading P10/P50/P90 →  
   **[Interrupt mid-sentence]** _"Wait, skip the distribution — just give me the worst-case scenario."_  
   → Nova Sonic stops (VAD built-in) and answers.

4. **"Save this insight to my vault and tag it semiconductors."**  
   → Tool 4 fires → `.md` file written to `vault/` → Nova Sonic confirms saved.

**Total: ~90 seconds. All 4 tools. One live interruption.**
