# Ask Alpha — Nova Sonic Financial Research Assistant 

> Speak a market question. Get live prices, SEC filings, and Monte Carlo simulations — spoken back.  
> Built on **AWS Bedrock Nova Sonic 2** for the [Amazon Nova Hackathon](https://amazon-nova.devpost.com/).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The 4 Financial Tools](#the-4-financial-tools)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Setup & Running](#setup--running)
   - [1. Clone & configure](#1-clone--configure)
   - [2. Run the backend](#2-run-the-backend)
   - [3. Run the frontend](#3-run-the-frontend)
7. [Project Structure](#project-structure)
8. [Testing](#testing)
9. [Documentation](#documentation)

---

## Overview

Ask Alpha is a **voice-native financial research assistant**. You open the Web UI, hit the mic button, and ask anything — *"What's AMD trading at right now?"*, *"What did Nvidia say about supply chains in their 10-K?"*, *"Run a Monte Carlo on NVDA for 30 days"*, *"Save this to my vault."*

Nova Sonic 2 handles everything on the AI side: speech recognition, understanding your intent, picking the right tool, generating the response, and speaking it back. All of that happens over a **single bidirectional WebSocket stream** — no separate STT, LLM, or TTS APIs.

The project adds four financial tool backends and an event router that connects them to Nova Sonic.

**What Nova Sonic manages:**
- Speech-to-text (ASR)
- Language understanding + tool selection (LLM)
- Text-to-speech (TTS)
- Voice Activity Detection + mid-sentence interruption (VAD)

**What this project adds:**
- Four financial tool backends (market data, SEC filings, Monte Carlo, vault logger)
- FastAPI Event Router connecting Nova Sonic to those backends
- React + Vite Web UI with real-time audio streaming

---

## Architecture

```
Browser (microphone) — React + Web Audio API
        │  raw PCM-16 @ 16 kHz (binary WebSocket frames)
        ▼
┌──────────────────────────────────────┐
│  FastAPI  /ws/voice  (WebSocket)     │  ← main.py + event_router/router.py
│  NovaSonicSession manages the pipe  │
└──────────────────┬───────────────────┘
                   │ bidirectional boto3 stream
                   ▼
        ┌──────────────────────┐
        │  AWS Bedrock         │
        │  Nova Sonic 2        │  STT → LLM → TTS (all managed by AWS)
        │  (nova-sonic-v1:0)   │  VAD + interruption (built-in)
        └──────────┬───────────┘
                   │ tool call JSON event
                   ▼
        ┌─────────────────────────────────────┐
        │  Event Router  (event_router/)      │
        │  dispatch(tool_name, input) → dict  │
        └──┬──────────┬───────────┬───────────┘
           │          │           │           │
           ▼          ▼           ▼           ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Tool 1   │ │ Tool 2   │ │ Tool 3   │ │ Tool 4   │
   │ market   │ │ sec_rag  │ │ quant_   │ │ vault_   │
   │ _data.py │ │ .py      │ │ model.py │ │ logger.py│
   └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**WebSocket event flow (one turn):**

```
Browser  →  startAudio (JSON control)
Browser  →  PCM audio frames (binary)
Browser  →  endAudio (JSON control)
Nova     →  user_transcript (JSON)
Nova     →  tool_result (JSON)
Nova     →  PCM audio frames (binary, TTS output)
Nova     →  response_complete (JSON)
```

---

## The 4 Financial Tools

| # | Tool | Trigger phrase | Data source | Fallback |
|---|------|----------------|-------------|---------|
| 1 | `query_live_market_data` | *"What's the price of AMD?"* | Finnhub real-time | Polygon EOD |
| 2 | `analyze_sec_filings_rag` | *"What did Nvidia say about supply chains in their 10-K?"* | AWS Bedrock Knowledge Base | Local FAISS index |
| 3 | `execute_quantitative_model` | *"Run a Monte Carlo on NVDA for 30 days."* | Tiingo (volatility) + Finnhub (price) | Polygon |
| 4 | `log_research_insight` | *"Save this to my vault, tag it semiconductors."* | Local `vault/` directory + Groq LLM | Structural template |

**Example Tool 3 output (spoken back by Nova):**

```json
{
  "ticker": "NVDA",
  "days": 30,
  "current_price": 875.5,
  "p10": 820.12,
  "p50": 878.44,
  "p90": 941.33,
  "mean": 879.11,
  "simulation_time_seconds": 0.03
}
```

Tool 4 writes **Obsidian-compatible Markdown notes** with full YAML front matter, structured sections (Executive Summary, Evidence, Key Takeaways, Risks, Next Steps), and session context embedded — composed by a Groq LLM call.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Voice AI | AWS Bedrock Nova Sonic 2 (`amazon.nova-sonic-v1:0`) |
| Backend | Python 3.11+, FastAPI, uvicorn, asyncio |
| AWS SDK | boto3 (`InvokeModelWithBidirectionalStream`) |
| Market data | Finnhub (primary), Polygon.io (fallback) |
| SEC filings | AWS Bedrock Knowledge Base + FAISS (local fallback) |
| Quant | NumPy — 10k GBM paths in ~30ms |
| Note LLM | Groq (`llama-3.3-70b-versatile`) |
| Frontend | React 19, Vite 8, TypeScript 5.9, Tailwind CSS 4, Framer Motion |
| Audio (browser) | Web Audio API + AudioWorklet (no native deps) |

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for the frontend)
- **AWS account** with Bedrock access enabled in `us-east-1`
  - Nova Sonic 2 model access approved in the Bedrock console
- **Polygon.io API key** (required — fallback market data)
- **Finnhub API key** (recommended — primary real-time quotes)
- **Groq API key** (recommended — vault note composition)
- A browser with microphone access (Chrome / Edge recommended)

---

## Setup & Running

### 1. Clone & Configure

```bash
git clone https://github.com/abandonedmonk/AskAlpha-Nova-Sonic-Financial-Research-Analyst.git
cd AskAlpha-Nova-Sonic-Financial-Research-Analyst
```

Copy the environment template and fill in your credentials:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set at minimum:

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

POLYGON_API_KEY=your_polygon_key

# Recommended
FINNHUB_API_KEY=your_finnhub_key
GROQ_API_KEY=your_groq_key

# Optional — for SEC filing search
BEDROCK_KB_ID=your_kb_id

# Optional — for historical volatility in Tool 3
TIINGO_API_KEY=your_tiingo_key
```

> All other settings (`NOVA_SONIC_MODEL_ID`, `VAULT_PATH`, etc.) have sensible defaults — see `.env.example` for the full annotated list.

---

### 2. Run the Backend

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server starts at **`http://localhost:8000`**.

Verify it's healthy:

```bash
curl http://localhost:8000/health
# → {"status":"ok","tools":["query_live_market_data","analyze_sec_filings_rag","execute_quantitative_model","log_research_insight"]}
```

**Alternative (with auto-reload for development):**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### 3. Run the Frontend

In a **second terminal** (keep the backend running):

```bash
cd frontend
npm install
npm run dev
```

The UI is available at **`http://localhost:5173`**.

**Using the UI:**

1. Open `http://localhost:5173` in Chrome or Edge
2. Allow microphone access when prompted
3. Click the **mic button** and speak your question
4. Click mic again (or just stop speaking) to send
5. Nova Sonic will respond — spoken aloud and shown as text
6. The right panel shows the raw tool output
7. The vault panel lists saved research notes; click any to read it

> **Note:** The backend must be running before you open the UI. The frontend connects to `ws://localhost:8000/ws/voice` automatically on page load.

---

### Optional: Build SEC Filing Index Locally

If you don't have a Bedrock Knowledge Base, you can build a local FAISS index from PDFs:

```bash
pip install llama-index faiss-cpu pypdf

# Drop 10-K PDFs into data/sec_filings/
# e.g. nvidia-2024-10k.pdf, amd-2024-10k.pdf (from SEC EDGAR)

python data/build_local_index.py
```

Leave `BEDROCK_KB_ID` empty in `.env` and the SEC tool will use this index automatically.

---

## Project Structure

```
Voice_AI_Agent/
│
├── main.py                      # FastAPI app entry point
├── config.py                    # Pydantic settings (reads .env)
├── requirements.txt
├── pytest.ini                   # asyncio_mode = auto
├── .env.example
│
├── frontend/                    # React + Vite + TypeScript Web UI
│   ├── src/
│   │   ├── App.tsx              # Root: routes / and /vault/:filename
│   │   └── components/          # Panel, VoiceVisualizer, VaultViewer, etc.
│   ├── package.json
│   └── vite.config.ts
│
├── nova_sonic/                  # AWS Bedrock / Nova Sonic layer
│   ├── client.py                # boto3 stream wrapper + event builders
│   ├── session.py               # Per-connection state machine
│   └── tool_schemas.py          # JSON tool schemas injected at session start
│
├── event_router/                # Dispatch layer
│   ├── router.py                # WebSocket + REST endpoints + dispatch()
│   └── schemas.py               # Pydantic I/O models
│
├── tools/                       # Financial tool backends
│   ├── market_data.py           # Tool 1 — Finnhub + Polygon
│   ├── sec_rag.py               # Tool 2 — Bedrock KB or FAISS
│   ├── quant_model.py           # Tool 3 — Monte Carlo
│   └── vault_logger.py          # Tool 4 — Markdown notes
│
├── compute/
│   └── monte_carlo.py           # GBM simulator (NumPy + pure-Python fallback)
│
├── data/
│   ├── build_local_index.py     # One-shot FAISS builder
│   └── sec_filings/             # Drop PDFs here
│
├── tests/                       # Unit + integration tests
├── vault/                       # Runtime: saved research notes
└── docs/                        # PROJECT_OVERVIEW.md and technical deep dives
```

---

## Testing

### Unit tests (no AWS credentials needed)

```bash
pytest tests/ -v
```

Covers: `NovaSonicClient` event builders, `NovaSonicSession` state machine, all four tool backends, and the event router's dispatch logic. All AWS and external API calls are mocked.

### Live integration tests (requires real credentials)

```bash
python tests/smoke_test_tools.py
```

Makes real network calls to Finnhub, Polygon, Bedrock KB, and Groq. Tests all four tools end-to-end including cache hits and fallback paths.

### Full audio + tool test (requires running server + microphone)

```bash
python tests/test_audio_with_tools.py
```

### WebSocket debug (requires running server)

```bash
python tests/test_web_ui_debug.py
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws/voice` | Bidirectional audio stream (PCM in, PCM out + JSON events) |
| `GET` | `/health` | Liveness check — lists all four tool names |
| `POST` | `/vault/log` | Direct vault write (bypasses voice pipeline) |
| `GET` | `/vault/files` | List all saved vault notes |
| `GET` | `/vault/files/{filename}` | Read a specific vault note |

---

## License

Submitted to the **Amazon Nova Hackathon** on Devpost.
