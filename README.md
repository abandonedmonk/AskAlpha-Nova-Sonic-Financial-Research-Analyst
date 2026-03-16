# Voice AI Agent with Nova Sonic 2

A voice-driven financial research assistant powered by AWS Bedrock's Nova Sonic 2 model with real-time tool calling and audio I/O streaming.

**[Devpost Submission](https://amazon-nova.devpost.com/?_gl=1*11d2kzb*_gcl_au*MTgwMjE3MDQzMS4xNzcwODM1OTMw*_ga*MzUxODYxMDk1LjE3NzA4MzU5MzE.*_ga_0YHJK3Y10M*czE3NzM0Mjk5ODgkbzgkZ0kZzAkdDE3NzM0Mjk5ODgkajYwJGwwJGgw)**

---

## 🎯 Overview

This project demonstrates **AWS Bedrock's Nova Sonic 2** handling bidirectional audio streaming with sophisticated tool calling. Users speak natural language queries, and the AI:

- Transcribes speech using Nova's built-in ASR
- Understands intent and selects the right tool
- Executes financial tools locally (Python async)
- Returns results to Nova for response generation
- Synthesizes audio responses in real-time

The entire workflow happens **on a single bidirectional WebSocket-like stream** — no juggling multiple APIs.

---

## 🏗️ Architecture

### High-Level Flow

```
User Voice Input (16kHz PCM)
    ↓
[AWS Bedrock - Nova Sonic 2]
    ├─ ASR Recognition (speech → text)
    ├─ LLM Understanding (intent detection)
    └─ Tool Selection (which financial tool to call?)
    ↓
[Local Tool Execution - Python Async]
    ├─ Tool 1: query_live_market_data
    ├─ Tool 2: analyze_sec_filings_rag (KB retrieval)
    ├─ Tool 3: execute_quantitative_model (Monte Carlo)
    └─ Tool 4: log_research_insight (vault storage)
    ↓
[Result Return to Nova]
    └─ Send tool output via event stream
    ↓
[Nova Response Generation]
    ├─ LLM synthesizes response text
    └─ TTS synthesizes audio output
    ↓
Voice Output (24kHz PCM + Base64)
    └─ Speaker Playback
```

### Key Innovation: Bidirectional Streaming

The architecture uses **AWS Bedrock's `InvokeModelWithBidirectionalStream` API**:
- Single persistent connection (no request/response cycles)
- Event-driven protocol (JSON events sent/received continuously)
- Audio, text, and tool calls on the same stream
- Async Python event loop processes responses while capturing audio

---

## 🔄 Complete Workflow

### 1. **Session Initialization**
```
Client → sessionStart (inference config)
      → promptStart (audio + tools + system prompt)
      → contentStart (SYSTEM role)
      → textInput (system message)
      → contentEnd
```

### 2. **User Speech Input**
```
Client → contentStart (USER, AUDIO role)
      → audioInput chunks (base64-encoded PCM, 16kHz)
      → contentEnd (signals end of speech)
```

### 3. **Nova Processing & Tool Detection**
```
Nova → completionStart
    → contentStart (TOOL role)
    → textOutput (ASR transcript)
    → toolUse event (tool_name, inputs)
    → contentEnd
```

### 4. **Local Tool Execution**
```
Client detects toolUse event
     → Extract tool name & inputs
     → dispatch(tool_name, inputs) — async Python execution
     → Get result
     → contentStart (TOOL role)
     → toolResult (JSON result)
     → contentEnd
```

### 5. **Response Generation & Audio**
```
Nova processes tool result
   → contentStart (ASSISTANT)
   → textOutput (response text)
   → contentStart (AUDIO)
   → audioOutput chunks (24kHz PCM, base64)
   → completionEnd
```

---

## 🛠️ The 4 Financial Tools

### **Tool 1: query_live_market_data**
**Purpose:** Get real-time stock prices and market metrics

```python
Input:  {"ticker": "AMD"}
Output: {
  "ticker": "AMD",
  "price": 194.13,
  "change": "-0.45%",
  "volume": 45230000
}
```

**Data Sources:** Finnhub API (primary) + Polygon fallback

---

### **Tool 2: analyze_sec_filings_rag**
**Purpose:** Search SEC 10-K/10-Q filings using Retrieval-Augmented Generation

```python
Input:  {
  "company": "AMD",
  "topic": "revenue growth",
  "filing_type": "10-Q"
}
Output: {
  "passages": [
    "Data Center net revenue reached $16.6 billion..."
  ],
  "sources": ["AMD Form 10-Q 2025"]
}
```

**Data Source:** AWS Bedrock Knowledge Base with SEC documents

**Smart ASR Normalization:** Automatically corrects speech-to-text errors:
- "tenk" → "10-K" | "tenq" → "10-Q"
- Matches company names from transcript context

---

### **Tool 3: execute_quantitative_model**
**Purpose:** Run Monte Carlo simulations for price forecasting

```python
Input:  {
  "ticker": "AMD",
  "days": 30,
  "simulations": 10000
}
Output: {
  "current_price": 194.13,
  "p10": 146.33,    # 10th percentile
  "p50": 190.48,    # median
  "p90": 248.86,    # 90th percentile
  "mean": 195.00
}
```

**Engine:** NumPy-based geometric Brownian motion simulation
**Performance:** Sub-millisecond computation for 10K simulations

---

### **Tool 4: log_research_insight**
**Purpose:** Save analysis notes to personal vault with tags

```python
Input:  {
  "content": "AMD is volatile despite strong data center growth",
  "tags": ["semiconductors", "amd", "analysis"],
  "title": "AMD Market Analysis"
}
Output: {
  "saved": true,
  "filepath": "vault/Research_Insight_-_2026-03-14.md"
}
```

**Storage:** Local markdown files in `vault/` directory
**Format:** Timestamped, searchable research notes

---

## How Nova Sonic 2 Powers Everything

### **Why Nova Sonic 2?**

1. **Native Audio I/O** - Built-in speech recognition + synthesis, no external APIs
2. **Streaming Foundation** - Handles bidirectional audio natively
3. **Tool Calling** - Understands custom tool schemas and selects appropriate tools
4. **Low Latency** - Sonic model optimized for voice conversations
5. **Cost-Effective** - Cheaper than multi-model pipelines

### **Nova Sonic's Role in This Project**

| Stage | Nova's Function |
|-------|-----------------|
| **ASR** | Converts user speech → text (on device, real-time) |
| **NLU** | Understands intent, extracts parameters |
| **Tool Selection** | Decides which of the 4 tools to call |
| **Response Gen** | Synthesizes response text from tool results |
| **TTS** | Converts response → 24kHz audio stream |

### **Tool Schema Configuration**

Nova discovers tools via `toolConfiguration` in API:

```python
"toolConfiguration": {
  "tools": [
    {
      "toolSpec": {
        "name": "execute_quantitative_model",
        "description": "Run Monte Carlo simulations",
        "inputSchema": {
          "json": json.dumps({
            "type": "object",
            "properties": {
              "ticker": {"type": "string"},
              "days": {"type": "integer"}
            }
          })
        }
      }
    }
    # ... 3 more tools
  ],
  "toolChoice": {"auto": {}}  # Nova auto-selects when needed
}
```

---

## 📊 Tech Stack

### **Core Technologies**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM/ASR/TTS** | AWS Bedrock - Nova Sonic 2 | Speech recognition, understanding, synthesis |
| **Streaming** | Python `asyncio` | Concurrent audio capture/playback + event processing |
| **Tool Dispatch** | Custom router (event_router/) | Maps tool calls to Python implementations |
| **Market Data** | Finnhub API | Real-time stock prices |
| **SEC Filings** | AWS Bedrock Knowledge Base | RAG-based document retrieval |
| **Quantitative** | NumPy | Monte Carlo simulations |
| **Audio I/O** | PyAudio | Microphone capture, speaker playback |
| **API Client** | AWS SDK (Bedrock Runtime) | Bidirectional streaming API |

### **Languages & Frameworks**

- **Python 3.10+** - Entire backend
- **FastAPI** (optional) - REST API wrapper
- **pytest** - Unit testing
- **asyncio** - Async I/O throughout

### **Audio Specifications**

- **Input:** 16 kHz, 16-bit PCM, mono (speech quality)
- **Output:** 24 kHz, 16-bit PCM, mono (natural synthesis)
- **Transport:** Base64-encoded over JSON events

---

## 🎯 Features

✅ **Real-time Voice Commands**  
✅ **Multi-Tool Intelligence** - Tool selection + execution  
✅ **SEC Filing Search** - RAG with AWS Knowledge Base  
✅ **Monte Carlo Simulations** - Fast quantitative analysis  
✅ **Research Vault** - Tag-based note storage  
✅ **Smart ASR Correction** - Context-aware input normalization  
✅ **Async Architecture** - Non-blocking event processing  
✅ **Streaming Audio** - Continuous I/O without buffering  

---

## Quick Start

### **Prerequisites**
- Python 3.10+
- AWS Account with Bedrock access (Nova Sonic 2)
- Microphone & speakers

### **Setup**

```bash
# Clone repo
git clone https://github.com/abandonedmonk/Voice_AI_Agent_Nova.git
cd Voice_AI_Agent_Nova

# Create virtual environment
python -m venv nova-env
source nova-env/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Fill in:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   BEDROCK_KB_ID (for SEC filings)
#   POLYGON_API_KEY (backup market data)
#   FINNHUB_API_KEY (primary market data)
```

### **Run Integration Test**

```bash
python tests/test_audio_with_tools.py
```

**Expected Output:**
```
================================================================================
SESSION READY - AWAITING USER INPUT
================================================================================

🎤 Speak now...

[TOOL] query_live_market_data
  Input: {"ticker": "AMD"}
  Retrieved: {"price": 194.13, "change": "-0.45%", ...}

[RESPONSE] The current price of AMD is $194.13, down 0.45%.
[AUDIO] Generating synthesized speech...

================================================================================
TEST SUMMARY
================================================================================
Tools called: 1
Audio chunks received: 305
  ✓ query_live_market_data
```

---

## 📂 Project Structure

```
Voice_AI_Agent_Nova/
├── nova_sonic/
│   ├── client.py          # Bedrock API wrapper
│   ├── session.py         # Session lifecycle
│   ├── tool_schemas.py    # Tool definitions for Nova
│   └── __init__.py
├── tools/                 # Tool implementations
│   ├── market_data.py     # Tool 1: Stock prices
│   ├── sec_rag.py         # Tool 2: SEC filings
│   ├── quant_model.py     # Tool 3: Monte Carlo
│   └── vault_logger.py    # Tool 4: Note saving
├── event_router/
│   ├── router.py          # Tool discovery & dispatch
│   ├── schemas.py         # Event type definitions
│   └── __init__.py
├── tests/
│   ├── test_audio_with_tools.py  # Main integration test
│   ├── test_quant_model.py
│   └── ...
├── docs/
│   ├── HOW_NOVA_SONIC_TOOLS_WORK.md  # Deep dive
│   ├── BIDIRECTIONAL_AUDIO_TEST.md
│   └── ...
├── vault/                 # User research notes
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

### **Unit Tests**
```bash
pytest tests/ -v
```

### **Integration Test (Audio + Tools)**
```bash
python tests/test_audio_with_tools.py
```

### **Manual Testing**

To test specific tools, see test files in `tests/`:
- `test_nova_sonic_client.py` - Bedrock connection
- `test_quant_model.py` - Monte Carlo accuracy
- `test_market_data.py` - API integration
- `test_vault_logger.py` - File I/O

---

## 📚 Documentation

Comprehensive guides in `docs/`:

- **[Nova_Sonic_implementation_with_tools.md](docs/Nova_Sonic_implementation_with_tools.md)** - Full technical deep dive  
- **[BIDIRECTIONAL_AUDIO_TEST.md](docs/BIDIRECTIONAL_AUDIO_TEST.md)** - Event protocol details  
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Architecture & design decisions

---

## 🎨 User Interface

The application provides a voice-first interface with real-time feedback:

- **Chat Session Panel** - Conversation history
- **Query Display** - Real-time tool results shown
- **Voice Waveform** - Visual feedback during speech
- **Vault Logger** - Quick-access research notes

---

## 🔐 Security & Best Practices

- ✅ Environment variables for all credentials (`.env`)
- ✅ Async/await for non-blocking operations
- ✅ Error handling for API failures (fallbacks)
- ✅ Input validation for tool parameters
- ✅ Structured logging for debugging

---

## 🚀 Future Enhancements

- [ ] Multi-turn conversations (context memory)
- [ ] Custom voice profiles (more voice options)
- [ ] Extended tool ecosystem (earnings calendar, options chains)
- [ ] Persistent chat history with SQLite
- [ ] Web UI dashboard
- [ ] Deployment to AWS Lambda/Container Apps

---

## 📄 License

This project is submitted to the **Amazon Nova Hackathon** on Devpost.

---

## 🤝 Team & Credits

Built with AWS Bedrock Nova Sonic 2 for the Amazon Nova Hackathon.

**Key Technologies:**
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Nova Sonic 2](https://www.anthropic.com/)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)

---

## 📞 Contact & Support

For questions about this project, see the documentation or visit the [Devpost page](https://amazon-nova.devpost.com/).

**Questions about Nova Sonic 2 integration?** Check [HOW_NOVA_SONIC_TOOLS_WORK.md](docs/HOW_NOVA_SONIC_TOOLS_WORK.md) for a complete technical walkthrough.
