# Nova Sonic 2 Tool Calling Works with Voice I/O

## Overview

This document explains how AWS Bedrock's Nova Sonic 2 model integrates with custom financial tools through bidirectional audio streaming. The system allows voice commands to trigger tool execution and return synthesized audio responses.

**What You Get:**
- ✅ Voice-triggered tool calling (4 financial tools)
- ✅ Natural language understanding via ASR
- ✅ Real-time tool execution (market data, SEC filings, Monte Carlo simulations, vault logging)
- ✅ Synthesized audio responses from Nova

---

## Architecture Overview

```
User Speech
    ↓
[Audio Input Stream] → 16kHz PCM mono
    ↓
Nova Sonic 2 [ASR + LLM]
    ↓
[Tool Detection] → Identifies which tool to call
    ↓
[Local Tool Execution] → Python async functions
    ↓
[Tool Result] → Sent back to Nova
    ↓
Nova [Response Generation] → Text + Audio synthesis
    ↓
[Audio Output Stream] ← 24kHz PCM base64-encoded
    ↓
Speaker Output
```

---

## What Are the 4 Tools?

### 1. **query_live_market_data** 📈
**Purpose:** Get real-time stock prices and market data
```python
Input:  {"ticker": "AMD"}
Output: {"price": 194.13, "change": "-0.45%", "volume": 45230000}
```
**Data Sources:** Finnhub API (primary) + Polygon fallback

### 2. **analyze_sec_filings_rag** 📄
**Purpose:** Search SEC 10-K/10-Q filings with RAG (Retrieval-Augmented Generation)
```python
Input:  {"company": "AMD", "topic": "revenue", "filing_type": "10-Q"}
Output: 
{
  "passages": [
    "Data Center net revenue of $16.6 billion..."
  ],
  "sources": ["AMD Form 10-Q 2025"]
}
```
**Data Source:** AWS Bedrock Knowledge Base (YNEMN3KE...) with SEC documents

### 3. **execute_quantitative_model** 📊
**Purpose:** Run Monte Carlo simulations for price forecasting
```python
Input:  {"ticker": "AMD", "days": 30, "simulations": 10000}
Output:
{
  "current_price": 194.13,
  "p10": 146.33,
  "p50": 190.48,
  "p90": 248.86,
  "mean": 195.00
}
```
**Calculation:** NumPy-based Monte Carlo with geometric Brownian motion

### 4. **log_research_insight** 💾
**Purpose:** Save analysis notes to vault with tags
```python
Input:  {"content": "AMD is volatile", "tags": ["semiconductors", "amd"]}
Output: {"saved": true, "filepath": "vault/Research_Insight_-_2026-03-13.md"}
```
**Storage:** Local markdown files in `vault/` directory

---

## The Bidirectional Streaming Protocol

Nova Sonic 2 uses **event-based bidirectional streaming** over AWS Bedrock's `InvokeModelWithBidirectionalStream` API. Here's the message flow:

### Session Initialization

```plaintext
[Client]                                    [Nova Sonic 2]
    |
    |-- sessionStart (inference config) ----→
    |
    |-- promptStart ----------------------→  (with toolConfiguration + audio I/O config)
    |
    |-- contentStart (SYSTEM role) --------→  (system prompt)
    |
    |-- textInput ------------------------→  (system message)
    |
    |-- contentEnd ----------------------→
    |
    ← Ready for user input
```

### User Input & Tool Calling

```plaintext
[Client - Audio Input]              [Nova Sonic 2]
    |
    |-- contentStart (USER, AUDIO) --→
    |
    |-- audioInput chunks ----------→  [ASR Recognition]
    |                                   
    |-- contentEnd ----------------→  [Query Understanding]
    |                                   [Tool Selection]
    |
    ← toolUse event (tool_name, input)
    |
    [Local Execution]
    |
    |-- contentStart (TOOL role) ---→  
    |-- toolResult (content) ------→  [Response Generation Triggered]
    |-- contentEnd ----------------→
    |
    ← contentStart (ASSISTANT, AUDIO)
    ← textOutput (transcribed text)
    ← audioOutput chunks (synthesized voice)
    ← completionEnd
```

### Key Protocol Details

1. **Tool Configuration** (in `promptStart`)
   ```json
   "toolConfiguration": {
     "tools": [
       {
         "toolSpec": {
           "name": "execute_quantitative_model",
           "description": "Run Monte Carlo simulations",
           "inputSchema": {
             "json": "{\"type\": \"object\", \"properties\": {...}}"
           }
         }
       }
     ],
     "toolChoice": {"auto": {}}
   }
   ```

2. **Audio Configuration** (in `promptStart`)
   ```json
   "audioOutputConfiguration": {
     "mediaType": "audio/lpcm",
     "sampleRateHertz": 24000,
     "sampleSizeBits": 16,
     "channelCount": 1,
     "voiceId": "matthew",
     "encoding": "base64"
   }
   ```

3. **Tool Result Format** (critical!)
   ```json
   {
     "event": {
       "contentStart": {
         "promptName": "...",
         "contentName": "...",
         "type": "TOOL",
         "role": "TOOL",
         "toolResultInputConfiguration": {
           "toolUseId": "...",
           "type": "TEXT",
           "textInputConfiguration": {"mediaType": "text/plain"}
         }
       }
     }
   }
   ⇒ THEN toolResult event
   ⇒ THEN contentEnd event
   ```

---

## Code Architecture

### 1. **Test Harness** (`tests/test_audio_with_tools.py`)
Main integration test that orchestrates the entire flow:

```python
class AudioToolsTest:
    ├── start_session()           # Initialize session with tools
    ├── start_audio_input()       # Begin recording
    ├── capture_audio()           # Async microphone capture
    ├── _process_responses()      # Event loop (reads Nova events)
    ├── _handle_tool_use()        # Execute tool + send result back
    ├── wait_for_turn_end()       # Detect when response completes
    ├── play_audio()              # Stream audio output
    └── close()                   # Cleanup & summary
```

**Flow:**
```
main()
  → start_session()        # Setup Nova + tools
  → capture_audio()        # Record 15s user speech (async)
  → play_audio()           # Play Nova's responses (async)
  → wait_for_turn_end()    # Wait for response completion
  → close()                # Summary + cleanup
```

**Key State Tracking:**
- `tool_requested` - Tool was triggered
- `response_completed` - Nova finished generating
- `tool_result_sent` - We sent result back to Nova
- `audio_chunks_received` - Count of audio packets from Nova
- `last_event_ts` - Timestamp of last event (quiescence detection)

### 2. **Tool Schemas** (`nova_sonic/tool_schemas.py`)
Defines tool metadata for Nova to discover:

```python
ALL_TOOLS = [
  {
    "toolSpec": {
      "name": "query_live_market_data",
      "description": "Get current stock price and market data",
      "inputSchema": {
        "json": json.dumps({
          "type": "object",
          "properties": {
            "ticker": {"type": "string", "description": "Stock ticker"}
          },
          "required": ["ticker"]
        })
      }
    }
  },
  # ... (3 more tools)
]
```

**Critical:** `inputSchema.json` must be a **JSON string**, not a dict object!

### 3. **Tool Dispatch** (`event_router/router.py`)
Routes tool names to Python implementations:

```python
async def dispatch(tool_name: str, tool_input: dict):
    if tool_name == "query_live_market_data":
        return await get_market_snapshot(tool_input["ticker"])
    elif tool_name == "analyze_sec_filings_rag":
        return await query_sec_filings(...)
    elif tool_name == "execute_quantitative_model":
        return await run_monte_carlo(...)
    elif tool_name == "log_research_insight":
        return await log_insight(...)
```

### 4. **Tool Implementations** (`tools/` directory)
Each tool is a standalone Python module:

```
tools/
  ├── market_data.py          # get_market_snapshot() → Finnhub API
  ├── sec_rag.py              # query_sec_filings() → Bedrock KB
  ├── quant_model.py          # run_monte_carlo() → NumPy simulation
  └── vault_logger.py         # log_insight() → File I/O
```

---

## Event Flow Example: "What's the price of AMD?"

### Step 1: User speaks
```
🎤 "What is the current price of AMD?"
```

### Step 2: Nova receives audio
```plaintext
Client → audioInput chunks (base64-encoded PCM)
Nova processes → ASR converts to text
Nova recognizes → "What is the current price of AMD?"
```

### Step 3: Nova selects tool
```plaintext
Nova LLM → "This is a market data query"
Nova detects → query_live_market_data tool is needed
Nova extracts → {"ticker": "AMD"}
```

### Step 4: Client receives tool request
```json
{
  "event": {
    "toolUse": {
      "toolName": "query_live_market_data",
      "content": "{\"ticker\": \"AMD\"}",
      "toolUseId": "5c16e08a-c70a-440b..."
    }
  }
}
```

### Step 5: Client executes tool
```python
result = await dispatch("query_live_market_data", {"ticker": "AMD"})
# Returns: {"price": 194.13, "change": "-0.45%", ...}
```

### Step 6: Client sends result back
```python
# contentStart → toolResult → contentEnd
client.send_event(json.dumps({
  "event": {
    "contentStart": {
      "role": "TOOL",
      "type": "TOOL",
      "toolResultInputConfiguration": {"toolUseId": "5c16e08a..."}
    }
  }
}))
client.send_event(json.dumps({
  "event": {
    "toolResult": {
      "content": json.dumps({"price": 194.13, ...})
    }
  }
}))
client.send_event(json.dumps({"event": {"contentEnd": {...}}}))
```

### Step 7: Nova generates response
```plaintext
Nova processes tool result
Nova generates → "The current price of AMD is $194.13, down 0.45%"
Nova synthesizes → Audio (24kHz PCM, 511 chunks)
```

### Step 8: Client receives & plays audio
```plaintext
Client ← audioOutput chunk 1 (base64 PCM)
Client ← audioOutput chunk 2
Client ← ... (511 total)
Client ← completionEnd
↓
Speaker 🔊 "The current price of AMD is $194.13, down 0.45%"
```

---

## Audio Format Details

### Input Audio
- **Sampling Rate:** 16,000 Hz
- **Bit Depth:** 16-bit (signed)
- **Channels:** 1 (mono)
- **Format:** PCM WAV
- **Encoding:** Base64 when sent over API

### Output Audio
- **Sampling Rate:** 24,000 Hz
- **Bit Depth:** 16-bit (signed)
- **Channels:** 1 (mono)
- **Format:** LPCM (Linear PCM)
- **Encoding:** Base64 in API response
- **Voice ID:** "matthew" (configurable)

**Note:** Output audio has higher sample rate for better speech quality.

---

## How the Code Handles Async Tool Execution

The system is fully **asynchronous** to prevent blocking the response stream:

```python
async def _handle_tool_use(self, tool_event):
    # Execute tool in background (don't wait here!)
    self.pending_tool_task = asyncio.create_task(
        self._dispatch_and_send_result(tool_event)
    )
    # Return immediately to keep processing events
    
async def _dispatch_and_send_result(self, tool_event):
    # Execute tool
    result = await dispatch(tool_name, tool_input)
    
    # Send result back to Nova
    await self.send_event(toolResult_event)
    
    # Mark as complete
    self.tool_result_sent.set()
```

**Why?** If we wait for the tool to finish before processing the next event, we might miss important Nova responses. With async, both happen concurrently.

---

## Response Generation Trigger

The **critical insight** that makes audio work:

1. After sending `contentEnd` for the tool result, Nova automatically starts generating its response
2. No explicit "resume" or "continue" signal needed
3. Nova streams back: `contentStart (ASSISTANT)` → `textOutput` → `audioOutput` chunks → `completionEnd`
4. We must NOT send `promptEnd` during tool flow (only after all tools complete)

---

## Robustness Features

### 1. **Input Quality Normalization** (for SEC filings)
```python
# ASR sometimes returns "tenk" instead of "10-K"
if '10k' in transcript or 'tenk' in transcript:
    filing_type = '10-K'
```

### 2. **Quiescence-Based Turn Detection**
Instead of waiting for explicit `completionEnd`, we detect end-of-turn by stream inactivity:
```python
async def wait_for_turn_end(self, timeout=60, quiet_seconds=2.0):
    """Wait until no events received for 2 seconds AND all tasks done"""
    while not timeout_exceeded:
        if no_pending_tasks and no_recent_events:
            return True  # Turn is complete
```

### 3. **Async Tool Execution**
Tools run in separate async tasks, never blocking the event loop:
```python
self.pending_tool_task = asyncio.create_task(execute_tool())
# Continue processing events while tool runs
```

---

## Testing & Validation

### Run the Full Integration Test
```bash
cd Voice_AI_Agent_Nova
python tests/test_audio_with_tools.py
```

### Example Voice Commands
```
✓ "What is the current price of AMD?"
  → Tool: query_live_market_data
  → Output: 😄 Audio with price + market move

✓ "Run a Monte Carlo simulation on AMD for 30 days"
  → Tool: execute_quantitative_model
  → Output: 😄 Audio with price distribution

✓ "What did AMD say about revenue in its recent earnings?"
  → Tool: analyze_sec_filings_rag
  → Output: 😄 Audio with SEC filing excerpts

✓ "Save this note: AMD is volatile. Tag it semiconductors and amd"
  → Tool: log_research_insight
  → Output: 😄 Audio confirmation + note saved
```

### Expected Results (from test.txt)
```
Tool 1 (Market Data):      ✅ Audio chunks received: 305
Tool 2 (SEC Filings):      ✅ Audio chunks received: 612
Tool 3 (Monte Carlo):      ✅ Audio chunks received: 511
Tool 4 (Vault Logging):    ✅ Audio chunks received: 163
```

---

## Configuration Files

### `.env` - API Credentials
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_KB_ID=YNEMN3KE...
FINNHUB_API_KEY=...
POLYGON_API_KEY=...
```

### `nova_sonic/tool_schemas.py` - Tool Definitions
Contains `ALL_TOOLS` list with schema for each tool. Nova uses this to understand what tools exist.

### `event_router/router.py` - Dispatcher
Maps `toolName` → Python implementation. When Nova calls tool, this routes to correct handler.

---

## Key Takeaways for Blog Post

1. **Nova Sonic 2 is a Foundation Model** - It doesn't natively know about custom tools. We teach it via `toolConfiguration` in the API.

2. **Event-Driven, Not Request-Response** - The bidirectional stream is a persistent WebSocket-like connection with discrete JSON events, not traditional HTTP requests.

3. **Three-Phase Tool Workflow:**
   - Phase 1: Nova selects tool (sends `toolUse` event)
   - Phase 2: We execute tool locally (Python async)
   - Phase 3: We send result back, Nova generates response

4. **Audio is Transparent** - Input/output audio handling is just another content type in the same event stream. No separate API calls needed.

5. **Async is Essential** - Event processing and tool execution must be concurrent, or you'll miss events while waiting for tool results.

6. **Turn Management** - Knowing when a "turn" (user input → assistant response) is complete requires monitoring event quiescence, not just explicit markers.

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| No audio output after tool | Tool result format wrong | Use contentStart(role=TOOL) → toolResult → contentEnd pattern |
| Tool not being called | inputSchema is dict, not JSON string | Wrap with `json.dumps()` |
| False timeout errors | Waiting for explicit event instead of quiescence | Use `wait_for_turn_end()` with quiet_seconds threshold |
| SEC filing search empty | Low ASR quality (e.g., "tenk" vs "10-K") | Add input normalization for filing_type |
| Audio playback cuts off | promptEnd sent too early during tool flow | Skip promptEnd if `tool_requested.is_set()` |

---

## Next Steps

1. **Extend Tools:** Add more financial tools (earnings calendar, analyst ratings, options chains)
2. **Multi-Turn Conversations:** Handle follow-up questions without re-listening
3. **Custom Voice:** Use Amazon Polly with different voice options
4. **Streaming SDK:** Transition from raw async to AWS SDK wrapper for cleaner code
5. **Deployment:** Package into Lambda or Container Apps for production

---

## References

- [AWS Bedrock Nova Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bidirectional Streaming Specification](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [Tool Calling Best Practices](https://docs.aws.amazon.com/bedrock/latest/user-guide/agents-tools.html)

