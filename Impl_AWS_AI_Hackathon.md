# 🏦 IMPL: Voice-Native Quantitative Research Terminal

**Context:** AWS Bedrock Nova Sonic Hackathon (Devpost). This is a focused pivot from the core Voice AI project.  
**Drops into:** same folder as `README.md`, `ROADMAP.md`, etc.  
**Timeline:** Hackathon sprint — assume 5–7 days of focused work.

---

## What You're Building (One Sentence)

A voice-driven financial research agent: you speak a market question, it orchestrates live ticker APIs, SEC filing RAG, and a Rust/Wasm compute module, then speaks the answer back — and you can interrupt it mid-sentence.

---

## How This Relates to the Core Voice AI Project

This is **not a replacement**. It is the core Voice AI project with a different tool layer dropped on top and AWS Nova Sonic as the voice interface instead of your custom FastAPI + Deepgram + Cartesia stack.

```
Core Voice AI Project          This Hackathon Project
─────────────────────          ──────────────────────
FastAPI WebSocket backend  →   AWS Bedrock / Nova Sonic (managed)
Deepgram STT               →   Nova Sonic (built-in STT)
Groq LLM                   →   Nova Sonic (built-in LLM)
Cartesia TTS               →   Nova Sonic (built-in TTS)
Rust VAD                   →   Nova Sonic (built-in VAD)
LangGraph orchestration    →   Event Router (Python) + Bedrock tool use
─────────────────────────────────────────────────────────────────────
Custom tools (none yet)    →   4 financial tools (this is the new work)
Ironclad Rust/Wasm         →   Reused directly for Monte Carlo compute
```

The hackathon wins you points specifically for connecting Nova Sonic to non-trivial backends. The financial tools **are** the submission. The voice pipeline is handled by AWS.

---

## Overlap With the Core Voice AI Project

Items marked 🔁 are directly reusable. Items marked 🆕 are net-new for this project.

| Component                  | Status               | Notes                                                                                                                                         |
| -------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| WebSocket audio pipeline   | 🔁 Conceptual only   | Nova Sonic manages this. Your FastAPI WebSocket knowledge still helps you understand event routing.                                           |
| VAD / interruption logic   | 🔁 Reused (AWS side) | Nova Sonic has built-in VAD. Your understanding of interrupt states maps directly. "Skip the revenue numbers" works out of the box.           |
| LangGraph ReAct loop       | 🔁 Port directly     | Replace Groq with Bedrock tool-use JSON schema. Same Reasoning → Action → Observation pattern.                                                |
| Rust/Wasm sandbox          | 🔁 **Direct reuse**  | The `ironclad-runtime` from the Ironclad Agent project runs the Monte Carlo module. Zero new Rust code required if Ironclad is already built. |
| Deepgram STT               | ❌ Not needed         | Nova Sonic handles it.                                                                                                                        |
| Cartesia TTS               | ❌ Not needed         | Nova Sonic handles it.                                                                                                                        |
| FastAPI backend            | 🔁 Partial           | You still need a Python Event Router. FastAPI is a fine choice for this.                                                                      |
| `execute_secure_code` tool | 🔁 Adapt             | Becomes `execute_quantitative_model`. Same subprocess call to ironclad-runtime.                                                               |
| Audit log (Ironclad)       | 🔁 Reuse             | Log every model execution for the demo. Shows security awareness to judges.                                                                   |

**Bottom line:** If you've started the core Voice AI project, you already understand 80% of this architecture. The net-new work is the 4 financial tools and the AWS Bedrock wiring.

---

## The 4 Tools — What You're Actually Building

Nova Sonic is given a JSON tool schema. When it detects the right trigger phrase, it fires the tool and reads back the result. Your job is to build the backends these tools call.

### Tool 1: `query_live_market_data`
**Trigger phrase:** "What is the current price / volume / price action for [ticker]?"  
**Backend:** Call Polygon.io REST API (free tier: 5 API calls/min). Fetch `previousClose`, `open`, `high`, `low`, `volume`. Format as structured JSON. Nova Sonic reads it back in natural language.  
**Data shape returned to Nova Sonic:**
```json
{
  "ticker": "AMD",
  "price": 162.45,
  "volume": 48200000,
  "change_pct": "+2.3%",
  "summary": "AMD is trading at $162.45, up 2.3% on volume of 48.2M shares."
}
```

### Tool 2: `analyze_sec_filings_rag`
**Trigger phrase:** "What did [company] say about [topic] in their latest 10-K/10-Q?"  
**Backend:** AWS Bedrock Knowledge Base with a vector store of SEC filings. You pre-ingest 10-K/10-Q PDFs from EDGAR. Query returns the top-3 most relevant paragraphs. Nova Sonic summarizes them aloud.  
**Data source:** SEC EDGAR full-text search API (free, no key needed) or pre-downloaded PDFs.

### Tool 3: `execute_quantitative_model`
**Trigger phrase:** "Run a Monte Carlo simulation on [ticker] for the next [N] days."  
**Backend:** Python passes live price + volatility data to `ironclad-runtime` (Rust/Wasm). The Wasm module runs N=10,000 simulation paths. Returns percentile outcomes (P10, P50, P90).  
**Why Wasm:** Judges see "voice → financial model → Rust/Wasm compute" and immediately understand you're building senior-level architecture. This is your "wow" moment.

### Tool 4: `log_research_insight`
**Trigger phrase:** "Save this / log this / add this to my vault, tag it [topic]."  
**Backend:** Format the last Nova Sonic response as markdown. POST to a local REST endpoint that writes to an Obsidian vault (or just a local `.md` file if Obsidian webhook is complex). Dead simple but surprisingly impressive in a demo.

---

## Step-by-Step Implementation

Difficulty: ⭐ Easy → ⭐⭐⭐⭐⭐ Very Hard

---

### Phase 1 — AWS Setup & Nova Sonic Hello World (Day 1)

#### Step 1 — AWS Account & Bedrock Access `⭐ Easy`
**What:** Create AWS account (free tier). Enable Amazon Bedrock in `us-east-1`. Request access to Nova Sonic (nova-sonic-v1). Confirm access granted in the console.  
**Gotcha:** Nova Sonic access approval can take a few hours. Do this first, before anything else.  
**Time:** ~1 hour (mostly waiting)

#### Step 2 — Nova Sonic "Hello World" `⭐⭐ Medium`
**What:** Run AWS's Nova Sonic sample from their GitHub. Get a basic voice conversation working in your browser. Understand the bidirectional audio stream structure.  
**Resources:** AWS Bedrock Nova Sonic docs + the official demo repo.  
**Milestone:** You can talk to Nova Sonic and it talks back. No tools yet — just voice.  
**Time:** 2–3 hours

#### Step 3 — Understand the Event Router Pattern `⭐⭐ Medium`
**What:** Read how Nova Sonic fires tool calls. When the model decides to use a tool, it emits a structured JSON event. Your backend (Event Router) receives this, calls the right function, and returns the result. Wire a dummy tool (`ping` → returns `"pong"`) end-to-end.  
**Milestone:** Nova Sonic says "I'm calling the ping tool" and you see the event in your Python backend.  
**Time:** 2–3 hours

---

### Phase 2 — Live Market Data (Day 2)

#### Step 4 — Polygon.io API Integration `⭐ Easy`
**What:** Sign up for Polygon.io free tier. Write a Python function `get_market_snapshot(ticker: str) -> dict`. Test it standalone — confirm it returns clean JSON for "AMD", "NVDA", "AAPL".  
**Gotcha:** Free tier is delayed 15 minutes on some endpoints. Use `v2/last/trade/{ticker}` or `v2/snapshot/locale/us/markets/stocks/tickers/{ticker}` for the cleanest data.  
**Time:** 1–2 hours

#### Step 5 — Wire `query_live_market_data` Tool `⭐⭐ Medium`
**What:** Write the Bedrock tool JSON schema for `query_live_market_data`. Register it with Nova Sonic. In your Event Router, parse the tool call, invoke `get_market_snapshot()`, format the result, return it. Test: say "What's the current price of Nvidia?" — Nova Sonic reads back live data.  
**Milestone:** First real financial query works end-to-end via voice.  
**Time:** 2–3 hours

---

### Phase 3 — SEC Filings RAG (Day 2–3)

> ⚠️ This is the most time-intensive phase. If you're short on time, do a simplified version (Step 6b) and move on.

#### Step 6a — Full Bedrock Knowledge Base Setup `⭐⭐⭐⭐ Hard`
**What:** Download 2–3 recent 10-K PDFs from SEC EDGAR (NVDA, AMD, INTC are good choices for the demo). Create an S3 bucket. Upload PDFs. Create a Bedrock Knowledge Base pointing to the S3 bucket. Choose the default embeddings model (Titan Embeddings v2). Wait for ingestion (~10–20 min). Test queries via the Bedrock console.  
**Gotcha:** Knowledge Base setup has many steps. Follow the official AWS doc exactly. The IAM permissions are the most common failure point — your Bedrock execution role needs S3 read access.  
**Time:** 3–5 hours

#### Step 6b — Simplified RAG (Fallback if 6a is too slow) `⭐⭐ Medium`
**What:** Skip Bedrock Knowledge Base. Use LlamaIndex or LangChain locally. Ingest the same 10-K PDFs into a local FAISS vector store. Query it directly. Less impressive to judges but gets the feature working.  
**Time:** 1–2 hours

#### Step 7 — Wire `analyze_sec_filings_rag` Tool `⭐⭐⭐ Hard`
**What:** Write the tool schema. In the Event Router, parse company name + topic from the tool call, query your Knowledge Base (or local vector store), take the top 3 paragraphs, return them to Nova Sonic as a string. Nova Sonic summarizes aloud.  
**Milestone:** Say "What did Nvidia say about supply chain in their latest 10-K?" — Nova Sonic quotes from the actual filing.  
**Time:** 2–3 hours

---

### Phase 4 — Monte Carlo Wasm Compute (Day 3–4)

> 🔁 **Ironclad Overlap:** If `ironclad-runtime` is already built, Steps 8–9 take ~2 hours total instead of 2 days. The Wasm sandbox is already your execution environment.

#### Step 8 — Monte Carlo Python Script `⭐⭐ Medium`
**What:** Write `monte_carlo.py` — takes `(current_price, volatility, days, simulations=10000)`, runs Geometric Brownian Motion paths, returns `{"p10": x, "p50": y, "p90": z, "mean": z}`. Test it standalone in Python first.  
**Math:** GBM formula: `S(t) = S(0) * exp((mu - sigma²/2)*t + sigma*sqrt(t)*Z)` where Z ~ N(0,1). NumPy makes this 5 lines.  
**Time:** 1–2 hours

#### Step 9 — Route Through Ironclad-Runtime `⭐⭐⭐ Hard`
**What:** Instead of running `monte_carlo.py` directly, write it to `/tmp/sandbox/` and execute it via `./ironclad-runtime`. The Wasm sandbox runs the computation. Output JSON is captured and returned to Nova Sonic.  
**Why this wins:** "We don't run untrusted compute — even our own financial models — outside a Wasm sandbox with cryptographic audit logging" is an extraordinary demo sentence.  
**Gotcha:** NumPy may not be available inside `python.wasm`. Fallback: implement GBM in pure Python (slower but works). Or: run the computation natively in Python and just *route through* ironclad-runtime for the audit log — judges see the same architecture.  
**Time:** 2–3 hours (or 30 minutes if ironclad-runtime already exists)

#### Step 10 — Wire `execute_quantitative_model` Tool `⭐⭐ Medium`
**What:** Tool schema for the model. Event Router fetches live volatility from Polygon, passes params to the Monte Carlo runner, formats P10/P50/P90 results, returns to Nova Sonic. Test: "Run a Monte Carlo on AMD for the next 30 days."  
**Milestone:** Nova Sonic gives you probability distribution of AMD price in 30 days.  
**Time:** 1–2 hours

---

### Phase 5 — Obsidian Vault Logger (Day 4)

#### Step 11 — `log_research_insight` Tool `⭐ Easy`
**What:** Write a tiny FastAPI endpoint `POST /vault/log` that takes `{content: str, tags: list[str]}` and writes a formatted `.md` file to a local directory (your Obsidian vault folder, or just `./vault/`). Wire the tool schema. Test: "Save this summary and tag it semiconductors."  
**The markdown format:**
```markdown
---
tags: [semiconductors, nvidia]
date: 2025-03-01
source: Nova Sonic Research Terminal
---

[content from Nova Sonic here]
```
**Time:** 1 hour

---

### Phase 6 — Interruption Demo & Polish (Day 4–5)

#### Step 12 — Validate VAD Interruption `⭐⭐ Medium`
**What:** Nova Sonic's built-in VAD handles interruption. But you need to verify it works mid-financial-summary. Trigger a long SEC filing summary. Talk over it. Confirm it stops. This is a key demo moment — rehearse it until it's reliable.  
**Why it matters for judging:** Financial summaries are long. Showing "skip the revenue numbers, just tell me the risk factors" mid-speech is the exact demo that proves your system is voice-native, not just voice-triggered.  
**Time:** 1–2 hours testing

#### Step 13 — Error Handling & API Fallbacks `⭐⭐ Medium`
**What:** Handle: API rate limits (Polygon free tier), no SEC filing found, Wasm compute timeout. Nova Sonic should speak a useful error ("I couldn't find recent filings for that ticker") rather than crashing.  
**Time:** 1–2 hours

#### Step 14 — Demo Script Rehearsal `⭐ Easy`
**What:** Write out the exact 90-second demo flow. Rehearse it 10 times. Know exactly what to say and what Nova Sonic will reply. Judges are watching a live demo — every second counts.

**Suggested demo flow:**
1. "What's the current trading volume and price action for AMD?" → live data read back
2. "What did Nvidia say about their datacenter supply chain in their latest 10-K?" → RAG answer from filing
3. Start a Monte Carlo: "Run a Monte Carlo simulation on Nvidia for the next 30 days." Nova Sonic begins reading probabilities. **Interrupt:** "Wait, skip the distribution — just give me the worst-case scenario." Nova Sonic stops and answers.
4. "Save this insight to my vault and tag it semiconductors." → confirm saved.

**Total demo time:** ~90 seconds. All 4 tools. One interruption.  
**Time:** 1 hour to write, 2 hours to rehearse

---

## Architecture Diagram

```
User voice
    │
    ▼
[Nova Sonic - AWS Bedrock]
  - STT (built-in)
  - LLM reasoning (built-in)  
  - VAD + interruption (built-in)
  - TTS (built-in)
    │
    │ tool call JSON event
    ▼
[Event Router - FastAPI/Python]
    │
    ├──── query_live_market_data ──────▶ [Polygon.io API]
    │                                         │
    ├──── analyze_sec_filings_rag ─────▶ [Bedrock Knowledge Base]
    │                                    (S3 + SEC PDFs)
    │
    ├──── execute_quantitative_model ──▶ [ironclad-runtime]
    │                                    (Rust/Wasm sandbox)
    │                                    → monte_carlo.py
    │
    └──── log_research_insight ────────▶ [Local vault / Obsidian]
                                         → .md file write
    │
    │ tool result JSON
    ▼
[Nova Sonic - speaks result]
```

---

## What's In Hackathon MVP

- [ ] Nova Sonic voice conversation working (AWS Bedrock)
- [ ] `query_live_market_data` → Polygon.io live data
- [ ] `analyze_sec_filings_rag` → Bedrock KB or local FAISS
- [ ] `execute_quantitative_model` → Monte Carlo via Python (routed through ironclad-runtime if available)
- [ ] `log_research_insight` → local markdown file write
- [ ] Interruption demo working (Nova Sonic built-in VAD)
- [ ] 90-second demo rehearsed and reliable

## What's NOT In Hackathon MVP

| Temptation | Skip because |
|-----------|-------------|
| Options pricing (Black-Scholes) | Monte Carlo is already impressive. More math ≠ better demo. |
| Portfolio optimization | Out of scope. One model, working, is the pitch. |
| Real-time streaming charts | UI distraction. Judges are watching the voice interaction. |
| Authentication / user accounts | Nobody is logging in at a hackathon demo. |
| More than 3–4 tickers in the KB | Ingest NVDA, AMD, INTC. That covers 90% of plausible demo questions. |
| Full Ironclad security audit logging | Include it if already built. Don't build it fresh for this hackathon. |

---

## AWS Free Tier / Cost Watchlist

| Service | Free tier | Risk |
|---------|-----------|------|
| Nova Sonic | Check current Bedrock pricing | May have per-minute cost — monitor usage |
| Bedrock Knowledge Base | First queries may be free | S3 storage cost is negligible |
| S3 (for KB) | 5GB free | No risk at hackathon scale |
| Polygon.io | 5 calls/min free | Rate limit your Event Router |
| SEC EDGAR API | Free, no key | No risk |

> Set a **AWS billing alert at $20** the moment you create your account. Nova Sonic audio processing can accumulate cost during heavy testing.

---

## Judging Criteria Mapping

| Judging axis | How you hit it |
|-------------|---------------|
| **Technical complexity** | RAG pipeline + Wasm compute + live API orchestration. Three backend systems, one voice interface. |
| **Nova Sonic usage** | It's the entire frontend. STT, LLM, TTS, VAD all come from Nova Sonic. |
| **Real-world usefulness** | Financial research is a $X billion industry. Every analyst does manually what this agent automates. |
| **Demo quality** | The interruption mid-summary is your "wow" moment. Rehearse it until it's flawless. |
| **Code quality** | Clean Event Router, typed tool schemas, proper error handling. Judges look at the repo. |

---

## Files in This Folder That Already Apply

These docs from the core Voice AI project are still relevant — read them alongside this file:

| File | Relevant sections |
|------|------------------|
| `ROADMAP.md` | Prerequisites (FastAPI, Python) still apply. Phase 3 tool concepts apply. |
| `TECH_STACK.md` | Backend Framework section. LLM section (skip — Nova Sonic replaces this). |
| `ARCHITECTURE.md` | The interruption path diagram. LangGraph state concepts. |
| `GLOSSARY.md` | WebSocket, streaming, VAD, LLM, tool definitions. |

The `ironclad/` folder is directly relevant if you're routing Monte Carlo through the Wasm sandbox.
