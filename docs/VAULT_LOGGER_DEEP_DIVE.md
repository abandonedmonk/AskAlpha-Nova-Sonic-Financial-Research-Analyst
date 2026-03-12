# Vault Logger Deep Dive

This document explains Tool 4 (`log_research_insight`) from scratch, including helper functions, LLM integration flow, fallback behavior, and testing notes.

---

## 1) What This Module Does

File: `tools/vault_logger.py`

The vault logger takes raw insight text from Nova Sonic and writes a structured Obsidian-compatible Markdown note to the configured vault path.

It does more than file write now:

1. Enriches the note with metadata from session context.
2. Builds a strict note schema (fixed section headings).
3. Calls an LLM (Groq currently) to generate polished note content.
4. Falls back to deterministic local templating if LLM is unavailable.
5. Saves note and returns provider metadata (`llm_provider`, `llm_model`).

---

## 2) High-Level Execution Flow

Entry point: `async def log_insight(...)`

1. Normalize incoming tags and context.
2. Extract candidate tickers from raw text and tool history.
3. Extract tool names used in session.
4. Resolve final note title.
5. Generate body via provider flow:
   - `note_llm_provider == nova_lite`: try Nova Lite placeholder first.
   - otherwise: try Groq.
   - on failure: use local fallback body.
6. Build YAML front matter.
7. Ensure vault directory exists.
8. Write the note with `aiofiles`.
9. Return save metadata and LLM provider info.

---

## 3) Function-by-Function Reference

### `_safe_filename(title: str, ts: str) -> str`

Purpose:

- Convert note title into a safe filename.
- Fallback to timestamp-based filename when title is empty.

Key behavior:

- Replaces unsafe chars with `_`.
- Converts spaces to `_`.
- Truncates title-derived filename to 60 chars.

Why it matters:

- Prevents path errors and invalid filesystem names.

---

### `_yaml_list(values: list[str]) -> str`

Purpose:

- Convert Python list of strings to YAML-inline style list string.

Key behavior:

- Returns `[]` for empty list.
- Escapes double quotes by replacing `"` with `'`.
- Produces like: `["nvda", "semiconductors"]`.

Why it matters:

- Keeps front matter machine-readable and consistent.

---

### `_extract_tickers(content: str, context: dict[str, Any]) -> list[str]`

Purpose:

- Infer ticker symbols from both free text and previous tool inputs.

Key behavior:

- Regex scan for uppercase words `\b[A-Z]{1,5}\b`.
- Filters common false positives (`THE`, `AND`, etc.).
- Also inspects `context.tool_history[*].input.ticker`.
- Deduplicates while preserving first-seen order.
- Caps output to first 8 tickers.

Why it matters:

- Makes notes searchable by ticker without requiring user to tag manually.

---

### `_extract_tools_used(context: dict[str, Any]) -> list[str]`

Purpose:

- Collect unique tool names used in session.

Key behavior:

- Reads from `context.tool_history`.
- Also considers `context.latest_tool_call`.
- Deduplicates in insertion order.

Why it matters:

- Adds provenance to note metadata (`tools_used`).

---

### `_resolve_title(title: str, tickers: list[str], ts: str) -> str`

Purpose:

- Compute a deterministic title if none provided.

Key behavior:

- If title provided, use as-is.
- If ticker exists: `<TICKER> Research Insight - <YYYY-MM-DD>`.
- Else: `Research Insight - <YYYY-MM-DD>`.

Why it matters:

- Keeps saved notes consistently named and easier to browse.

---

### `_fallback_body(...) -> str`

Purpose:

- Build a complete structured note body without LLM.

Key behavior:

- Includes all required headings.
- Injects summary, session ID, ticker list, and tools used.
- Leaves clear TODO-style bullets for user follow-up.

Why it matters:

- Guarantees useful output even during provider/network outages.

---

### `_ensure_required_sections(markdown_body: str, title: str) -> str`

Purpose:

- Validate/repair generated markdown shape.

Key behavior:

- Ensures H1 exists.
- Appends missing required sections.
- Ensures trailing newline.

Why it matters:

- LLM output can drift; this function enforces schema contract.

---

### `_build_front_matter(...) -> str`

Purpose:

- Build final YAML front matter block.

Fields currently written:

- `title`
- `date`
- `updated`
- `source`
- `note_type`
- `session_id`
- `tags`
- `tickers`
- `tools_used`
- `llm_provider`
- `llm_model`

Why it matters:

- Enables strong Obsidian filtering, graphing, and template compatibility.

---

### `_build_llm_prompt(...) -> str`

Purpose:

- Prepare strict generation instructions and compact context payload.

Key behavior:

- Requests markdown only (no front matter).
- Forces exact heading schema.
- Includes serialized JSON payload containing:
  - title, raw content, tags, tickers, tools
  - context subset (`session_id`, `last_user_summary`, `latest_tool_call`, recent tool history)
- Uses `json.dumps(..., ensure_ascii=True)` for stable prompt serialization.

Why it matters:

- Reduces hallucination and structure drift from model output.

---

### `async _compose_with_groq(prompt: str) -> tuple[str | None, str]`

Purpose:

- Call Groq Chat Completions API and return generated markdown.

Key behavior:

- Reads key/model/base URL/timeout from `config.settings`.
- Returns `(None, model)` when API key is missing.
- Uses `httpx.AsyncClient` with timeout.
- Sends system + user messages.
- Parses `choices[0].message.content`.
- On exception: logs warning, returns `None`.

Why it matters:

- Current active provider path for production-like note generation.

---

### `async _compose_with_nova_lite(prompt: str) -> tuple[str | None, str]`

Purpose:

- Placeholder hook for future Nova Lite integration.

Current behavior:

- Returns `(None, model)`.
- Contains TODO marker for Bedrock text-generation wiring.

Why it matters:

- Provider switch can happen without changing outer pipeline.

---

### `async _compose_structured_body(...) -> tuple[str, str, str]`

Purpose:

- Orchestrate provider selection and fallback.

Return tuple:

- markdown body
- provider name
- model id

Flow:

1. Build prompt.
2. If preferred provider is `nova_lite`, try that first.
3. Try Groq path.
4. If all fail, build fallback body.

Why it matters:

- Central control point for provider behavior and resilience.

---

### `async log_insight(content, tags=None, title=None, context=None) -> dict[str, Any]`

Purpose:

- Public tool entrypoint used by router and smoke tests.

Inputs:

- `content`: raw summary / transcript-like insight text.
- `tags`: optional tags.
- `title`: optional title.
- `context`: optional session context payload.

Outputs:

- `saved`: bool
- `filepath`: absolute path string
- `message`: user-friendly save message
- `llm_provider`: `groq`, `nova_lite`, or `none`
- `llm_model`: model id or `none`

Why it matters:

- This is the contract consumed by Event Router and downstream confirmation responses.

---

## 4) Context Contract Expected by Tool 4

The logger accepts generic `context: dict[str, Any]`, but it benefits most from:

- `session_id`
- `last_user_summary`
- `latest_tool_call`
- `tool_history`: list of entries with `tool_name`, `input`, and optional `result`

This context is currently populated in `nova_sonic/session.py` and forwarded by `event_router/router.py`.

---

## 5) Less-Obvious Functions and Patterns Explained

### `re.findall(r"\b[A-Z]{1,5}\b", content)`

- Finds uppercase words of length 1-5.
- Useful for quick ticker extraction.
- Tradeoff: may catch non-tickers; stop-word filtering reduces noise.

### `getattr(settings, "field", default)`

- Reads optional settings safely without AttributeError.
- Important for test stubs and partial configs.

### `.rstrip("/")` on base URL

- Prevents accidental double slash in final endpoint path.
- Example: `https://api.groq.com/openai/v1/` -> `.../v1`.

### `json.dumps(..., ensure_ascii=True)`

- Ensures serialized prompt is ASCII-safe.
- Reduces encoding edge cases in provider requests.

### Ordered dedupe pattern

- Uses list plus membership checks rather than set conversion.
- Preserves first-seen order for user-facing metadata.

### `aiofiles.open(..., "w", encoding="utf-8")`

- Async file write avoids blocking event loop in FastAPI async path.

---

## 6) Failure Modes and Fallback Behavior

1. Missing `GROQ_API_KEY`

- Groq path returns `None` immediately.
- Logger falls back to deterministic template unless another provider succeeds.

2. Network/API failure from provider

- Exception is caught and logged.
- Logger still writes note via fallback template.

3. LLM returns incomplete schema

- `_ensure_required_sections` appends missing sections.

4. Invalid/empty title

- `_resolve_title` and `_safe_filename` generate stable defaults.

---

## 7) How to Smoke Test Live LLM Note Generation

Run Tool 4 smoke only:

```powershell
python tests/smoke_test_tools.py vault
```

Strict mode (fail if fallback path used):

```powershell
$env:REQUIRE_NOTE_LLM='1'; python tests/smoke_test_tools.py vault
```

Expected signal of real LLM usage:

- HTTP 200 call to Groq endpoint in logs.
- PASS output including `llm=groq:<model>`.

---

## 8) Future Work Checklist (Nova Lite)

1. Implement `_compose_with_nova_lite` with Bedrock text-generation client.
2. Add provider-specific retries and observability.
3. Add golden snapshot tests for generated markdown schema.
4. Add confidence/quality metadata in front matter.
