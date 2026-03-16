"""Structured vault logger used by the log_research_insight tool."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from config import settings

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = (
    "## Executive Summary",
    "## Context Snapshot",
    "## Evidence and Tool Outputs",
    "## Key Takeaways",
    "## Risks and Unknowns",
    "## Suggested Next Steps",
    "## User Additions",
)


def _safe_filename(title: str, ts: str) -> str:
    """Produce a safe filename from the note title or timestamp."""
    if title:
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        safe = safe.strip().replace(" ", "_")
        return f"{safe[:60]}.md"
    return f"note_{ts.replace(':', '-').replace('T', '_')}.md"


def _yaml_list(values: list[str]) -> str:
    if not values:
        return "[]"
    escaped = [v.replace('"', "'") for v in values]
    return "[" + ", ".join(f'"{v}"' for v in escaped) + "]"


def _extract_tickers(content: str, context: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    stop_words = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "NOTE"}

    for match in re.findall(r"\b[A-Z]{1,5}\b", content):
        if match not in stop_words:
            tickers.append(match)

    for entry in context.get("tool_history", []):
        tool_input = entry.get("input", {})
        ticker = str(tool_input.get("ticker", "")).upper().strip()
        if ticker and ticker not in stop_words:
            tickers.append(ticker)

    deduped: list[str] = []
    for ticker in tickers:
        if ticker not in deduped:
            deduped.append(ticker)
    return deduped[:8]


def _extract_tools_used(context: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for entry in context.get("tool_history", []):
        name = str(entry.get("tool_name", "")).strip()
        if name and name not in tools:
            tools.append(name)
    latest_name = str(context.get("latest_tool_call", {}).get("tool_name", "")).strip()
    if latest_name and latest_name not in tools:
        tools.append(latest_name)
    return tools


def _resolve_title(title: str, tickers: list[str], ts: str) -> str:
    if title:
        return title
    date_part = ts.split("T", maxsplit=1)[0]
    if tickers:
        return f"{tickers[0]} Research Insight - {date_part}"
    return f"Research Insight - {date_part}"


def _fallback_body(
    *,
    title: str,
    content: str,
    tickers: list[str],
    tools_used: list[str],
    context: dict[str, Any],
) -> str:
    summary = content.strip() or "No summary was provided."
    session_id = context.get("session_id", "unknown")
    tool_lines = "\n".join([f"- {name}" for name in tools_used]) or "- None"
    ticker_line = ", ".join(tickers) if tickers else "Not identified"

    return (
        f"# {title}\n\n"
        "## Executive Summary\n"
        f"{summary}\n\n"
        "## Context Snapshot\n"
        f"- Session ID: {session_id}\n"
        f"- Tickers: {ticker_line}\n"
        "- Source: Nova Sonic Research Terminal\n\n"
        "## Evidence and Tool Outputs\n"
        "### Tools Used\n"
        f"{tool_lines}\n\n"
        "## Key Takeaways\n"
        "- Add the main conclusions here.\n\n"
        "## Risks and Unknowns\n"
        "- Add known uncertainties, assumptions, and risks.\n\n"
        "## Suggested Next Steps\n"
        "- Add follow-up actions to investigate.\n\n"
        "## User Additions\n"
        "- Add your own notes here.\n"
    )


def _ensure_required_sections(markdown_body: str, title: str) -> str:
    text = markdown_body.strip()
    if not text.startswith("# "):
        text = f"# {title}\n\n" + text

    missing = [section for section in _REQUIRED_SECTIONS if section not in text]
    for section in missing:
        if section == "## User Additions":
            text += "\n\n## User Additions\n- Add your own notes here.\n"
        else:
            text += f"\n\n{section}\n- Add details.\n"
    return text + ("\n" if not text.endswith("\n") else "")


def _build_front_matter(
    *,
    title: str,
    ts: str,
    tags: list[str],
    tickers: list[str],
    tools_used: list[str],
    context: dict[str, Any],
    llm_provider: str,
    llm_model: str,
) -> str:
    session_id = str(context.get("session_id", ""))
    safe_title = title.replace('"', "'")

    return (
        "---\n"
        f"title: \"{safe_title}\"\n"
        f"date: {ts}\n"
        f"updated: {ts}\n"
        "source: Nova Sonic Research Terminal\n"
        "note_type: research_insight\n"
        f"session_id: \"{session_id}\"\n"
        f"tags: {_yaml_list(tags)}\n"
        f"tickers: {_yaml_list(tickers)}\n"
        f"tools_used: {_yaml_list(tools_used)}\n"
        f"llm_provider: \"{llm_provider}\"\n"
        f"llm_model: \"{llm_model}\"\n"
        "---\n\n"
    )


def _build_llm_prompt(
    *,
    content: str,
    title: str,
    tags: list[str],
    context: dict[str, Any],
    tickers: list[str],
    tools_used: list[str],
) -> str:
    compact_context = {
        "session_id": context.get("session_id", ""),
        "last_user_summary": context.get("last_user_summary", ""),
        "latest_tool_call": context.get("latest_tool_call", {}),
        "tool_history": context.get("tool_history", [])[-6:],
    }

    schema_instructions = (
        "Return markdown only. Do not include YAML frontmatter.\n"
        "Use exactly this section structure and headings:\n"
        "# <title>\n"
        "## Executive Summary\n"
        "## Context Snapshot\n"
        "## Evidence and Tool Outputs\n"
        "## Key Takeaways\n"
        "## Risks and Unknowns\n"
        "## Suggested Next Steps\n"
        "## User Additions\n"
        "Keep the writing concise, factual, and readable in Obsidian.\n"
        "Use bullet points where useful.\n"
        "Never invent data; if unknown, explicitly say unknown."
    )

    payload = {
        "title": title,
        "raw_content": content,
        "tags": tags,
        "tickers": tickers,
        "tools_used": tools_used,
        "context": compact_context,
    }
    return f"{schema_instructions}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=True)}"


async def _compose_with_groq(prompt: str) -> tuple[str | None, str]:
    api_key = str(getattr(settings, "groq_api_key", "") or "").strip()
    model = str(getattr(settings, "groq_model", "llama-3.3-70b-versatile") or "").strip()

    if not api_key:
        return None, model

    timeout_seconds = int(getattr(settings, "note_llm_timeout_seconds", 20) or 20)
    endpoint = str(
        getattr(settings, "groq_base_url", "https://api.groq.com/openai/v1") or ""
    ).rstrip("/")

    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert financial research editor that writes "
                    "structured markdown notes for Obsidian."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{endpoint}/chat/completions", json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return (content or None), model
    except Exception as exc:
        logger.warning("Groq note generation failed: %s", exc)
        return None, model


async def _compose_with_nova_lite(prompt: str) -> tuple[str | None, str]:
    """Placeholder for Nova Lite note generation integration."""
    model = str(getattr(settings, "nova_lite_model_id", "amazon.nova-lite-v1:0") or "").strip()
    # TODO: Integrate Bedrock Nova Lite text generation path here.
    ...
    return None, model


async def _compose_structured_body(
    *,
    title: str,
    content: str,
    tags: list[str],
    context: dict[str, Any],
    tickers: list[str],
    tools_used: list[str],
) -> tuple[str, str, str]:
    prompt = _build_llm_prompt(
        content=content,
        title=title,
        tags=tags,
        context=context,
        tickers=tickers,
        tools_used=tools_used,
    )

    preferred_provider = str(getattr(settings, "note_llm_provider", "groq") or "groq").lower()

    if preferred_provider == "nova_lite":
        body, model = await _compose_with_nova_lite(prompt)
        if body:
            return _ensure_required_sections(body, title), "nova_lite", model

    body, model = await _compose_with_groq(prompt)
    if body:
        return _ensure_required_sections(body, title), "groq", model

    fallback = _fallback_body(
        title=title,
        content=content,
        tickers=tickers,
        tools_used=tools_used,
        context=context,
    )
    return fallback, "none", "none"


async def log_insight(
    content: str,
    tags: list[str] | None = None,
    title: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a structured research note to the configured vault directory."""
    tags = [t.strip().lower() for t in (tags or []) if t and t.strip()]
    context = context or {}

    ts = datetime.now().isoformat(timespec="seconds")
    tickers = _extract_tickers(content=content, context=context)
    tools_used = _extract_tools_used(context)

    for ticker in tickers:
        ticker_tag = ticker.lower()
        if ticker_tag not in tags:
            tags.append(ticker_tag)

    resolved_title = _resolve_title((title or "").strip(), tickers, ts)
    filename = _safe_filename(resolved_title, ts)

    body, llm_provider, llm_model = await _compose_structured_body(
        title=resolved_title,
        content=content,
        tags=tags,
        context=context,
        tickers=tickers,
        tools_used=tools_used,
    )

    front_matter = _build_front_matter(
        title=resolved_title,
        ts=ts,
        tags=tags,
        tickers=tickers,
        tools_used=tools_used,
        context=context,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    vault_dir: Path = settings.vault_path
    vault_dir.mkdir(parents=True, exist_ok=True)

    filepath = vault_dir / filename
    markdown = front_matter + body

    async with aiofiles.open(filepath, "w", encoding="utf-8") as fh:
        await fh.write(markdown)

    logger.info("Vault note saved: %s", filepath)

    return {
        "saved": True,
        "filepath": str(filepath),
        "message": f"Note saved as '{filename}' in vault.",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }
