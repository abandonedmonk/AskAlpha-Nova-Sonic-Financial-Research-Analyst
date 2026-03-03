"""
vault_logger.py — Tool 4: log_research_insight

Saves a research insight as a Markdown note to the local vault directory
(or an Obsidian vault if VAULT_PATH points to one).

Note format
───────────
  ---
  tags: [semiconductors, nvidia]
  date: 2025-03-01T14:32:00
  source: Nova Sonic Research Terminal
  ---

  # Note title

  [content body]
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from config import settings

logger = logging.getLogger(__name__)


def _build_markdown(content: str, tags: list[str], title: str) -> str:
    """Render the Obsidian-compatible YAML front-matter + body."""
    ts = datetime.now().isoformat(timespec="seconds")
    tags_str = "[" + ", ".join(tags) + "]" if tags else "[]"

    front_matter = (
        "---\n"
        f"tags: {tags_str}\n"
        f"date: {ts}\n"
        "source: Nova Sonic Research Terminal\n"
        "---\n\n"
    )

    heading = f"# {title}\n\n" if title else ""
    return front_matter + heading + content.strip() + "\n"


def _safe_filename(title: str, ts: str) -> str:
    """Produce a safe filename from the note title or timestamp."""
    if title:
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        safe = safe.strip().replace(" ", "_")
        return f"{safe[:60]}.md"
    return f"note_{ts.replace(':', '-').replace('T', '_')}.md"


async def log_insight(
    content: str,
    tags: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """
    Write *content* to a Markdown file in the vault directory.

    Parameters
    ----------
    content : Research text to save.
    tags    : Optional list of topic tags.
    title   : Optional note title (used as filename & H1 heading).

    Returns
    -------
    dict compatible with the ``log_research_insight`` tool schema.
    """
    tags = tags or []
    title = (title or "").strip()
    ts = datetime.now().isoformat(timespec="seconds")
    filename = _safe_filename(title, ts)

    vault_dir: Path = settings.vault_path
    vault_dir.mkdir(parents=True, exist_ok=True)

    filepath = vault_dir / filename
    markdown = _build_markdown(content, tags, title)

    async with aiofiles.open(filepath, "w", encoding="utf-8") as fh:
        await fh.write(markdown)

    logger.info("Vault note saved: %s", filepath)

    return {
        "saved": True,
        "filepath": str(filepath),
        "message": f"Note saved as '{filename}' in vault.",
    }
