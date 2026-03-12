"""
test_vault_logger.py — Tests for tools/vault_logger.py

Run:  pytest tests/test_vault_logger.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_log_insight_creates_file(tmp_path):
    """log_insight should create a .md file in the vault directory."""
    with patch("tools.vault_logger.settings") as mock_settings:
        mock_settings.vault_path = tmp_path

        from tools.vault_logger import log_insight

        result = await log_insight(
            content="Nvidia mentioned supply chain constraints.",
            tags=["semiconductors", "nvidia"],
            title="NVDA Supply Chain Note",
        )

    assert result["saved"] is True
    saved_path = Path(result["filepath"])
    assert saved_path.exists()
    text = saved_path.read_text(encoding="utf-8")
    assert "semiconductors" in text
    assert "Nvidia mentioned supply chain" in text
    assert "## Executive Summary" in text
    assert "## User Additions" in text


@pytest.mark.asyncio
async def test_log_insight_front_matter(tmp_path):
    """Saved file should contain valid YAML front matter."""
    with patch("tools.vault_logger.settings") as mock_settings:
        mock_settings.vault_path = tmp_path

        from tools.vault_logger import log_insight

        result = await log_insight(content="Test content.", tags=["test"])

    text = Path(result["filepath"]).read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "source: Nova Sonic Research Terminal" in text
    assert "tags:" in text
    assert "test" in text
    assert "note_type: research_insight" in text


@pytest.mark.asyncio
async def test_log_insight_no_tags(tmp_path):
    """log_insight should work without tags."""
    with patch("tools.vault_logger.settings") as mock_settings:
        mock_settings.vault_path = tmp_path

        from tools.vault_logger import log_insight

        result = await log_insight(content="No tags note.")

    assert result["saved"] is True


@pytest.mark.asyncio
async def test_log_insight_filename_sanitised(tmp_path):
    """Special characters in title should not break the filename."""
    with patch("tools.vault_logger.settings") as mock_settings:
        mock_settings.vault_path = tmp_path

        from tools.vault_logger import log_insight

        result = await log_insight(
            content="Special chars test.",
            title="Q1/2025 <Report> & Notes!",
        )

    fname = Path(result["filepath"]).name
    assert "/" not in fname
    assert "<" not in fname
    assert fname.endswith(".md")


@pytest.mark.asyncio
async def test_log_insight_includes_context_metadata(tmp_path):
    """Session context should be reflected in frontmatter and body metadata."""
    with patch("tools.vault_logger.settings") as mock_settings:
        mock_settings.vault_path = tmp_path

        from tools.vault_logger import log_insight

        result = await log_insight(
            content="Please save the NVDA update.",
            tags=["research"],
            context={
                "session_id": "session-42",
                "tool_history": [
                    {
                        "tool_name": "query_live_market_data",
                        "input": {"ticker": "NVDA"},
                    }
                ],
            },
        )

    text = Path(result["filepath"]).read_text(encoding="utf-8")
    assert 'session_id: "session-42"' in text
    assert "tools_used:" in text
    assert "query_live_market_data" in text
