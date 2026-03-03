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
    assert "tags: [test]" in text


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
