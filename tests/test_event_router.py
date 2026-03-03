"""
test_event_router.py — Tests for event_router/router.py dispatch logic.

Run:  pytest tests/test_event_router.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_dispatch_known_tool_delegates():
    """dispatch() should call the correct backend for each tool name."""
    expected = {"ticker": "NVDA", "price": 875.0, "summary": "mock"}

    with patch(
        "event_router.router.get_market_snapshot", new=AsyncMock(return_value=expected)
    ):
        from event_router.router import dispatch

        result = await dispatch("query_live_market_data", {"ticker": "NVDA"})

    assert result == expected


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error():
    """dispatch() with an unknown tool name should return an error dict."""
    from event_router.router import dispatch

    result = await dispatch("nonexistent_tool", {})
    assert "error" in result
    assert "Unknown tool" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_tool_exception_returns_error():
    """If a backend raises, dispatch() should return an error dict (not raise)."""
    with patch(
        "event_router.router.get_market_snapshot",
        new=AsyncMock(side_effect=RuntimeError("API timeout")),
    ):
        from event_router.router import dispatch

        result = await dispatch("query_live_market_data", {"ticker": "NVDA"})

    assert "error" in result
    assert "API timeout" in result["error"]


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return 200 with list of tools."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert "tools" in body
    assert "query_live_market_data" in body["tools"]
