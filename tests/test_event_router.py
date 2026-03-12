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

    with patch.dict(
        "event_router.router.TOOL_DISPATCH",
        {"query_live_market_data": AsyncMock(return_value=expected)},
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
    with patch.dict(
        "event_router.router.TOOL_DISPATCH",
        {
            "query_live_market_data": AsyncMock(
                side_effect=RuntimeError("API timeout")
            )
        },
    ):
        from event_router.router import dispatch

        result = await dispatch("query_live_market_data", {"ticker": "NVDA"})

    assert "error" in result
    assert "API timeout" in result["error"]


@pytest.mark.asyncio
async def test_dispatch_log_tool_passes_session_context():
    """dispatch() should pass session context into the vault logger backend."""
    mock_logger = AsyncMock(return_value={"saved": True, "filepath": "vault/note.md", "message": "ok"})

    with patch.dict("event_router.router.TOOL_DISPATCH", {"log_research_insight": mock_logger}):
        from event_router.router import dispatch

        context = {"session_id": "session-123", "tool_history": []}
        result = await dispatch(
            "log_research_insight",
            {"content": "Save this", "tags": ["nvda"], "title": "NVDA"},
            context,
        )

    assert result["saved"] is True
    mock_logger.assert_awaited_once_with(
        content="Save this",
        tags=["nvda"],
        title="NVDA",
        context=context,
    )


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return 200 with list of tools."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from event_router.router import router as event_router

    app = FastAPI()
    app.include_router(event_router)

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert "tools" in body
    assert "query_live_market_data" in body["tools"]
