"""
test_market_data.py — Tests for tools/market_data.py

Run:  pytest tests/test_market_data.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def polygon_snapshot_response() -> dict:
    """Minimal Polygon.io snapshot API response shape."""
    return {
        "status": "OK",
        "ticker": {
            "ticker": "NVDA",
            "lastTrade": {"p": 875.50},
            "day": {"o": 860.0, "h": 882.0, "l": 855.0, "c": 875.50, "v": 32_000_000},
            "prevDay": {"c": 856.20, "v": 28_000_000},
        },
    }


@pytest.mark.asyncio
async def test_get_market_snapshot_returns_expected_keys(polygon_snapshot_response):
    """get_market_snapshot should return all required keys."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = polygon_snapshot_response
    mock_response.raise_for_status = MagicMock()

    with patch("tools.market_data.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("nvda")

    assert result["ticker"] == "NVDA"
    assert result["price"] == 875.50
    assert "summary" in result
    assert "change_pct" in result


@pytest.mark.asyncio
async def test_get_market_snapshot_rate_limit():
    """A 429 response should return an error dict, not raise."""
    mock_response = MagicMock()
    mock_response.status_code = 429

    with patch("tools.market_data.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("NVDA")

    assert "error" in result
    assert "rate limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_get_market_snapshot_ticker_normalised():
    """Ticker should be uppercased regardless of input case."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Minimal valid shape — just enough to not crash
    mock_response.json.return_value = {
        "ticker": {"lastTrade": {"p": 100.0}, "day": {}, "prevDay": {}}
    }
    mock_response.raise_for_status = MagicMock()

    with patch("tools.market_data.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("amd")

    assert result["ticker"] == "AMD"
