"""
test_market_data.py — Tests for tools/market_data.py

Run:  pytest tests/test_market_data.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def finnhub_quote_response() -> dict:
    """Minimal Finnhub /quote response shape (free-tier endpoint)."""
    return {
        "c": 875.50,  # current price
        "o": 860.0,  # open
        "h": 882.0,  # high
        "l": 855.0,  # low
        "pc": 870.0,  # previous close
        "d": 5.50,  # change
        "dp": 0.63,  # change %
        "t": 1709942400,
    }


@pytest.fixture
def polygon_prev_agg_response() -> dict:
    """Minimal Polygon.io previous-day aggregate response shape for fallback."""
    return {
        "status": "OK",
        "ticker": "NVDA",
        "results": [{"o": 860.0, "h": 882.0, "l": 855.0, "c": 875.50, "v": 32_000_000}],
    }


@pytest.mark.asyncio
async def test_get_market_snapshot_returns_expected_keys(finnhub_quote_response):
    """Finnhub primary success should return all required keys."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = finnhub_quote_response

    with (
        patch("tools.market_data.httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.market_data", fromlist=["settings"]).settings,
            "finnhub_api_key",
            "finnhub-test-key",
        ),
    ):
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
    assert result["data_source"] == "Finnhub"


@pytest.mark.asyncio
async def test_get_market_snapshot_falls_back_to_polygon_with_eod_notice(
    polygon_prev_agg_response,
):
    """If Finnhub fails, Polygon fallback should be used and EOD must be disclosed."""
    finnhub_fail = MagicMock()
    finnhub_fail.status_code = 503
    finnhub_fail.json.return_value = {"error": "service unavailable"}

    polygon_ok = MagicMock()
    polygon_ok.status_code = 200
    polygon_ok.json.return_value = polygon_prev_agg_response

    with (
        patch("tools.market_data.httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.market_data", fromlist=["settings"]).settings,
            "finnhub_api_key",
            "finnhub-test-key",
        ),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[finnhub_fail, polygon_ok])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("NVDA")

    assert result["ticker"] == "NVDA"
    assert result["data_source"] == "Polygon fallback"
    assert "eod" in result["summary"].lower()
    assert result["data_freshness"].lower().startswith("eod")


@pytest.mark.asyncio
async def test_get_market_snapshot_ticker_normalised():
    """Ticker should be uppercased regardless of input case for Finnhub request."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "c": 100.0,
        "o": 99.0,
        "h": 101.0,
        "l": 98.0,
        "pc": 99.5,
        "t": 1709942400,
    }

    with (
        patch("tools.market_data.httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.market_data", fromlist=["settings"]).settings,
            "finnhub_api_key",
            "finnhub-test-key",
        ),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("amd")

    assert result["ticker"] == "AMD"
    _, kwargs = mock_client.get.call_args
    assert kwargs["params"]["symbol"] == "AMD"


@pytest.mark.asyncio
async def test_get_market_snapshot_both_providers_fail_returns_combined_error():
    """If both providers fail, return one clear combined error."""
    finnhub_fail = MagicMock()
    finnhub_fail.status_code = 401
    finnhub_fail.json.return_value = {"error": "invalid api key"}

    polygon_fail = MagicMock()
    polygon_fail.status_code = 403
    polygon_fail.json.return_value = {
        "status": "NOT_AUTHORIZED",
        "message": "You are not entitled to this data.",
    }

    with (
        patch("tools.market_data.httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.market_data", fromlist=["settings"]).settings,
            "finnhub_api_key",
            "bad-key",
        ),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[finnhub_fail, polygon_fail])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.market_data import get_market_snapshot

        result = await get_market_snapshot("NVDA")

    assert "error" in result
    assert "both providers" in result["error"].lower()
    assert "finnhub" in result["error"].lower()
    assert "polygon" in result["error"].lower()
