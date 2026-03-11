"""
test_quant_model.py - Tests for tools/quant_model.py volatility provider path.

Run: pytest tests/test_quant_model.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_price_and_volatility_uses_tiingo_daily_prices():
    """Volatility should be computed from Tiingo adjusted daily price closes."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"close": 100.0, "adjClose": 99.8, "open": 99.5, "high": 100.5, "low": 99.0, "volume": 1000000},
        {"close": 101.0, "adjClose": 100.8, "open": 100.0, "high": 101.5, "low": 100.2, "volume": 1100000},
        {"close": 99.5, "adjClose": 99.3, "open": 101.0, "high": 101.2, "low": 99.3, "volume": 950000},
        {"close": 102.0, "adjClose": 101.8, "open": 99.8, "high": 102.3, "low": 99.5, "volume": 1050000},
        {"close": 103.2, "adjClose": 103.0, "open": 101.8, "high": 103.5, "low": 101.5, "volume": 1200000},
        {"close": 104.1, "adjClose": 103.9, "open": 103.0, "high": 104.3, "low": 102.8, "volume": 1150000},
        {"close": 103.8, "adjClose": 103.6, "open": 104.0, "high": 104.2, "low": 103.5, "volume": 1000000},
    ]

    with (
        patch(
            "tools.quant_model.get_market_snapshot",
            new=AsyncMock(return_value={"price": 123.45}),
        ),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.quant_model", fromlist=["settings"]).settings,
            "tiingo_api_key",
            "tiingo-test-key",
        ),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.quant_model import _get_price_and_volatility

        price, vol = await _get_price_and_volatility("NVDA")

    assert price == 123.45
    assert vol > 0

    _, kwargs = mock_client.get.call_args
    assert "startDate" in kwargs["params"]
    assert "endDate" in kwargs["params"]
    assert kwargs["params"]["token"] == "tiingo-test-key"


@pytest.mark.asyncio
async def test_get_price_and_volatility_falls_back_on_tiingo_failure():
    """If Tiingo call fails, Polygon bars should be used for volatility."""
    tiingo_response = MagicMock()
    tiingo_response.status_code = 200
    tiingo_response.json.return_value = []  # Empty response

    polygon_response = MagicMock()
    polygon_response.status_code = 200
    polygon_response.json.return_value = {
        "status": "OK",
        "results": [
            {"c": 200.0},
            {"c": 201.5},
            {"c": 199.8},
            {"c": 202.2},
            {"c": 203.0},
            {"c": 204.1},
            {"c": 205.0},
        ],
    }

    with (
        patch(
            "tools.quant_model.get_market_snapshot",
            new=AsyncMock(return_value={"price": 210.0}),
        ),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.quant_model", fromlist=["settings"]).settings,
            "tiingo_api_key",
            "tiingo-test-key",
        ),
        patch.object(
            __import__("tools.quant_model", fromlist=["settings"]).settings,
            "polygon_api_key",
            "poly-test-key",
        ),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[tiingo_response, polygon_response])
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.quant_model import _get_price_and_volatility

        price, vol = await _get_price_and_volatility("AMD")

    assert price == 210.0
    assert vol > 0


@pytest.mark.asyncio
async def test_get_price_and_volatility_defaults_when_tiingo_key_missing():
    """If both keys are missing, volatility should default to 0.30."""
    with (
        patch(
            "tools.quant_model.get_market_snapshot",
            new=AsyncMock(return_value={"price": 88.0}),
        ),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(
            __import__("tools.quant_model", fromlist=["settings"]).settings,
            "tiingo_api_key",
            "",
        ),
        patch.object(
            __import__("tools.quant_model", fromlist=["settings"]).settings,
            "polygon_api_key",
            "",
        ),
    ):
        from tools.quant_model import _get_price_and_volatility

        price, vol = await _get_price_and_volatility("INTC")

    assert price == 88.0
    assert vol == 0.30
    mock_client_cls.assert_not_called()
