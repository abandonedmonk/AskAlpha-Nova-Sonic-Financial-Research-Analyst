"""
test_quant_model.py - Tests for tools/quant_model.py volatility provider path.

Run: pytest tests/test_quant_model.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_get_price_and_volatility_uses_finnhub_daily_candles():
    """Volatility should be computed from Finnhub daily candle closes."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "s": "ok",
        "c": [100.0, 101.0, 99.5, 102.0, 103.2, 104.1, 103.8],
    }

    with (
        patch("tools.quant_model.get_market_snapshot", new=AsyncMock(return_value={"price": 123.45})),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(__import__("tools.quant_model", fromlist=["settings"]).settings, "finnhub_api_key", "fh-test-key"),
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
    assert kwargs["params"]["symbol"] == "NVDA"
    assert kwargs["params"]["resolution"] == "D"
    assert kwargs["params"]["token"] == "fh-test-key"


@pytest.mark.asyncio
async def test_get_price_and_volatility_falls_back_on_finnhub_failure():
    """If Finnhub response is not usable, volatility should default to 0.30."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"s": "no_data", "c": []}

    with (
        patch("tools.quant_model.get_market_snapshot", new=AsyncMock(return_value={"price": 210.0})),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(__import__("tools.quant_model", fromlist=["settings"]).settings, "finnhub_api_key", "fh-test-key"),
    ):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.quant_model import _get_price_and_volatility

        price, vol = await _get_price_and_volatility("AMD")

    assert price == 210.0
    assert vol == 0.30


@pytest.mark.asyncio
async def test_get_price_and_volatility_defaults_when_finnhub_key_missing():
    """If FINNHUB_API_KEY is missing, no Finnhub call is attempted."""
    with (
        patch("tools.quant_model.get_market_snapshot", new=AsyncMock(return_value={"price": 88.0})),
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.object(__import__("tools.quant_model", fromlist=["settings"]).settings, "finnhub_api_key", ""),
    ):
        from tools.quant_model import _get_price_and_volatility

        price, vol = await _get_price_and_volatility("INTC")

    assert price == 88.0
    assert vol == 0.30
    mock_client_cls.assert_not_called()
