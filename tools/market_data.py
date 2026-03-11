"""
market_data.py — Tool 1: query_live_market_data

Fetches market data for a US equity ticker using a dual-provider strategy:

1) Finnhub (primary)
2) Polygon.io previous-day aggregate (fallback)

Endpoints used:
    Finnhub: GET https://finnhub.io/api/v1/stock/candle
    Polygon: GET https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true
"""

import logging
import time
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_POLYGON_BASE = "https://api.polygon.io"
_FINNHUB_BASE = "https://finnhub.io/api/v1"
_CACHE_TTL_SECONDS = 60
_SNAPSHOT_CACHE: dict[str, dict[str, Any]] = {}


def _build_snapshot_response(
    ticker: str,
    price: float,
    open_price: float,
    high: float,
    low: float,
    volume: int,
    *,
    source: str,
    data_freshness: str,
    change_reference: str,
) -> dict[str, Any]:
    change = price - open_price
    change_pct = (change / open_price * 100) if open_price else 0.0
    change_pct_str = f"{change_pct:+.2f}%"

    summary = (
        f"{ticker} {source} data ({data_freshness}): price ${price:,.2f}, "
        f"{change_pct_str} vs {change_reference}, "
        f"volume {volume / 1_000_000:.1f}M shares."
    )

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "open": round(open_price, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "volume": volume,
        "change_pct": change_pct_str,
        "summary": summary,
        "data_source": source,
        "data_freshness": data_freshness,
    }


async def _get_finnhub_snapshot(ticker: str) -> dict[str, Any]:
    if not settings.finnhub_api_key:
        return {"error": "Finnhub API key is not configured."}

    # /quote is the free-tier real-time endpoint.
    # /stock/candle requires a premium plan and will 403 on free keys.
    url = f"{_FINNHUB_BASE}/quote"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            url,
            params={
                "symbol": ticker,
                "token": settings.finnhub_api_key,
            },
        )

    data: dict[str, Any] = {}
    try:
        data = response.json()
    except ValueError:
        data = {}

    if response.status_code == 429:
        return {"error": "Finnhub rate limit reached."}

    if response.status_code in (401, 403):
        return {"error": "Finnhub authentication failed. Check FINNHUB_API_KEY."}

    if response.status_code == 404:
        return {"error": f"Ticker '{ticker}' was not found on Finnhub."}

    if response.status_code >= 500:
        return {"error": "Finnhub is temporarily unavailable."}

    if response.status_code >= 400:
        detail = str(data.get("error", "")).strip()
        return {"error": detail or "Failed to fetch market data from Finnhub."}

    # Quote response: c=current, o=open, h=high, l=low, pc=prev close, v not included
    price = float(data.get("c") or 0.0)
    open_price = float(data.get("o") or price)
    high = float(data.get("h") or price)
    low = float(data.get("l") or price)

    if price == 0.0:
        return {"error": f"No market data available for ticker '{ticker}' on Finnhub."}

    return _build_snapshot_response(
        ticker,
        price,
        open_price,
        high,
        low,
        volume=0,  # /quote does not return volume; Polygon fallback does
        source="Finnhub",
        data_freshness="real-time",
        change_reference="daily open",
    )


async def _get_polygon_prev_snapshot(ticker: str) -> dict[str, Any]:
    url = f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/prev"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            url,
            params={"adjusted": "true", "apiKey": settings.polygon_api_key},
        )

    data: dict[str, Any] = {}
    try:
        data = response.json()
    except ValueError:
        data = {}

    if response.status_code == 429:
        logger.warning("Polygon.io rate limit hit for ticker=%s", ticker)
        return {
            "error": "Polygon rate limit reached. Free tier allows 5 requests/min. Try again shortly."
        }

    if response.status_code == 404:
        logger.info("Polygon.io ticker not found ticker=%s", ticker)
        return {"error": f"Ticker '{ticker}' was not found on Polygon."}

    if response.status_code in (401, 403):
        message = str(data.get("message", "")).strip()
        status = str(data.get("status", "")).strip().upper()
        if status == "NOT_AUTHORIZED" or "not entitled" in message.lower():
            logger.warning("Polygon.io entitlement error for ticker=%s", ticker)
            return {
                "error": "Your Polygon plan is not entitled for this data endpoint."
            }
        logger.warning("Polygon.io auth error status=%s ticker=%s", response.status_code, ticker)
        return {"error": "Polygon authentication failed. Check POLYGON_API_KEY."}

    if response.status_code >= 500:
        logger.warning("Polygon.io upstream error status=%s ticker=%s", response.status_code, ticker)
        return {"error": "Polygon is temporarily unavailable."}

    if response.status_code >= 400:
        logger.warning("Polygon.io request failed status=%s ticker=%s", response.status_code, ticker)
        detail = str(data.get("message", "")).strip()
        return {"error": detail or "Failed to fetch market data from Polygon."}

    if str(data.get("status", "")).upper() == "NOT_AUTHORIZED":
        logger.warning("Polygon.io entitlement payload returned for ticker=%s", ticker)
        return {
            "error": "Your Polygon plan is not entitled for this data endpoint."
        }

    results = data.get("results", [])
    if not results:
        logger.info("Polygon.io returned no aggregate data for ticker=%s", ticker)
        return {"error": f"No market data available for ticker '{ticker}' on Polygon."}

    bar = results[0]
    close_price = float(bar.get("c") or 0.0)
    open_price = float(bar.get("o") or close_price)
    high = float(bar.get("h") or close_price)
    low = float(bar.get("l") or close_price)
    volume = int(bar.get("v") or 0)

    payload = _build_snapshot_response(
        ticker,
        close_price,
        open_price,
        high,
        low,
        volume,
        source="Polygon fallback",
        data_freshness="EOD (previous trading day)",
        change_reference="daily open",
    )
    payload["summary"] = (
        f"{payload['summary']} Note: This is end-of-day (EOD) data served by the Polygon fallback provider."
    )
    return payload


async def get_market_snapshot(ticker: str) -> dict[str, Any]:
    """
    Return a concise market data snapshot for *ticker*.

    Parameters
    ----------
    ticker : str  Uppercase ticker symbol, e.g. "NVDA".

    Returns
    -------
    dict compatible with the ``query_live_market_data`` tool schema.
    """
    ticker = ticker.upper().strip()
    cached_entry = _SNAPSHOT_CACHE.get(ticker)
    if cached_entry and time.time() - cached_entry["timestamp"] < _CACHE_TTL_SECONDS:
        logger.info("Market data cache hit for ticker=%s", ticker)
        return dict(cached_entry["payload"])

    primary_result = await _get_finnhub_snapshot(ticker)
    if "error" not in primary_result:
        _SNAPSHOT_CACHE[ticker] = {
            "timestamp": time.time(),
            "payload": primary_result,
        }
        return dict(primary_result)

    logger.warning(
        "Finnhub primary provider failed for ticker=%s, falling back to Polygon: %s",
        ticker,
        primary_result.get("error", "unknown error"),
    )

    fallback_result = await _get_polygon_prev_snapshot(ticker)
    if "error" not in fallback_result:
        _SNAPSHOT_CACHE[ticker] = {
            "timestamp": time.time(),
            "payload": fallback_result,
        }
        return dict(fallback_result)

    return {
        "error": (
            "Market data request failed on both providers. "
            f"Finnhub error: {primary_result.get('error', 'unknown')}. "
            f"Polygon error: {fallback_result.get('error', 'unknown')}."
        )
    }
