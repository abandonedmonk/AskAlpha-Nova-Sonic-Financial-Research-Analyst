"""
market_data.py — Tool 1: query_live_market_data

Fetches the latest price/volume snapshot for a US equity ticker from
Polygon.io's REST API (free tier, 5 req/min).

Endpoint used:
  GET https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}
  Docs: https://polygon.io/docs/stocks/get_v2_snapshot_locale_us_markets_stocks_tickers__stocksTicker_
"""

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

_POLYGON_BASE = "https://api.polygon.io"


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
    url = f"{_POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params={"apiKey": settings.polygon_api_key})

    if response.status_code == 429:
        logger.warning("Polygon.io rate limit hit for ticker=%s", ticker)
        return {
            "error": "Rate limit reached. Polygon.io free tier allows 5 requests/min. Try again shortly."
        }

    response.raise_for_status()
    data = response.json()

    ticker_data = data.get("ticker", {})
    day = ticker_data.get("day", {})
    prev_day = ticker_data.get("prevDay", {})
    last_trade = ticker_data.get("lastTrade", {})

    current_price: float = last_trade.get("p") or day.get("c") or prev_day.get("c", 0.0)
    prev_close: float = prev_day.get("c", current_price)
    open_price: float = day.get("o", prev_close)
    high: float = day.get("h", current_price)
    low: float = day.get("l", current_price)
    volume: int = int(day.get("v") or prev_day.get("v", 0))

    change = current_price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0
    change_pct_str = f"{change_pct:+.2f}%"

    summary = (
        f"{ticker} is trading at ${current_price:,.2f}, "
        f"{change_pct_str} from yesterday's close, "
        f"on volume of {volume / 1_000_000:.1f}M shares."
    )

    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "open": round(open_price, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "volume": volume,
        "change_pct": change_pct_str,
        "summary": summary,
    }
