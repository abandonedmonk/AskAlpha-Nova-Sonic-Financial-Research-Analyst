"""
quant_model.py — Tool 3: execute_quantitative_model

Runs a Monte Carlo (Geometric Brownian Motion) simulation for a given ticker.

Execution paths
───────────────
1. Ironclad Wasm sandbox (preferred)
   If IRONCLAD_RUNTIME_PATH points to a valid binary, the monte_carlo.py
   script is written to the sandbox input directory and executed via
   ironclad-runtime.  Results are read from the sandbox output.

2. Native Python (fallback)
   monte_carlo.py is imported directly and run in-process.
   Faster, but without the Wasm audit layer.

Live price is fetched via Tool 1 and volatility is fetched from Tiingo daily bars
before the simulation.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from config import settings
from tools.market_data import get_market_snapshot

logger = logging.getLogger(__name__)


# ── Fetch live price & compute historical volatility ─────────────────────────


async def _get_price_and_volatility(ticker: str) -> tuple[float, float]:
    """
    Returns (current_price, annualised_volatility).

    Volatility is fetched from daily bars using provider fallback:
    1) Tiingo daily prices (90-day window)
    2) Polygon daily aggregates
    Falls back to a conservative 30% if both fail.
    """
    import datetime
    import httpx

    snapshot = await get_market_snapshot(ticker)
    current_price: float = snapshot.get("price", 100.0)

    # Attempt to compute realised vol from recent daily closes.
    # We request a 120-day window to reliably capture ~90 trading days.
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        from_dt = now_utc - datetime.timedelta(days=120)
        today = now_utc.date().isoformat()
        from_iso = from_dt.date().isoformat()

        cleaned_closes: list[float] = []

        # Primary: Tiingo daily prices endpoint.
        if settings.tiingo_api_key:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                    params={
                        "startDate": from_iso,
                        "endDate": today,
                        "token": settings.tiingo_api_key,
                    },
                )

            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, list):
                    cleaned_closes = [
                        float(item.get("adjClose"))
                        for item in payload
                        if isinstance(item, dict)
                        and isinstance(item.get("adjClose"), (int, float))
                        and float(item.get("adjClose")) > 0
                    ]

        # Fallback: Polygon daily aggregates.
        if len(cleaned_closes) <= 5 and settings.polygon_api_key:
            polygon_url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
                f"/{from_iso}/{today}"
            )
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    polygon_url,
                    params={"apiKey": settings.polygon_api_key, "limit": 120},
                )

            if resp.status_code == 200:
                payload = resp.json()
                results = payload.get("results", [])
                cleaned_closes = [
                    float(item.get("c"))
                    for item in results
                    if isinstance(item, dict)
                    and isinstance(item.get("c"), (int, float))
                    and float(item.get("c")) > 0
                ]

        if len(cleaned_closes) > 5:
            import numpy as np  # type: ignore

            close_arr = np.array(cleaned_closes[-90:], dtype=float)
            log_returns = np.diff(np.log(close_arr))
            daily_vol = float(np.std(log_returns))
            annual_vol = daily_vol * (252**0.5)
            return current_price, annual_vol

        raise RuntimeError("No sufficient daily closes from Tiingo/Polygon")
    except Exception as exc:
        logger.warning(
            "Could not fetch historical volatility: %s — defaulting to 0.30", exc
        )

    return current_price, 0.30  # sensible default


# ── Native Python execution ───────────────────────────────────────────────────


def _run_monte_carlo_native(
    current_price: float,
    volatility: float,
    days: int,
    simulations: int,
) -> dict[str, float]:
    """Import and run compute/monte_carlo.py directly in-process."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from compute.monte_carlo import simulate  # type: ignore

    return simulate(
        current_price=current_price,
        volatility=volatility,
        days=days,
        simulations=simulations,
    )


# ── Ironclad Wasm sandbox execution ──────────────────────────────────────────


def _run_monte_carlo_ironclad(
    current_price: float,
    volatility: float,
    days: int,
    simulations: int,
) -> dict[str, float]:
    """
    Write a self-contained Python script to the ironclad sandbox temp dir
    and execute it via ironclad-runtime.  Captures JSON output from stdout.
    """
    script = f"""
import json, math, random

current_price = {current_price}
volatility = {volatility}
days = {days}
simulations = {simulations}
mu = 0.0  # drift (risk-neutral)
dt = 1 / 252

import sys
try:
    import numpy as np
    Z = np.random.standard_normal((simulations, days))
    log_ret = (mu - 0.5 * volatility**2) * dt + volatility * (dt**0.5) * Z
    paths = current_price * np.exp(np.cumsum(log_ret, axis=1))
    finals = paths[:, -1]
    result = {{
        "p10": float(np.percentile(finals, 10)),
        "p50": float(np.percentile(finals, 50)),
        "p90": float(np.percentile(finals, 90)),
        "mean": float(np.mean(finals)),
    }}
except ImportError:
    # Pure-Python GBM fallback (slower)
    finals = []
    for _ in range(simulations):
        price = current_price
        for _ in range(days):
            z = random.gauss(0, 1)
            price *= math.exp((mu - 0.5 * volatility**2) * dt + volatility * (dt**0.5) * z)
        finals.append(price)
    finals.sort()
    n = len(finals)
    result = {{
        "p10": finals[int(n * 0.10)],
        "p50": finals[int(n * 0.50)],
        "p90": finals[int(n * 0.90)],
        "mean": sum(finals) / n,
    }}

print(json.dumps(result))
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "monte_carlo_run.py"
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run(
            [str(settings.ironclad_runtime_path), str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"ironclad-runtime exited {proc.returncode}: {proc.stderr[:500]}"
        )

    return json.loads(proc.stdout.strip())


# ── Public tool function ──────────────────────────────────────────────────────


async def run_monte_carlo(
    ticker: str,
    days: int,
    simulations: int = 10_000,
) -> dict[str, Any]:
    """
    Run a GBM Monte Carlo simulation for *ticker* over *days* trading days.

    Parameters
    ----------
    ticker      : Equity ticker symbol.
    days        : Number of trading days to project.
    simulations : Number of Monte Carlo paths (default 10 000).

    Returns
    -------
    dict compatible with the ``execute_quantitative_model`` tool schema.
    """
    ticker = ticker.upper().strip()
    logger.info(
        "Monte Carlo — ticker=%s days=%d simulations=%d ironclad=%s",
        ticker,
        days,
        simulations,
        settings.ironclad_available,
    )

    current_price, volatility = await _get_price_and_volatility(ticker)

    loop = asyncio.get_event_loop()
    if settings.ironclad_available:
        logger.info("Routing through ironclad-runtime Wasm sandbox.")
        result = await loop.run_in_executor(
            None,
            _run_monte_carlo_ironclad,
            current_price,
            volatility,
            days,
            simulations,
        )
    else:
        logger.info("Running Monte Carlo natively (ironclad-runtime not found).")
        result = await loop.run_in_executor(
            None,
            _run_monte_carlo_native,
            current_price,
            volatility,
            days,
            simulations,
        )

    p10, p50, p90, mean = result["p10"], result["p50"], result["p90"], result["mean"]

    summary = (
        f"Monte Carlo on {ticker} over {days} trading days "
        f"({simulations:,} paths): "
        f"worst-case P10 ${p10:,.2f} | median P50 ${p50:,.2f} | "
        f"best-case P90 ${p90:,.2f} | mean ${mean:,.2f}. "
        f"Current price: ${current_price:,.2f}."
    )

    return {
        "ticker": ticker,
        "days": days,
        "simulations": simulations,
        "current_price": round(current_price, 2),
        "p10": round(p10, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "mean": round(mean, 2),
        "summary": summary,
    }
