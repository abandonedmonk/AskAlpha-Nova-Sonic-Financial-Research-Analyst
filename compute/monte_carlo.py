"""
monte_carlo.py — Geometric Brownian Motion Monte Carlo simulator.

This module can be executed in three ways:

1. Imported directly by tools/quant_model.py (native Python path).
2. Written to /tmp and executed via ironclad-runtime (Wasm sandbox path).
3. Run standalone from the CLI for testing:

      python compute/monte_carlo.py --ticker NVDA --days 30

Maths
─────
Stock price follows GBM:

    S(t) = S(0) · exp( (μ - σ²/2)·t  +  σ·√t·Z )

where Z ~ N(0,1).

With risk-neutral drift μ = 0 (conservative assumption for projection),
each simulated daily step is:

    S_{i+1} = S_i · exp( -½σ²·dt + σ·√dt·Z_i )

where dt = 1/252.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from typing import Any


logger = logging.getLogger(__name__)


def simulate(
    current_price: float,
    volatility: float,
    days: int,
    simulations: int = 10_000,
    drift: float = 0.0,
) -> dict[str, Any]:
    """
    Run a Monte Carlo simulation using GBM.

    Parameters
    ----------
    current_price : Starting price S(0).
    volatility    : Annualised volatility (σ), e.g. 0.30 = 30 %.
    days          : Number of trading days to project.
    simulations   : Number of simulated paths.
    drift         : Annualised expected return (μ). Default 0 (risk-neutral).

    Returns
    -------
    dict with keys: p10, p50, p90, mean  (all in price units)
    """
    try:
        result = _simulate_numpy(current_price, volatility, days, simulations, drift)
        result["engine"] = "numpy"
        logger.info("Monte Carlo math engine=numpy")
        return result
    except ImportError:
        result = _simulate_pure_python(
            current_price, volatility, days, simulations, drift
        )
        result["engine"] = "pure_python"
        logger.info("Monte Carlo math engine=pure_python")
        return result


def _simulate_numpy(
    current_price: float,
    volatility: float,
    days: int,
    simulations: int,
    drift: float,
) -> dict[str, float]:
    import numpy as np  # type: ignore

    dt = 1.0 / 252.0
    # Shape: (simulations, days)
    Z = np.random.standard_normal((simulations, days))
    log_returns = (drift - 0.5 * volatility**2) * dt + volatility * math.sqrt(dt) * Z
    # Cumulative product along time axis
    price_paths = current_price * np.exp(np.cumsum(log_returns, axis=1))
    final_prices = price_paths[:, -1]

    return {
        "p10": float(np.percentile(final_prices, 10)),
        "p50": float(np.percentile(final_prices, 50)),
        "p90": float(np.percentile(final_prices, 90)),
        "mean": float(np.mean(final_prices)),
    }


def _simulate_pure_python(
    current_price: float,
    volatility: float,
    days: int,
    simulations: int,
    drift: float,
) -> dict[str, float]:
    """Fallback GBM implementation that requires only the stdlib."""
    dt = 1.0 / 252.0
    half_vol_sq_dt = 0.5 * volatility**2 * dt
    vol_sqrt_dt = volatility * math.sqrt(dt)
    drift_dt = drift * dt

    finals: list[float] = []
    for _ in range(simulations):
        price = current_price
        for _ in range(days):
            z = random.gauss(0.0, 1.0)
            price *= math.exp(drift_dt - half_vol_sq_dt + vol_sqrt_dt * z)
        finals.append(price)

    finals.sort()
    n = len(finals)
    return {
        "p10": finals[int(n * 0.10)],
        "p50": finals[int(n * 0.50)],
        "p90": finals[min(int(n * 0.90), n - 1)],
        "mean": sum(finals) / n,
    }


# ── CLI entry-point (also used by ironclad-runtime as the script target) ──────


def _cli() -> None:
    """
    Minimal argument parser so this file can be executed directly:

        python compute/monte_carlo.py \\
            --price 162.45 --volatility 0.35 --days 30 --simulations 10000
    """
    import argparse

    parser = argparse.ArgumentParser(description="GBM Monte Carlo simulator")
    parser.add_argument(
        "--price", type=float, required=True, help="Current stock price"
    )
    parser.add_argument(
        "--volatility", type=float, default=0.30, help="Annual volatility"
    )
    parser.add_argument(
        "--days", type=int, required=True, help="Trading days to simulate"
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument(
        "--drift", type=float, default=0.0, help="Annual drift (risk-neutral=0)"
    )
    args = parser.parse_args()

    result = simulate(
        current_price=args.price,
        volatility=args.volatility,
        days=args.days,
        simulations=args.simulations,
        drift=args.drift,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
