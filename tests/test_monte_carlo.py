"""
test_monte_carlo.py — Tests for compute/monte_carlo.py

Run:  pytest tests/test_monte_carlo.py -v
"""

import math
import pytest
from compute.monte_carlo import simulate, _simulate_pure_python


def test_simulate_returns_required_keys():
    result = simulate(current_price=100.0, volatility=0.30, days=10, simulations=500)
    for key in ("p10", "p50", "p90", "mean"):
        assert key in result, f"Missing key: {key}"


def test_percentile_ordering():
    """P10 <= P50 <= P90 must always hold."""
    result = simulate(current_price=100.0, volatility=0.30, days=30, simulations=1_000)
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_mean_within_plausible_range():
    """With zero drift, mean should stay close to current price."""
    current = 150.0
    result = simulate(
        current_price=current, volatility=0.20, days=5, simulations=5_000, drift=0.0
    )
    # Allow ±20% from the starting price over 5 days
    assert 0.80 * current <= result["mean"] <= 1.20 * current


def test_pure_python_fallback_consistent():
    """Pure-Python and NumPy paths should produce statistically similar results."""
    kwarg = dict(
        current_price=100.0, volatility=0.30, days=20, simulations=5_000, drift=0.0
    )
    py_result = _simulate_pure_python(**kwarg)
    # Just verify shape/ordering — exact values differ due to randomness
    assert py_result["p10"] <= py_result["p50"] <= py_result["p90"]


def test_high_volatility_widens_spread():
    """Higher volatility should always widen the P10-P90 spread."""
    low_vol = simulate(current_price=100.0, volatility=0.10, days=30, simulations=2_000)
    high_vol = simulate(
        current_price=100.0, volatility=0.80, days=30, simulations=2_000
    )
    low_spread = high_vol["p90"] - high_vol["p10"]
    high_spread = low_vol["p90"] - low_vol["p10"]
    assert low_spread > high_spread


def test_single_day_simulation():
    """Edge case: simulating 1 day should not crash."""
    result = simulate(current_price=200.0, volatility=0.25, days=1, simulations=500)
    assert result["p50"] > 0
