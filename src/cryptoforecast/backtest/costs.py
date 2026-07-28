"""Turnover-based trading frictions.

A signal can be right about direction and still lose money once you pay to trade
it. Costs here are charged on turnover — the absolute change in position — so a
strategy that flips between long and short every period pays twice as much as one
that holds.
"""

from __future__ import annotations

import pandas as pd


def turnover(positions: pd.Series) -> pd.Series:
    """Absolute change in position per period.

    The first period is treated as entering from flat, so its turnover is the
    absolute initial position.
    """
    return positions.diff().abs().fillna(positions.abs())


def trading_costs(positions: pd.Series, cost_per_side: float) -> pd.Series:
    """Fraction of capital lost to friction each period."""
    return turnover(positions) * cost_per_side
