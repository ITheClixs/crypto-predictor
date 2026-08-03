"""Immutable configuration objects for the study.

Every knob that affects a reported number lives here so a run is fully described
by a single ``StudyConfig``. All dataclasses are frozen: configuration is data,
not state, and should never be mutated in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostModel:
    """Round-trip trading friction, expressed per side in basis points.

    A basis point is 1e-4. Costs are charged on *turnover*: rebalancing the
    position by a fraction ``dw`` of capital incurs ``dw * cost_per_side``.
    Defaults are deliberately conservative for liquid spot crypto.
    """

    fee_bps: float = 10.0  # exchange taker fee, ~0.10%
    slippage_bps: float = 5.0  # execution slippage / market impact
    half_spread_bps: float = 2.0  # half the quoted bid-ask spread

    @property
    def cost_per_side(self) -> float:
        """Total friction paid to trade one unit of capital, as a fraction."""
        return (self.fee_bps + self.slippage_bps + self.half_spread_bps) * 1e-4


@dataclass(frozen=True)
class WalkForwardConfig:
    """Rolling/expanding walk-forward split geometry (in trading days)."""

    train_size: int = 504  # ~2 years of daily bars
    test_size: int = 63  # ~1 quarter
    embargo: int = 5  # bars purged between train end and test start
    mode: str = "expanding"  # "expanding" | "rolling"
    min_train: int = 252  # smallest acceptable training window

    def __post_init__(self) -> None:
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {self.mode!r}")
        for name in ("train_size", "test_size", "min_train"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embargo < 0:
            raise ValueError("embargo must be non-negative")


@dataclass(frozen=True)
class StudyConfig:
    """Top-level description of a reproducible study run."""

    assets: tuple[str, ...] = ("BTC", "ETH", "SOL")
    horizons: tuple[int, ...] = (1, 7)
    start: str = "2019-01-01"
    #: Pinned, not ``None``. Resolving the end date to "today" meant that any run
    #: missing the committed cache silently re-dated the whole study, which is how
    #: one Monte Carlo pass ended up measured on a sample twelve bars longer than
    #: the manuscript's. Override explicitly to extend the window.
    end: str | None = "2026-07-18"
    interval: str = "1d"
    seed: int = 7
    costs: CostModel = field(default_factory=CostModel)
    wf: WalkForwardConfig = field(default_factory=WalkForwardConfig)


DEFAULT_CONFIG = StudyConfig()
