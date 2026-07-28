"""Shared plotting style so every figure in the report reads as one document.

Headless Agg backend, a fixed colour assignment per model, and a serif-ish
default that survives being dropped into a PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models.registry import BENCHMARK_NAMES, ML_NAMES

#: Colour-blind-safe qualitative palette (Okabe-Ito), reused across all figures.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)

BENCHMARK_GREY = "#8C8C8C"
REFERENCE_BLACK = "#222222"
POSITIVE = "#009E73"
NEGATIVE = "#D55E00"

MODEL_ORDER: tuple[str, ...] = (*BENCHMARK_NAMES, *ML_NAMES)
MODEL_COLORS: dict[str, str] = {
    name: (BENCHMARK_GREY if name in BENCHMARK_NAMES else PALETTE[i % len(PALETTE)])
    for i, name in enumerate(MODEL_ORDER)
}
MODEL_COLORS["buy_and_hold"] = REFERENCE_BLACK

#: Human-readable model labels for axis ticks and legends.
MODEL_LABELS: dict[str, str] = {
    "random_walk": "Random walk",
    "historical_mean": "Hist. mean",
    "ar1": "AR(1)",
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "gbm": "GBM",
    "buy_and_hold": "Buy & hold",
}

_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


# Applied once at import, so every figure module inherits the same look.
plt.style.use(_RC)


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def color(model: str) -> str:
    return MODEL_COLORS.get(model, BENCHMARK_GREY)


def finish(fig: Figure, out_path: Path, caption: str | None = None) -> None:
    """Add an optional caption beneath the axes, save, and close the figure."""
    if caption:
        fig.supxlabel(caption, fontsize=7.5, color="#555555", y=-0.01)
    fig.savefig(out_path)
    plt.close(fig)
