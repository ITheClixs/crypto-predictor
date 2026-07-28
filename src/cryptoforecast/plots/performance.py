"""Figures 6 to 8: net-of-cost performance, its error bars, and its fragility."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.registry import ML_NAMES
from ..study import ModelRun, StudyResults
from .style import (
    MODEL_ORDER,
    NEGATIVE,
    POSITIVE,
    REFERENCE_BLACK,
    color,
    finish,
    label,
)


def _by_key(study: StudyResults) -> dict[tuple[str, int, str], ModelRun]:
    return {(r.asset, r.horizon, r.model): r for r in study.runs}


def plot_equity(study: StudyResults, horizon: int, out_path: Path) -> None:
    """Net equity per model against the always-long reference, one panel per asset."""
    runs = _by_key(study)
    assets = study.config.assets
    fig, axes = plt.subplots(1, len(assets), figsize=(4.4 * len(assets), 3.7), squeeze=False)

    for ax, asset in zip(axes[0], assets, strict=False):
        for model in MODEL_ORDER:
            run = runs.get((asset, horizon, model))
            if run is None:
                continue
            equity = run.strategy.equity
            ax.plot(
                equity.index,
                equity.to_numpy(),
                label=label(model),
                color=color(model),
                lw=1.2,
                alpha=0.55 if model not in ML_NAMES else 1.0,
            )
        reference = study.buy_and_hold.get((asset, horizon))
        if reference is not None:
            ax.plot(
                reference.equity.index,
                reference.equity.to_numpy(),
                label="Buy & hold",
                color=REFERENCE_BLACK,
                ls="--",
                lw=1.3,
            )
        ax.set_yscale("log")
        ax.set_title(f"{asset}, h = {horizon}d")
        ax.set_ylabel("net equity (log scale)")
        ax.tick_params(axis="x", rotation=30)

    axes[0][0].legend(loc="upper left", ncol=2)
    fig.suptitle("Out-of-sample equity, net of costs", fontweight="bold")
    fig.tight_layout()
    finish(
        fig,
        out_path,
        "Starting capital 1.0. Buy & hold pays the same costs on the same schedule.",
    )


def plot_sharpe_intervals(table: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of net Sharpe with bootstrap intervals, per asset-horizon block."""
    ml = table[table["model"].isin(ML_NAMES)].copy()
    reference = {
        (r["asset"], r["horizon"]): r["sharpe_net"]
        for _, r in table[table["model"] == "buy_and_hold"].iterrows()
    }
    blocks = sorted({(a, h) for a, h in zip(ml["asset"], ml["horizon"], strict=False)})

    fig, axes = plt.subplots(
        1, len(blocks), figsize=(1.85 * len(blocks) + 1.4, 3.9), squeeze=False, sharey=True
    )
    for ax, (asset, horizon) in zip(axes[0], blocks, strict=False):
        block = ml[(ml["asset"] == asset) & (ml["horizon"] == horizon)]
        block = block.set_index("model").loc[[m for m in ML_NAMES if m in set(block["model"])]]
        y = np.arange(len(block))
        for i, (model, row) in enumerate(block.iterrows()):
            excludes_zero = row["sharpe_lo"] > 0
            ax.plot(
                [row["sharpe_lo"], row["sharpe_hi"]],
                [i, i],
                color=POSITIVE if excludes_zero else "#9A9A9A",
                lw=2.0,
                solid_capstyle="round",
            )
            ax.plot(
                row["sharpe_net"],
                i,
                "o",
                color=color(str(model)),
                markersize=6,
                markeredgecolor=REFERENCE_BLACK,
                markeredgewidth=0.5,
                zorder=3,
            )
        ax.axvline(0, color=REFERENCE_BLACK, lw=0.9)
        bnh = reference.get((asset, horizon))
        if bnh is not None:
            ax.axvline(bnh, color=NEGATIVE, ls="--", lw=1.1)
        ax.set_yticks(y, [label(str(m)) for m in block.index])
        ax.set_ylim(len(block) - 0.5, -0.5)  # first model on top, as in the tables
        ax.set_title(f"{asset}\nh = {horizon}d", fontsize=9)
        ax.set_xlabel("net Sharpe")

    fig.suptitle("Net Sharpe with 95% circular-block-bootstrap intervals", fontweight="bold")
    fig.tight_layout()
    finish(
        fig,
        out_path,
        "Dashed red line is buy & hold on the same schedule. Green intervals exclude zero.",
    )


def plot_phase_sensitivity(study: StudyResults, horizon: int, out_path: Path) -> None:
    """Net Sharpe on each of the ``horizon`` possible schedule start offsets."""
    runs = [r for r in study.runs if r.horizon == horizon and r.model in ML_NAMES]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    for i, run in enumerate(runs):
        phases = np.asarray(run.phase_sharpes, dtype=float)
        phases = phases[np.isfinite(phases)]
        if phases.size == 0:
            continue
        ax.plot([i, i], [phases.min(), phases.max()], color="#B8B8B8", lw=1.6, zorder=1)
        ax.scatter(np.full(phases.size, i), phases, s=22, color=color(run.model), zorder=2)
        ax.scatter(
            i,
            phases[0],
            s=64,
            facecolor="none",
            edgecolor=REFERENCE_BLACK,
            linewidth=1.2,
            zorder=3,
        )

    ax.axhline(0, color=REFERENCE_BLACK, lw=0.9)
    ax.set_xticks(
        range(len(runs)),
        [f"{r.asset}\n{label(r.model)}" for r in runs],
        fontsize=7.5,
    )
    ax.set_ylabel("net Sharpe")
    ax.set_title(f"Sensitivity to the trading schedule's start offset (h = {horizon}d)")
    finish(
        fig,
        out_path,
        "Each dot is one of the h non-overlapping schedules. The circled dot is the "
        "offset the results table happens to report; nothing privileges it.",
    )
