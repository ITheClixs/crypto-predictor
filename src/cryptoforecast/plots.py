"""Figures for the results report. Headless (Agg), deterministic, theme-neutral."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models.registry import default_models
from .study import ModelRun, StudyResults

_MODELS = list(default_models())
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
_COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(_MODELS)}


def _by_key(study: StudyResults) -> dict[tuple[str, int, str], ModelRun]:
    return {(r.asset, r.horizon, r.model): r for r in study.runs}


def plot_equity(study: StudyResults, horizon: int, out_path: Path) -> None:
    runs = _by_key(study)
    assets = study.config.assets
    fig, axes = plt.subplots(1, len(assets), figsize=(5.2 * len(assets), 4.2), squeeze=False)
    for ax, asset in zip(axes[0], assets, strict=False):
        for model in _MODELS:
            run = runs.get((asset, horizon, model))
            if run is None:
                continue
            eq = run.strategy.equity
            ax.plot(eq.index, eq.to_numpy(), label=model, color=_COLORS[model], lw=1.3)
        # The same cost-charged always-long reference the results table reports,
        # not a raw price ratio — otherwise the comparison flatters buy & hold.
        reference = study.buy_and_hold.get((asset, horizon))
        if reference is not None:
            bh = reference.equity
            ax.plot(bh.index, bh.to_numpy(), label="buy & hold", color="black", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{asset} · h={horizon}d")
        ax.set_ylabel("net equity (log)")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Net-of-cost strategy equity vs buy & hold", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _grouped_bar(
    study: StudyResults,
    table_col: str,
    horizon: int,
    title: str,
    ylabel: str,
    reference: float | None,
    out_path: Path,
    table: pd.DataFrame,
) -> None:
    assets = study.config.assets
    fig, axes = plt.subplots(1, len(assets), figsize=(5.2 * len(assets), 4.0), squeeze=False)
    sub = table[table["horizon"] == horizon]
    for ax, asset in zip(axes[0], assets, strict=False):
        grp = sub[sub["asset"] == asset].set_index("model")
        models = [m for m in _MODELS if m in grp.index]
        values = [grp.loc[m, table_col] for m in models]
        colors = [_COLORS[m] for m in models]
        ax.bar(range(len(models)), values, color=colors)
        if reference is not None:
            ax.axhline(reference, color="black", ls="--", lw=1.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{asset} · h={horizon}d")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_figures(study: StudyResults, table: pd.DataFrame, out_dir: Path) -> list[str]:
    """Write the standard figures; return their paths relative to the reports dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon = study.config.horizons[0]

    plot_equity(study, horizon, out_dir / "fig_equity.png")
    _grouped_bar(
        study,
        "sharpe_net",
        horizon,
        "Net-of-cost annualized Sharpe",
        "Sharpe",
        0.0,
        out_dir / "fig_sharpe.png",
        table,
    )
    _grouped_bar(
        study,
        "dir_acc",
        horizon,
        "Directional accuracy (dashed = coin flip)",
        "hit rate",
        0.5,
        out_dir / "fig_diracc.png",
        table,
    )
    return ["figures/fig_equity.png", "figures/fig_sharpe.png", "figures/fig_diracc.png"]
