"""Figures 4 and 5: what the two predictive-accuracy tests conclude, and what
survives once the size of the search is priced in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.registry import ML_NAMES
from .style import NEGATIVE, POSITIVE, REFERENCE_BLACK, finish, label

CRIT_TWO_SIDED = 1.959963985  # 5%, two-sided
CRIT_ONE_SIDED = 1.644853627  # 5%, one-sided


def _ml_rows(table: pd.DataFrame) -> pd.DataFrame:
    sub = table[table["model"].isin(ML_NAMES)].copy()
    sub["setting"] = (
        sub["asset"] + " h=" + sub["horizon"].astype(str) + "d " + sub["model"].map(label)
    )
    return sub


def plot_dm_versus_clark_west(table: pd.DataFrame, out_path: Path) -> None:
    """Each setting placed by its DM statistic against its Clark-West statistic.

    The two tests disagree by construction when the models are nested, and the
    scatter shows the disagreement rather than asserting it.
    """
    sub = _ml_rows(table)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    ax.axhspan(CRIT_ONE_SIDED, 100, color=POSITIVE, alpha=0.08, lw=0)
    ax.axvspan(CRIT_TWO_SIDED, 100, color=NEGATIVE, alpha=0.08, lw=0)
    ax.axhline(CRIT_ONE_SIDED, color=POSITIVE, ls="--", lw=1.0)
    ax.axvline(CRIT_TWO_SIDED, color=NEGATIVE, ls="--", lw=1.0)
    ax.axhline(0, color=REFERENCE_BLACK, lw=0.8)
    ax.axvline(0, color=REFERENCE_BLACK, lw=0.8)

    for horizon, marker in ((1, "o"), (7, "s")):
        block = sub[sub["horizon"] == horizon]
        ax.scatter(
            block["dm_stat"],
            block["cw_stat"],
            marker=marker,
            s=46,
            edgecolor=REFERENCE_BLACK,
            linewidth=0.6,
            label=f"h = {horizon}d",
            zorder=3,
        )

    # Label only the settings Clark-West rescues; the losing cluster is described
    # in one annotation instead of six overlapping ones.
    rescued = sub[sub["cw_stat"] > CRIT_ONE_SIDED].sort_values("cw_stat", ascending=False)
    for offset, (_, row) in enumerate(rescued.iterrows()):
        ax.annotate(
            f"{row['asset']} {label(row['model'])}",
            (row["dm_stat"], row["cw_stat"]),
            textcoords="offset points",
            xytext=(8, 3 if offset % 2 == 0 else -9),
            fontsize=7,
            color="#333333",
        )

    losing = sub[(sub["dm_stat"] > CRIT_TWO_SIDED) & (sub["cw_stat"] < 0)]
    if not losing.empty:
        ax.annotate(
            f"{len(losing)} settings: both tests agree\nthe model is worse",
            (float(losing["dm_stat"].mean()), float(losing["cw_stat"].mean())),
            textcoords="offset points",
            xytext=(-104, -4),
            fontsize=7,
            color="#333333",
            ha="left",
        )

    x_pad, y_pad = 0.7, 0.55
    ax.set_xlim(sub["dm_stat"].min() - 1.4, sub["dm_stat"].max() + x_pad)
    ax.set_ylim(sub["cw_stat"].min() - y_pad, sub["cw_stat"].max() + y_pad)
    ax.set_xlabel("Diebold-Mariano statistic  (positive = model looks worse)")
    ax.set_ylabel("Clark-West statistic  (positive = model predicts)")
    ax.set_title("The nested-model correction moves the verdict")
    ax.legend(loc="upper right")
    finish(
        fig,
        out_path,
        "Red band: DM calls the model significantly worse. Green band: Clark-West "
        "rejects no-predictability. Points in both are the ones DM misreads.",
    )


def plot_multiple_testing(table: pd.DataFrame, out_path: Path) -> None:
    """Sorted Clark-West p-values against the Holm and Benjamini-Hochberg lines."""
    sub = _ml_rows(table).dropna(subset=["cw_p_vs_rw"]).sort_values("cw_p_vs_rw")
    p = sub["cw_p_vs_rw"].to_numpy(dtype=float)
    m = p.size
    ranks = np.arange(1, m + 1)
    alpha = 0.05

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [POSITIVE if v < alpha else "#B8B8B8" for v in p]
    ax.bar(ranks, p, color=colors, width=0.7, zorder=2)
    ax.axhline(alpha, color=REFERENCE_BLACK, ls=":", lw=1.2, label=r"uncorrected $\alpha=0.05$")
    ax.plot(
        ranks,
        ranks * alpha / m,
        color=NEGATIVE,
        lw=1.5,
        marker="o",
        markersize=3,
        label="Benjamini-Hochberg (FDR 5%)",
    )
    ax.plot(
        ranks,
        alpha / (m - ranks + 1),
        color="#0072B2",
        lw=1.5,
        ls="--",
        marker="s",
        markersize=3,
        label="Holm-Bonferroni (FWER 5%)",
    )

    n_raw = int((p < alpha).sum())
    n_bh = int((sub["cw_p_bh"] < alpha).sum())
    n_holm = int((sub["cw_p_holm"] < alpha).sum())

    # Log scale: the whole decision happens below p = 0.05, which is invisible on
    # a linear axis that has to reach 0.96.
    ax.set_yscale("log")
    ax.set_ylim(min(float(p.min()) * 0.6, 1e-3), 1.6)
    ax.set_xticks(
        ranks,
        [f"{r['asset']} {label(r['model'])}" for _, r in sub.iterrows()],
        rotation=45,
        ha="right",
        size=7,
    )
    ax.set_ylabel("Clark-West p-value (log scale)")
    ax.set_xlabel(f"the {m} ML settings, ranked by p-value", labelpad=2)
    ax.set_title(
        f"Multiple testing: {n_raw} raw rejections, {n_bh} survive BH, {n_holm} survive Holm"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    finish(
        fig,
        out_path,
        "A bar must fall below a line to be rejected by that procedure. Testing "
        f"{m} settings at 5% is expected to produce {alpha * m:.1f} findings from noise alone.",
    )
