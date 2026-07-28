"""Figure 1: the purged, embargoed walk-forward design.

Drawn from the splits the study actually uses, not an illustration of them, so
the picture cannot drift away from the code that produced the results.

Two panels, because the design operates at two scales: the fold layout spans
thousands of bars, while the gap that prevents the leak is a handful of bars
wide and would be a hairline on the full timeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..config import WalkForwardConfig
from ..splits import walk_forward_splits
from .style import NEGATIVE, POSITIVE, finish

TRAIN_COLOR = "#0072B2"
PURGE_COLOR = NEGATIVE
EMBARGO_COLOR = "#E69F00"
TEST_COLOR = POSITIVE


def _draw_all_folds(ax: plt.Axes, splits: list, n_samples: int) -> None:
    """Every fold as a train / gap / test band on one shared timeline."""
    for row, (train, test) in enumerate(splits):
        y = len(splits) - row
        ax.barh(y, train.max() - train.min() + 1, left=train.min(), height=0.9, color=TRAIN_COLOR)
        ax.barh(
            y, test.min() - train.max() - 1, left=train.max() + 1, height=0.9, color=PURGE_COLOR
        )
        ax.barh(y, test.max() - test.min() + 1, left=test.min(), height=0.9, color=TEST_COLOR)

    ticks = [t for t in range(1, len(splits) + 1) if t % 5 == 0 or t == 1]
    ax.set_yticks(ticks, [f"fold {len(splits) - t + 1}" for t in ticks])
    ax.set_xlabel("bar index (chronological)")
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0.4, len(splits) + 0.6)
    ax.set_title(f"(a)  {len(splits)} expanding folds over {n_samples} bars", loc="left")


def _draw_gap_detail(ax: plt.Axes, split: tuple, horizon: int, embargo: int) -> None:
    """Zoom on one train/test boundary, where the gap is actually visible."""
    train, test = split
    train_end, test_start = int(train.max()), int(test.min())
    window = max(3 * (horizon + embargo), 18)
    left, right = train_end - window, test_start + window

    ax.barh(0, train_end - left + 1, left=left, height=0.55, color=TRAIN_COLOR)
    ax.barh(0, horizon, left=train_end + 1, height=0.55, color=PURGE_COLOR)
    ax.barh(0, embargo, left=train_end + 1 + horizon, height=0.55, color=EMBARGO_COLOR)
    ax.barh(0, right - test_start, left=test_start, height=0.55, color=TEST_COLOR)

    # The label of the last training row is realized h bars later, inside the gap.
    ax.annotate(
        "",
        xy=(train_end + horizon, 0.48),
        xytext=(train_end, 0.48),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.1},
    )
    ax.text(
        train_end + horizon / 2,
        0.62,
        f"last train label\nresolves {horizon} bar{'s' if horizon > 1 else ''} later",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    ax.axvline(test_start, color="#333333", lw=1.0, ls=":")
    ax.text(test_start, -0.62, " test begins", fontsize=7.5, va="center")

    ax.set_yticks([])
    ax.set_ylim(-0.9, 1.15)
    ax.set_xlim(left, right)
    ax.set_xlabel("bar index")
    ax.set_title("(b)  the boundary in detail", loc="left")
    ax.grid(False)


def plot_walk_forward_design(
    n_samples: int, wf: WalkForwardConfig, horizon: int, out_path: Path
) -> None:
    """Fold layout and the purged gap that makes each fold honest."""
    splits = walk_forward_splits(
        n_samples,
        train_size=wf.train_size,
        test_size=wf.test_size,
        horizon=horizon,
        embargo=wf.embargo,
        mode=wf.mode,
        min_train=wf.min_train,
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), height_ratios=[2.6, 1.0])
    _draw_all_folds(axes[0], splits, n_samples)
    _draw_gap_detail(axes[1], splits[0], horizon, wf.embargo)

    fig.legend(
        handles=[
            Patch(color=TRAIN_COLOR, label="train"),
            Patch(color=PURGE_COLOR, label=f"purge ({horizon} bars)"),
            Patch(color=EMBARGO_COLOR, label=f"embargo ({wf.embargo} bars)"),
            Patch(color=TEST_COLOR, label="test (out-of-sample)"),
        ],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    finish(
        fig,
        out_path,
        "No training label is realized on or after the first test bar, which is what "
        f"the {horizon + wf.embargo}-bar gap buys.",
    )
