"""The report's figure set, numbered in the order the paper refers to them.

Each figure is generated from the study's own outputs, so a figure cannot drift
away from the table it illustrates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..study import StudyResults
from .accuracy import plot_directional_accuracy, plot_r2_heatmap
from .design import plot_walk_forward_design
from .inference import plot_dm_versus_clark_west, plot_multiple_testing
from .performance import plot_equity, plot_phase_sensitivity, plot_sharpe_intervals

__all__ = [
    "generate_figures",
    "plot_directional_accuracy",
    "plot_dm_versus_clark_west",
    "plot_equity",
    "plot_multiple_testing",
    "plot_phase_sensitivity",
    "plot_r2_heatmap",
    "plot_sharpe_intervals",
    "plot_walk_forward_design",
]

#: Figure number -> (filename, caption) in the order the report presents them.
FIGURES: tuple[tuple[str, str], ...] = (
    ("fig1_design.png", "Purged, embargoed walk-forward design."),
    ("fig2_r2.png", "Out-of-sample $R^2$ against the drift benchmark."),
    ("fig3_diracc.png", "Directional accuracy against a coin flip."),
    ("fig4_dm_vs_cw.png", "Diebold-Mariano against Clark-West."),
    ("fig5_multiple_testing.png", "Clark-West p-values under multiple-testing correction."),
    ("fig6_equity.png", "Out-of-sample equity, net of costs."),
    ("fig7_sharpe.png", "Net Sharpe with bootstrap intervals."),
    ("fig8_phase.png", "Sensitivity to the trading schedule's start offset."),
)


def _design_sample_size(study: StudyResults, horizon: int) -> int:
    """Labeled row count for the first asset, so Figure 1 shows a real fold layout."""
    first = next((run for run in study.runs if run.horizon == horizon), None)
    return len(first.oos) + study.config.wf.train_size if first else 1500


def generate_figures(study: StudyResults, table: pd.DataFrame, out_dir: Path) -> list[str]:
    """Write every figure; return paths relative to the reports directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    short, long = study.config.horizons[0], study.config.horizons[-1]

    # Drawn at the longest horizon: at h=1 the purge is a single bar and the
    # detail panel cannot show it apart from the embargo.
    plot_walk_forward_design(
        _design_sample_size(study, long), study.config.wf, long, out_dir / "fig1_design.png"
    )
    plot_r2_heatmap(table, out_dir / "fig2_r2.png")
    plot_directional_accuracy(table, out_dir / "fig3_diracc.png")
    plot_dm_versus_clark_west(table, out_dir / "fig4_dm_vs_cw.png")
    plot_multiple_testing(table, out_dir / "fig5_multiple_testing.png")
    plot_equity(study, short, out_dir / "fig6_equity.png")
    plot_sharpe_intervals(table, out_dir / "fig7_sharpe.png")
    plot_phase_sensitivity(study, long, out_dir / "fig8_phase.png")

    return [f"figures/{name}" for name, _ in FIGURES]
