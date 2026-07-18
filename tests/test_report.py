"""Reporting: results table, honest headline logic, markdown, and figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_study
from cryptoforecast.evaluate.report import results_table, write_markdown
from cryptoforecast.plots import generate_figures


@pytest.mark.unit
def test_results_table_columns_and_rw_trades() -> None:
    table = results_table(make_study())
    assert {"dm_stat", "pt_stat", "sharpe_net", "dsr", "trades"} <= set(table.columns)
    rw = table[table["model"] == "random_walk"].iloc[0]
    assert rw["trades"] == 0  # a zero forecast never actually trades


@pytest.mark.unit
def test_headline_counts_only_genuine_wins(tmp_path: Path) -> None:
    study = make_study()
    table = results_table(study)
    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    # ridge: dm_stat < 0 and p < 0.05 -> exactly one genuine win.
    assert "**1** beat the random walk" in md
    assert "does anything beat the random walk" in md


@pytest.mark.unit
def test_generate_figures_writes_pngs(tmp_path: Path) -> None:
    study = make_study()
    table = results_table(study)
    figures = generate_figures(study, table, tmp_path / "figures")
    assert len(figures) == 3
    for fig in figures:
        assert (tmp_path / fig).exists()
