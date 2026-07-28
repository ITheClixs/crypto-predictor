"""Reporting: results table, honest headline logic, markdown, and figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conftest import make_study
from cryptoforecast.evaluate.report import BUY_AND_HOLD, results_table, write_markdown
from cryptoforecast.plots import generate_figures


@pytest.mark.unit
def test_results_table_columns_and_rw_trades() -> None:
    table = results_table(make_study())
    expected = {"dm_stat", "cw_stat", "pt_stat", "sharpe_net", "dsr", "trades", "oos_start"}
    assert expected <= set(table.columns)
    rw = table[table["model"] == "random_walk"].iloc[0]
    assert rw["trades"] == 0  # a zero forecast never actually trades


@pytest.mark.unit
def test_results_table_carries_the_buy_and_hold_reference() -> None:
    table = results_table(make_study())
    reference = table[table["model"] == BUY_AND_HOLD]
    assert len(reference) == 1  # one per (asset, horizon)
    row = reference.iloc[0]
    assert row["trades"] == 1  # enter once and hold
    assert pd.isna(row["rmse"])  # a reference, not a forecaster
    assert pd.isna(row["dsr"])  # and not part of the deflated model search


@pytest.mark.unit
def test_headline_counts_only_genuine_wins(tmp_path: Path) -> None:
    study = make_study()
    table = results_table(study)
    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    # ridge: dm_stat < 0 and p < 0.05 -> exactly one genuine win.
    assert "**1** beat the random walk" in md
    assert "does anything beat the random walk" in md
    # The deflation count must name the models raced, not some other grouping.
    assert f"the {int(table['n_trials'].max())}-model race" in md


@pytest.mark.unit
def test_a_significantly_worse_model_is_not_counted_as_a_win(tmp_path: Path) -> None:
    """The failure mode this report format exists to prevent.

    A model with a tiny p-value in the *wrong* direction must be counted as
    worse, and the rendered table must show the sign next to the p-value.
    """
    study = make_study()
    table = results_table(study)
    ridge = table["model"] == "ridge"
    table.loc[ridge, ["dm_stat", "dm_p_vs_rw"]] = [3.2, 0.001]  # significantly worse
    table.loc[ridge, ["pt_stat", "pt_p"]] = [-3.3, 0.001]  # reliably wrong-way

    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    assert "**0** beat the random walk" in md
    assert "**1** were significantly *worse*" in md
    assert "+3.20 (0.001)" in md  # sign is rendered, not just the p-value
    assert "-3.30 (0.001)" in md


@pytest.mark.unit
def test_multiple_testing_columns_cover_the_ml_family_only() -> None:
    """Benchmarks are the null being tested against, not candidates in the search."""
    table = results_table(make_study())
    assert {"cw_p_holm", "cw_p_bh"} <= set(table.columns)
    assert pd.isna(table.loc[table["model"] == "random_walk", "cw_p_holm"]).all()
    ridge = table.loc[table["model"] == "ridge"].iloc[0]
    # One ML setting in this fixture, so the correction is a no-op on it.
    assert ridge["cw_p_holm"] == pytest.approx(ridge["cw_p_vs_rw"])
    assert ridge["cw_p_bh"] == pytest.approx(ridge["cw_p_vs_rw"])


@pytest.mark.unit
def test_interpretation_reports_no_evidence_when_nothing_survives(tmp_path: Path) -> None:
    study = make_study()
    table = results_table(study)
    table["cw_p_bh"] = 0.9  # nothing survives correction
    table["sharpe_lo"] = -1.0  # and nothing is profitably distinguishable from zero
    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    assert "no evidence of predictability" in md


@pytest.mark.unit
def test_interpretation_flags_when_the_significant_and_the_profitable_differ(
    tmp_path: Path,
) -> None:
    """Statistical winners that are not the economic winners are the noise signature."""
    study = make_study()
    table = results_table(study)
    ridge = table["model"] == "ridge"
    table.loc[ridge, "cw_p_bh"] = 0.01  # significant after correction
    table.loc[ridge, "sharpe_lo"] = -0.5  # but not profitable
    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    assert "no setting is in both groups" in md


@pytest.mark.unit
def test_interpretation_explains_a_survivor_with_negative_out_of_sample_r2(
    tmp_path: Path,
) -> None:
    """Clark-West can reject while the realized forecast still loses to a constant."""
    study = make_study()
    table = results_table(study)
    ridge = table["model"] == "ridge"
    table.loc[ridge, "cw_p_bh"] = 0.01
    table.loc[ridge, "r2_oos"] = -0.01
    write_markdown(study.config, table, [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    assert "one of these has a *negative* out-of-sample" in md
    assert "not of a usable forecast" in md


@pytest.mark.unit
def test_markdown_documents_every_sign_convention(tmp_path: Path) -> None:
    study = make_study()
    write_markdown(study.config, results_table(study), [], tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    assert "**Negative favours the model**" in md  # DM
    assert "**Positive favours the model.**" in md  # CW
    assert "**Positive means sign-timing skill**" in md  # PT
    assert "buy_and_hold" in md


@pytest.mark.unit
def test_generate_figures_writes_pngs(tmp_path: Path) -> None:
    study = make_study()
    table = results_table(study)
    figures = generate_figures(study, table, tmp_path / "figures")
    assert len(figures) == 3
    for fig in figures:
        assert (tmp_path / fig).exists()
