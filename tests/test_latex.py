"""The manuscript's tables are emitted from the results, not transcribed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_study
from cryptoforecast.evaluate.latex import (
    accuracy_table,
    all_tables,
    data_summary_table,
    inference_summary_table,
    performance_table,
)
from cryptoforecast.evaluate.report import results_table


@pytest.fixture
def table() -> pd.DataFrame:
    return results_table(make_study())


@pytest.mark.unit
def test_every_table_is_a_complete_tabular(table: pd.DataFrame) -> None:
    for name, body in all_tables(table).items():
        assert body.startswith("\\begin{tabular}"), name
        assert body.rstrip().endswith("\\end{tabular}"), name
        assert body.count("\\toprule") == 1, name
        assert body.count("\\bottomrule") == 1, name


def _separators(row: str) -> int:
    """Count column separators, ignoring escaped ampersands such as ``Buy \\& hold``."""
    return sum(1 for i, char in enumerate(row) if char == "&" and row[i - 1] != "\\")


@pytest.mark.unit
def test_column_counts_match_the_header(table: pd.DataFrame) -> None:
    """A row with the wrong number of separators is a LaTeX error, not a typo."""
    for name, body in all_tables(table).items():
        lines = [ln for ln in body.split("\n") if ln.endswith("\\\\")]
        header, *rows = lines
        expected = _separators(header)
        for row in rows:
            if "multicolumn" in row or not row.strip():
                continue
            assert _separators(row) == expected, f"{name}: {row}"


@pytest.mark.unit
def test_significant_entries_are_emphasised(table: pd.DataFrame) -> None:
    """Bold marks p < 0.05, so the reader sees which cells the text discusses."""
    body = accuracy_table(table)
    ridge = table[table["model"] == "ridge"].iloc[0]
    assert ridge["dm_p_vs_rw"] < 0.05
    assert f"{ridge['dm_stat']:+.2f}" in body
    assert "\\textbf{" in body


@pytest.mark.unit
def test_statistics_always_carry_their_sign(table: pd.DataFrame) -> None:
    """Same rule as the rendered report: never a bare p-value."""
    body = accuracy_table(table)
    for row in body.split("\n"):
        if "Ridge" not in row:
            continue
        # Each of the three test columns is "<signed stat>(p)" or an em dash.
        assert row.count("+") + row.count("-") >= 1


@pytest.mark.unit
def test_missing_values_render_as_dashes_not_nan(table: pd.DataFrame) -> None:
    for name, body in all_tables(table).items():
        assert "nan" not in body.lower(), name
        assert "None" not in body, name


@pytest.mark.unit
def test_buy_and_hold_appears_only_in_the_performance_table(table: pd.DataFrame) -> None:
    """It is a reference strategy, not a forecaster, so it has no accuracy row."""
    assert "Buy \\& hold" in performance_table(table)
    assert "Buy \\& hold" not in accuracy_table(table)


@pytest.mark.unit
def test_inference_summary_counts_match_the_table(table: pd.DataFrame) -> None:
    body = inference_summary_table(table)
    ml = table[table["model"].isin(["ridge", "elastic_net", "gbm"])]
    n = len(ml)
    beats = int(((ml["dm_stat"] < 0) & (ml["dm_p_vs_rw"] < 0.05)).sum())
    assert f"favours model & {beats} / {n}" in body


@pytest.mark.unit
def test_inference_summary_names_the_benchmark_on_every_clark_west_row(
    table: pd.DataFrame,
) -> None:
    """The two Clark-West rows test different nulls; an unlabelled row hides the distinction."""
    body = inference_summary_table(table)
    assert "Clark--West vs.\\ zero forecast" in body
    assert "Clark--West vs.\\ recursive mean" in body


@pytest.mark.unit
def test_inference_summary_drops_the_nominal_expected_count(table: pd.DataFrame) -> None:
    """``0.05 n`` assumes the test has its nominal size; this study measured that it does not.

    Printing 0.9 expected rejections beside a count produced by a test whose measured
    rejection rate is 14-30% is the arithmetic the paper exists to retract.
    """
    body = inference_summary_table(table)
    assert "Expected by chance" not in body


@pytest.mark.unit
def test_data_summary_reports_the_evaluated_window(table: pd.DataFrame) -> None:
    body = data_summary_table(table)
    row = table[table["model"] == "ridge"].iloc[0]
    assert str(row["oos_start"]) in body
    assert str(row["oos_end"]) in body
    assert str(int(row["n_oos"])) in body


@pytest.mark.unit
def test_percentages_and_ampersands_are_escaped(table: pd.DataFrame) -> None:
    """An unescaped % comments out the rest of the line and silently eats a row."""
    for name, body in all_tables(table).items():
        for i, char in enumerate(body):
            if char in "%&" and body[i - 1] != "\\":
                # A bare & is the column separator and is expected; % never is.
                assert char == "&", f"{name}: unescaped % at offset {i}"


@pytest.mark.unit
def test_accuracy_table_carries_both_clark_west_benchmarks(table: pd.DataFrame) -> None:
    """The manuscript's headline count is against the recursive mean, so it must be shown.

    Rendering only ``CW vs the zero forecast`` puts the appendix in contradiction with
    the abstract: those two statistics test different nulls and disagree about which
    settings reject.
    """
    rendered = all_tables(table)["accuracy"]
    assert "CW vs.\\ 0 $(p)$" in rendered
    assert "CW vs.\\ mean $(p)$" in rendered


@pytest.mark.unit
def test_tables_survive_a_row_with_missing_statistics(table: pd.DataFrame) -> None:
    """The random-walk row has no directional metrics; it must not crash rendering."""
    blanked = table.copy()
    for column in ("dm_stat", "cw_stat", "pt_stat", "sharpe_phase_lo", "dsr"):
        blanked[column] = np.nan
    rendered = all_tables(blanked)
    for name, body in rendered.items():
        assert "nan" not in body.lower(), name
    # The two tables carrying those columns fall back to dashes.
    assert "--" in rendered["accuracy"]
    assert "--" in rendered["performance"]
