"""Emit the paper's result tables as LaTeX, straight from the results table.

The manuscript in ``paper/`` includes these rather than restating numbers in
prose, for the same reason ``reports/results.md`` is generated: a table that is
typed by hand drifts from the study the moment the study is rerun, and nobody
notices until a reader checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.registry import BENCHMARK_NAMES, ML_NAMES

#: Display names for the model column.
MODEL_LABELS: dict[str, str] = {
    "random_walk": "Random walk",
    "historical_mean": "Historical mean",
    "ar1": "AR(1)",
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "gbm": "GBM",
    "buy_and_hold": "Buy \\& hold",
}

_ORDER = (*BENCHMARK_NAMES, *ML_NAMES, "buy_and_hold")


def _num(value: float, spec: str = ".3f", dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return dash
    return format(value, spec)


def _signed(stat: float, p_value: float) -> str:
    """``statistic (p)``, with the statistic's sign always shown."""
    if stat is None or np.isnan(stat):
        return "--"
    body = f"{stat:+.2f}\\,({_num(p_value)})"
    return f"\\textbf{{{body}}}" if p_value < 0.05 else body


def _rows_in_order(block: pd.DataFrame) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(_ORDER)}
    return block.assign(_o=block["model"].map(rank)).sort_values("_o").drop(columns="_o")


def _tabular(body: list[str], spec: str, header: list[str]) -> str:
    lines = [
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def data_summary_table(table: pd.DataFrame) -> str:
    """Assets, evaluated window, and out-of-sample counts."""
    body = []
    for (asset, horizon), block in table.groupby(["asset", "horizon"]):
        row = block[block["model"] == "ridge"]
        if row.empty:
            continue
        r = row.iloc[0]
        body.append(
            f"{asset} & {int(horizon)} & {int(r['n_oos'])} & "
            f"{r['oos_start']} & {r['oos_end']} \\\\"
        )
    header = ["Asset", "$h$ (days)", "OOS forecasts", "First", "Last"]
    return _tabular(body, "llrll", header)


def accuracy_table(table: pd.DataFrame) -> str:
    """Point accuracy and the predictive-accuracy tests, every setting.

    Clark-West appears twice, against the zero forecast and against the recursively
    estimated mean. They are different hypotheses, not a robustness check: only the
    second is the slopes-zero restriction of an estimator that fits an intercept, and
    on this data they disagree about which settings reject.
    """
    body: list[str] = []
    for (asset, horizon), block in table.groupby(["asset", "horizon"]):
        body.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{asset}, $h={int(horizon)}$}}}} \\\\")
        for _, r in _rows_in_order(block[block["model"] != "buy_and_hold"]).iterrows():
            body.append(
                f"\\quad {MODEL_LABELS.get(r['model'], r['model'])} & "
                f"{_num(r['rmse'], '.4f')} & {_num(r['r2_oos'], '.4f')} & "
                f"{_num(r['dir_acc'])} & {_num(r['rank_ic'])} & "
                f"{_signed(r['dm_stat'], r['dm_p_vs_rw'])} & "
                f"{_signed(r['cw_stat'], r['cw_p_vs_rw'])} & "
                f"{_signed(r.get('cw_stat_vs_mean'), r.get('cw_p_vs_mean'))} & "
                f"{_signed(r.get('sign_excess'), r.get('sign_p'))} \\\\"
            )
        body.append("\\addlinespace")
    header = [
        "Model",
        "RMSE",
        "$R^2_{OS}$",
        "Dir.\\ acc.",
        "Rank IC",
        "DM $(p)$",
        "CW vs.\\ 0 $(p)$",
        "CW vs.\\ mean $(p)$",
        "Sign excess $(p)$",
    ]
    return _tabular(body[:-1], "lrrrrrrrr", header)


def performance_table(table: pd.DataFrame) -> str:
    """Net-of-cost performance with intervals, deflation, and phase spread."""
    body: list[str] = []
    for (asset, horizon), block in table.groupby(["asset", "horizon"]):
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{asset}, $h={int(horizon)}$}}}} \\\\")
        for _, r in _rows_in_order(block).iterrows():
            phases = (
                "--"
                if horizon == 1 or np.isnan(r.get("sharpe_phase_lo", np.nan))
                else f"[{_num(r['sharpe_phase_lo'], '.2f')}, {_num(r['sharpe_phase_hi'], '.2f')}]"
            )
            body.append(
                f"\\quad {MODEL_LABELS.get(r['model'], r['model'])} & "
                f"{_num(r['sharpe_net'], '.2f')} & "
                f"[{_num(r['sharpe_lo'], '.2f')}, {_num(r['sharpe_hi'], '.2f')}] & "
                f"{phases} & {_num(r['max_dd'] * 100, '.1f')}\\% & "
                f"{_num(r['psr'], '.2f')} & {_num(r['dsr'], '.2f')} \\\\"
            )
        body.append("\\addlinespace")
    header = ["Model", "Sharpe", "95\\% CI", "Phases", "Max DD", "PSR", "DSR"]
    return _tabular(body[:-1], "lrrrrrr", header)


def inference_summary_table(table: pd.DataFrame) -> str:
    """How many of the ML settings each test and correction rejects, by benchmark.

    The benchmark is named in every row because the two columns test different hypotheses:
    against the zero forecast the null is "no drift and no conditional predictability", against
    the recursively estimated mean it is "the features add nothing beyond an intercept". A
    single "Clark--West" row would hide the only distinction the paper is about.

    The old "expected by chance" footer reported ``0.05 n`` regardless of the measured size of
    the test, which is the arithmetic this paper exists to retract. It is gone; calibrated
    expectations and the joint null distribution live in the calibration section.
    """
    ml = table[table["model"].isin(ML_NAMES)]
    n = len(ml)

    def _count(mask: pd.Series) -> str:
        return f"{int(mask.sum())} / {n}"

    rows = [
        (
            "Diebold--Mariano vs.\\ zero, favours model",
            _count((ml["dm_stat"] < 0) & (ml["dm_p_vs_rw"] < 0.05)),
        ),
        (
            "Diebold--Mariano vs.\\ zero, favours benchmark",
            _count((ml["dm_stat"] > 0) & (ml["dm_p_vs_rw"] < 0.05)),
        ),
        ("\\addlinespace", ""),
        ("Clark--West vs.\\ zero forecast, uncorrected", _count(ml["cw_p_vs_rw"] < 0.05)),
        ("Clark--West vs.\\ recursive mean, uncorrected", _count(ml["cw_p_vs_mean"] < 0.05)),
        ("\\addlinespace", ""),
        ("Clark--West vs.\\ zero, Benjamini--Hochberg", _count(ml["cw_p_bh"] < 0.05)),
        ("Clark--West vs.\\ zero, Holm--Bonferroni", _count(ml["cw_p_holm"] < 0.05)),
        ("\\addlinespace", ""),
        ("Sign rotation test, skill", _count((ml["sign_excess"] > 0) & (ml["sign_p"] < 0.05))),
        (
            "Sign rotation test, anti-predictive",
            _count((ml["sign_excess"] < 0) & (ml["sign_p"] < 0.05)),
        ),
        ("Net Sharpe interval excludes zero", _count(ml["sharpe_lo"] > 0)),
    ]
    body = []
    for name, count in rows:
        blank = name == "\\addlinespace"
        body.append("\\addlinespace" if blank else f"{name} & {count} \\\\")
    return _tabular(body, "lr", ["Criterion (all at a nominal 5\\%, $h-1$ bandwidth)", "Settings"])


def all_tables(table: pd.DataFrame) -> dict[str, str]:
    """Filename stem to LaTeX ``tabular`` body."""
    return {
        "data_summary": data_summary_table(table),
        "accuracy": accuracy_table(table),
        "performance": performance_table(table),
        "inference_summary": inference_summary_table(table),
    }
