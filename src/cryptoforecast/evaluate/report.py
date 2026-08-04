"""Assemble the results table and render the reproducible results.md.

results.md is a generated artifact (committed as the study's evidence), not prose to
be hand-edited. The narrative lives in the top-level README.
"""

from __future__ import annotations

from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.strategy import StrategyResult
from ..config import StudyConfig
from ..models.registry import ML_NAMES, default_models
from ..plots import FIGURES
from ..study import R2_BENCHMARK, StudyResults
from .stats import (
    benjamini_hochberg_adjusted,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    holm_adjusted,
    is_degenerate,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

#: The always-long reference, carried in the results table as its own row so the
#: committed CSV is the complete evidence and the markdown is a pure rendering.
BUY_AND_HOLD = "buy_and_hold"

_MODEL_ORDER = [*default_models(), BUY_AND_HOLD]


def _per_period_sharpe(net: pd.Series) -> float:
    """Non-annualized Sharpe, the unit the deflation benchmark is expressed in."""
    r = net.to_numpy(dtype=float)
    return 0.0 if is_degenerate(r) else float(r.mean() / r.std(ddof=1))


def _pnl_fields(strategy: StrategyResult) -> dict[str, object]:
    """PnL columns shared by model rows and the buy-and-hold reference row."""
    net, ppy = strategy.net, strategy.periods_per_year
    # 500 draws left the interval endpoints themselves noisy, which matters because two of
    # them sit within 0.02 of zero. 10,000 puts the Monte Carlo error on those endpoints well
    # below the rounding shown in the tables.
    lo, hi = block_bootstrap_ci(net, partial(sharpe_ratio, periods_per_year=ppy), n_boot=10_000)
    return {
        "sharpe_net": sharpe_ratio(net, ppy),
        "sharpe_lo": lo,
        "sharpe_hi": hi,
        "max_dd": max_drawdown(strategy.equity),
        "psr": probabilistic_sharpe_ratio(net),
        "trades": int((strategy.turnover > 0).sum()),
    }


def results_table(study: StudyResults) -> pd.DataFrame:
    """One tidy row per (asset, horizon, model) with metrics, stats, and PnL."""
    group_sr: dict[tuple[str, int], list[float]] = {}
    for run in study.runs:
        group_sr.setdefault((run.asset, run.horizon), []).append(
            _per_period_sharpe(run.strategy.net)
        )

    rows: list[dict[str, object]] = []
    for run in study.runs:
        # The deflation is over the models raced against each other on this
        # (asset, horizon); the group is the selection set a picker would face.
        srs = group_sr[(run.asset, run.horizon)]
        trials_sr_std = float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0
        phases = [s for s in run.phase_sharpes if not np.isnan(s)]
        rows.append(
            {
                "asset": run.asset,
                "horizon": run.horizon,
                "model": run.model,
                "oos_start": run.oos.index.min().date().isoformat(),
                "oos_end": run.oos.index.max().date().isoformat(),
                "n_oos": len(run.oos),
                "rmse": run.metrics["rmse"],
                "r2_oos": run.metrics["r2_oos"],
                "r2_vs_sample_mean": run.metrics["r2_vs_sample_mean"],
                "dir_acc": run.metrics["dir_acc"],
                "rank_ic": run.metrics["rank_ic"],
                "dm_stat": run.dm_stat_vs_rw,
                "dm_p_vs_rw": run.dm_p_vs_rw,
                "cw_stat": run.cw_stat_vs_rw,
                "cw_p_vs_rw": run.cw_p_vs_rw,
                "cw_stat_vs_mean": run.cw_stat_vs_mean,
                "cw_p_vs_mean": run.cw_p_vs_mean,
                "pt_stat": run.pt_stat,
                "pt_p": run.pt_p,
                "sign_excess": run.sign_excess,
                "sign_p": run.sign_p,
                "sharpe_phase_lo": min(phases) if phases else float("nan"),
                "sharpe_phase_hi": max(phases) if phases else float("nan"),
                "dsr": deflated_sharpe_ratio(run.strategy.net, len(srs), trials_sr_std),
                "n_trials": len(srs),
                **_pnl_fields(run.strategy),
            }
        )

    for (asset, horizon), reference in study.buy_and_hold.items():
        # A reference, not a searched model: no forecast metrics and no deflation.
        rows.append(
            {
                "asset": asset,
                "horizon": horizon,
                "model": BUY_AND_HOLD,
                **_pnl_fields(reference),
            }
        )

    table = pd.DataFrame(rows)
    table = _add_multiple_testing_columns(table)
    order = {name: i for i, name in enumerate(_MODEL_ORDER)}
    table["_o"] = table["model"].map(order)
    return table.sort_values(["asset", "horizon", "_o"]).drop(columns="_o").reset_index(drop=True)


def _add_multiple_testing_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Adjust the Clark-West p-values across the whole family of ML settings.

    Each (model, asset, horizon) is one hypothesis, and the study runs many of
    them. An uncorrected 5% threshold applied that many times manufactures
    findings; the two adjusted columns say how many survive when the search is
    priced in. Benchmarks are excluded: they are the null, not candidates.
    """
    table = table.copy()
    is_ml = table["model"].isin(ML_NAMES)
    family = table.loc[is_ml, "cw_p_vs_rw"].to_numpy(dtype=float)
    table["cw_p_holm"] = np.nan
    table["cw_p_bh"] = np.nan
    table.loc[is_ml, "cw_p_holm"] = holm_adjusted(family)
    table.loc[is_ml, "cw_p_bh"] = benjamini_hochberg_adjusted(family)
    return table


def _fmt(value: float, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return format(value, spec)


def _stat_p(stat: float, p_value: float) -> str:
    """Render a test as ``statistic (p)``, never a p-value on its own.

    A bare p-value cannot distinguish a model that beats the benchmark from one
    that is significantly worse than it, and several models here are the latter.
    """
    if stat is None or np.isnan(stat):
        return "—"
    return f"{stat:+.2f} ({_fmt(p_value, '.3f')})"


def _range(lo: float, hi: float, spec: str = ".2f") -> str:
    if np.isnan(lo) and np.isnan(hi):
        return "—"
    return f"[{_fmt(lo, spec)}, {_fmt(hi, spec)}]"


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _headline(table: pd.DataFrame) -> str:
    ml = table[table["model"].isin(ML_NAMES)]
    # "Beat" requires the right *direction*, not merely a significant difference:
    # DM is negative when the model has lower loss, CW positive when it predicts.
    beats_dm = ml[(ml["dm_stat"] < 0) & (ml["dm_p_vs_rw"] < 0.05)]
    worse_dm = ml[(ml["dm_stat"] > 0) & (ml["dm_p_vs_rw"] < 0.05)]
    beats_cw = ml[ml["cw_p_vs_rw"] < 0.05]
    timing = ml[(ml["pt_stat"] > 0) & (ml["pt_p"] < 0.05)]
    ci_pos = ml[ml["sharpe_lo"] > 0]
    n_trials = int(table["n_trials"].max())
    expected_fp = 0.05 * len(ml)

    survive_holm = ml[ml["cw_p_holm"] < 0.05]
    survive_bh = ml[ml["cw_p_bh"] < 0.05]
    beat_bnh = _beats_buy_and_hold(table)

    lines = [
        f"- Of **{len(ml)}** ML (model x asset x horizon) settings, "
        f"**{len(beats_dm)}** beat the random walk on squared error at p<0.05 "
        f"(Diebold-Mariano), and **{len(worse_dm)}** were significantly *worse* than it. "
        f"But DM is the wrong test here: the random walk is *nested* in every one of "
        f"these models, which biases DM toward the benchmark.",
        f"- Under Clark-West, which corrects that bias, **{len(beats_cw)}** of {len(ml)} "
        f"reject the no-predictability null at an uncorrected p<0.05, against "
        f"**{expected_fp:.1f}** expected by chance at that threshold across {len(ml)} tests. "
        f"Adjusting for the size of the search, **{len(survive_holm)}** survive "
        f"Holm-Bonferroni (family-wise 5%) and **{len(survive_bh)}** survive "
        f"Benjamini-Hochberg (5% false-discovery rate).",
        f"- **{len(timing)}** showed significant sign-timing skill at p<0.05 "
        f"(Pesaran-Timmermann), the property a directional strategy actually trades on.",
        f"- After costs, **{len(ci_pos)}** ML strategies had a net Sharpe whose 95% "
        f"bootstrap CI excluded zero, and **{beat_bnh}** out-Sharpe simply holding the "
        f"asset on the same schedule and costs, on the point estimate, which is the "
        f"weakest form of that claim.",
    ]
    if table["dsr"].notna().any():
        best = table.loc[table["dsr"].idxmax()]
        lines.append(
            f"- Best deflated Sharpe (deflating for the {n_trials}-model race run within "
            f"each asset-horizon): **{_fmt(best['dsr'], '.2f')}** "
            f"({best['model']}, {best['asset']}, h={int(best['horizon'])}d, "
            f"{int(best['trades'])} position changes)."
        )
    return "\n".join(lines)


def _label(row: pd.Series) -> str:
    return f"{row['model']} on {row['asset']} h={int(row['horizon'])}d"


def _interpretation(table: pd.DataFrame) -> str:
    """Read the statistical and the economic results against each other.

    Written from the numbers rather than asserted, because the interesting
    question is whether the settings that look predictable are the same ones that
    make money. When they are not, both sets are most likely noise.
    """
    ml = table[table["model"].isin(ML_NAMES)]
    survivors = ml[ml["cw_p_bh"] < 0.05]
    earners = ml[ml["sharpe_lo"] > 0]
    lines: list[str] = []

    if survivors.empty:
        lines.append(
            "No setting survives multiple-testing correction, so there is nothing "
            "to reconcile against the trading results: the study finds no evidence "
            "of predictability."
        )
    else:
        detail = "; ".join(
            f"{_label(r)} (R²_oos {_fmt(r['r2_oos'], '.4f')}, net Sharpe "
            f"{_fmt(r['sharpe_net'], '.2f')}, 95% CI "
            f"{_range(r['sharpe_lo'], r['sharpe_hi'])})"
            for _, r in survivors.iterrows()
        )
        lines.append(f"Surviving Benjamini-Hochberg: {detail}.")
        negative_r2 = survivors[survivors["r2_oos"] < 0]
        if not negative_r2.empty:
            count = len(negative_r2)
            subject = "one of these has" if count == 1 else f"{count} of these have"
            lines.append(
                f"Note that {subject} a *negative* out-of-sample "
                "R² against the drift benchmark. That is not a contradiction: "
                "Clark-West tests whether the population mean squared error is lower, "
                "which a model can achieve while its own estimation noise leaves the "
                "realized forecast worse than a constant. It is evidence of a weak "
                "signal, not of a usable forecast."
            )

    overlap = set(map(tuple, survivors[["asset", "horizon", "model"]].to_numpy())) & set(
        map(tuple, earners[["asset", "horizon", "model"]].to_numpy())
    )
    if not survivors.empty or not earners.empty:
        if overlap:
            lines.append(
                f"{len(overlap)} setting(s) are both statistically significant after "
                "correction and profitable with a Sharpe CI clear of zero, the only "
                "combination that would constitute a finding."
            )
        else:
            lines.append(
                "Crucially, **no setting is in both groups**: the settings that pass "
                "the corrected statistical test are not the ones that make money, and "
                "vice versa. Two unrelated sets of winners drawn from the same search "
                "is the signature of noise, not of an edge."
            )
    return "\n".join(f"- {line}" for line in lines)


def _beats_buy_and_hold(table: pd.DataFrame) -> int:
    """How many ML settings out-Sharpe the always-long reference on their own group."""
    reference = {
        (r["asset"], r["horizon"]): r["sharpe_net"]
        for _, r in table[table["model"] == BUY_AND_HOLD].iterrows()
    }
    ml = table[table["model"].isin(ML_NAMES)]
    return int(
        sum(
            1
            for _, r in ml.iterrows()
            if r["sharpe_net"] > reference.get((r["asset"], r["horizon"]), np.inf)
        )
    )


def _evaluated_window(table: pd.DataFrame) -> str:
    """The actual out-of-sample date span, which is shorter than the data span."""
    if "oos_start" not in table.columns or table["oos_start"].dropna().empty:
        return "—"
    return f"{table['oos_start'].dropna().min()} → {table['oos_end'].dropna().max()}"


def write_markdown(
    config: StudyConfig, table: pd.DataFrame, figures: list[str], path: Path
) -> None:
    cfg = config
    wf = cfg.wf
    parts: list[str] = []
    parts.append("# Out-of-sample results\n")
    parts.append(
        f"_Generated by `make backtest` on {date.today().isoformat()}. Do not edit by hand._\n"
    )
    parts.append(
        f"**Study.** Assets {', '.join(cfg.assets)}; horizons "
        f"{', '.join(f'{h}d' for h in cfg.horizons)}; data from {cfg.start} to "
        f"{cfg.end or 'today'}. Walk-forward: {wf.mode}, train {wf.train_size}d / "
        f"test {wf.test_size}d, embargo {wf.embargo}d, so the evaluated out-of-sample "
        f"window is **{_evaluated_window(table)}**. Costs: "
        f"{cfg.costs.cost_per_side * 1e4:.0f} bps per side.\n"
    )
    parts.append("## Headline: does anything beat the random walk?\n")
    parts.append(_headline(table) + "\n")
    parts.append("### Reading the statistics and the PnL together\n")
    parts.append(_interpretation(table) + "\n")

    parts.append("## Forecast accuracy & market timing\n")
    parts.append(
        "Every test is shown as **statistic (p-value)**, because the sign is the "
        "half that matters: a model that is significantly *worse* than the benchmark "
        "produces the same small p as one that is better.\n"
        "\n"
        f"- `R²_oos` — Campbell-Thompson out-of-sample R² against the `{R2_BENCHMARK}` "
        "forecast. Negative means the model loses to an ex-ante drift estimate.\n"
        "- `DM` — Diebold-Mariano, two-sided. **Negative favours the model** "
        "(lower squared error than the random walk).\n"
        "- `CW` — Clark-West, one-sided, valid for the nested comparison DM is not. "
        "**Positive favours the model.**\n"
        "- `CW p adj` — that same p-value after Holm-Bonferroni and Benjamini-Hochberg "
        "correction over all ML settings in the study, shown as `holm / BH`. This is "
        "the column to read: the raw one has been mined across every model, asset, and "
        "horizon here.\n"
        "- `PT` — Pesaran-Timmermann, two-sided. **Positive means sign-timing skill**; "
        "negative means the forecast is reliably wrong-way.\n"
    )
    acc_header = [
        "model",
        "RMSE",
        "R²_oos",
        "DirAcc",
        "RankIC",
        "DM (p)",
        "CW (p)",
        "CW p adj",
        "PT (p)",
    ]
    for (asset, horizon), grp in table[table["model"] != BUY_AND_HOLD].groupby(
        ["asset", "horizon"]
    ):
        parts.append(f"\n**{asset} · h={horizon}d**\n")
        rows = [
            [
                r["model"],
                _fmt(r["rmse"], ".4f"),
                _fmt(r["r2_oos"], ".4f"),
                _fmt(r["dir_acc"], ".3f"),
                _fmt(r["rank_ic"], ".3f"),
                _stat_p(r["dm_stat"], r["dm_p_vs_rw"]),
                _stat_p(r["cw_stat"], r["cw_p_vs_rw"]),
                f"{_fmt(r['cw_p_holm'], '.3f')} / {_fmt(r['cw_p_bh'], '.3f')}",
                _stat_p(r["pt_stat"], r["pt_p"]),
            ]
            for _, r in grp.iterrows()
        ]
        parts.append(_md_table(rows, acc_header))

    parts.append("\n\n## Net-of-cost trading performance\n")
    parts.append(
        "Sign strategy on a non-overlapping h-day schedule, after "
        f"{cfg.costs.cost_per_side * 1e4:.0f} bps/side, against an always-long "
        "reference on the same schedule and costs, the line any long-biased "
        "forecaster has to clear before its Sharpe means anything.\n"
        "\n"
        "- `Sharpe` is annualized at 365/h with a 95% circular-block-bootstrap CI.\n"
        "- `Phases` spans the h possible start offsets of the sampling schedule; a "
        "signal that only works on one offset is an artifact.\n"
        "- `PSR`/`DSR` are the probabilistic and deflated Sharpe ratios. DSR deflates "
        "for the models raced *within* each asset-horizon, which understates the real "
        "search: a reader picking the best cell in this document is choosing among all "
        f"{len(table)} rows, plus the feature set, horizons, and cost level that were "
        "fixed before any of it ran. Treat DSR here as an upper bound.\n"
        "- Drawdowns are deep across the board because a unit-leverage daily-flipping "
        "position on ~70% annualized volatility carries a large variance drag; that is "
        "a property of the position sizing, not of any one model.\n"
    )
    pnl_header = [
        "model",
        "Sharpe (net)",
        "95% CI",
        "Phases",
        "MaxDD",
        "PSR",
        "DSR",
        "changes",
    ]
    for (asset, horizon), grp in table.groupby(["asset", "horizon"]):
        parts.append(f"\n**{asset} · h={horizon}d**\n")
        rows = [
            [
                r["model"],
                _fmt(r["sharpe_net"], ".2f"),
                _range(r["sharpe_lo"], r["sharpe_hi"]),
                "—" if horizon == 1 else _range(r["sharpe_phase_lo"], r["sharpe_phase_hi"]),
                _fmt(r["max_dd"], ".2%"),
                _fmt(r["psr"], ".2f"),
                _fmt(r["dsr"], ".2f"),
                str(int(r["trades"])),
            ]
            for _, r in grp.iterrows()
        ]
        parts.append(_md_table(rows, pnl_header))

    if figures:
        parts.append("\n\n## Figures\n")
        captions = dict(FIGURES)
        for number, fig in enumerate(figures, start=1):
            caption = captions.get(Path(fig).name, Path(fig).stem)
            parts.append(f"**Figure {number}.** {caption}\n")
            parts.append(f"![Figure {number}]({fig})\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")
