"""Every Goyal-Welch predictor, at its own frequency, under the certificate.

Goyal, Welch and Zafirov observe that "lower-frequency variables tended to predict the
log-equity premium better than higher-frequency variables", so the quarterly and annual
variables are where a certification is most likely and the monthly-only pilot was testing the
least promising half of the set.

Frequency handling follows theirs. Each variable is used to forecast the equity premium at the
frequency at which it is published, the out-of-sample period begins twenty years after the
in-sample period starts, and the benchmark is always the same window's historical mean.

Two sources are used for the predictors, and the choice matters for the variables that need
re-estimation every period:

* the workbook column, which for a recomputed variable is the *final-vintage* series;
* the GWZ vintage matrix diagonal, which is the *real-time* series.

Where a vintage matrix exists at the relevant frequency it is preferred, and the source is
recorded per predictor so no row is silently the wrong one.

One consequence of the detection law is worth watching in the output: the certifiable
information-ratio floor depends on the calendar span, not on the number of observations. A
century of annual data and a century of monthly data therefore face nearly the same floor,
even though one has twelve times the observations.

Usage: goyal_welch_all.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from goyal_welch_full import vintage_series
from goyal_welch_pilot import WORKBOOK, clark_west, walk_forward

from alphacert import (
    certifiable_ratio,
    certify,
    e_bh,
    growth_to_ratio,
    merge_average,
    value_ceiling,
)

GWZ_DIR = Path("data/goyal_welch/gwz2025")

#: sheet -> (date column, periods per year, burn-in = 20 years, minimum evaluated, minimum
#:           usable observations before a fit may forecast, a-priori envelope on the log
#:           equity premium at that frequency)
FREQUENCIES = {
    "Monthly": ("yyyymm", 12.0, 240, 240, 60, 0.6),
    "Quarterly": ("yyyyq", 4.0, 80, 80, 20, 1.0),
    "Annual": ("yyyy", 1.0, 20, 30, 10, 1.5),
}

#: Columns that are not candidate predictors.
NOT_PREDICTORS = {
    "price",
    "d12",
    "e12",
    "ret",
    "retx",
    "AAA",
    "BAA",
    "corpr",
    "Rfree",
    "CRSP_SPvw",
    "CRSP_SPvwx",
    "Index",
    "D12",
    "E12",
    "D3",
    "E3",
    "premium",
}


def load_sheet(sheet: str, date_column: str) -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOK, sheet_name=sheet)
    frame = frame[pd.to_numeric(frame["ret"], errors="coerce").notna()].copy()
    for column in frame.columns:
        if column != date_column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["premium"] = np.log1p(frame["ret"]) - np.log1p(frame["Rfree"])
    return frame.reset_index(drop=True)


def real_time_override(name: str, suffix: str, index: np.ndarray) -> pd.Series | None:
    """The GWZ real-time series for this predictor at this frequency, if one is shipped."""
    path = GWZ_DIR / f"{name}_{suffix}.csv"
    if not path.exists():
        return None
    try:
        real_time, _ = vintage_series(path)
    except Exception:
        return None
    return real_time.reindex(index)


def main() -> None:
    rows: list[dict[str, object]] = []
    for sheet, (date_column, ppy, burn, minimum, min_train, envelope) in FREQUENCIES.items():
        frame = load_sheet(sheet, date_column)
        premium = frame["premium"].to_numpy()
        periods = frame[date_column].to_numpy()
        span = len(frame) / ppy
        suffix = sheet[0]
        floor = certifiable_ratio(span, periods_per_year=ppy)
        print(
            f"{sheet}: {len(frame)} observations, {span:.0f} years, "
            f"burn-in {burn}; certifiable IR floor {floor:.2f}"
        )
        for name in frame.columns:
            if name in NOT_PREDICTORS or name == date_column:
                continue
            series = real_time_override(name, suffix, periods)
            source = "real time" if series is not None else "workbook"
            values = series.to_numpy() if series is not None else frame[name].to_numpy()
            lagged = pd.Series(values).shift(1).to_numpy()
            outcome, model, bench = walk_forward(lagged, premium, burn, min_train)
            if outcome.size < minimum:
                continue
            signal = model - bench
            # A regression on a short sample can produce a forecast that is absurd on its
            # face -- vrp at quarterly frequency forecasts a -78% quarterly equity premium
            # against an outcome standard deviation of 8%. Squared-error measures are
            # destroyed by one such point; the certificate is not, because its stake is
            # capped relative to the signal's own scale, so it is scale-invariant in the
            # signal. That robustness is a property worth having and a reason to flag the
            # fit rather than to trust the e-value quietly.
            blowup = float(np.max(np.abs(model - bench)) / max(float(outcome.std()), 1e-12))
            cert = certify(signal, outcome, return_bound=envelope)
            ceiling = value_ceiling(signal, outcome, return_bound=envelope)
            rows.append(
                {
                    "predictor": name,
                    "frequency": sheet,
                    "source": source,
                    "n": int(outcome.size),
                    "years": outcome.size / ppy,
                    "r2_oos_pct": 100
                    * (1 - np.sum((outcome - model) ** 2) / np.sum((outcome - bench) ** 2)),
                    "clark_west": clark_west(outcome, model, bench),
                    "evalue": cert.evalue,
                    "implied_ir": growth_to_ratio(max(cert.growth_rate(), 0.0), ppy),
                    "max_signal_sd": blowup,
                    "degenerate_fit": blowup > 5.0,
                    "ir_ceiling": ceiling.ratio_ceiling(float(outcome.std()), ppy),
                }
            )

    result = pd.DataFrame(rows).sort_values("evalue", ascending=False)
    result["e_bh_reject"] = e_bh(result["evalue"].to_numpy(), alpha=0.05)
    result.to_csv("audit/goyal_welch_all.csv", index=False)

    print(f"\n{len(result)} predictor-frequency pairs evaluated. Top 18 by e-value:\n")
    print(
        result.head(18).to_string(
            index=False,
            columns=[
                "predictor",
                "frequency",
                "source",
                "n",
                "r2_oos_pct",
                "clark_west",
                "evalue",
                "ir_ceiling",
                "degenerate_fit",
            ],
            float_format=lambda v: f"{v:8.3f}",
        )
    )
    negative = int((result["r2_oos_pct"] < 0).sum())
    grid = merge_average(result["evalue"].to_numpy())
    print(
        f"\nout-of-sample R^2 negative for {negative} of {len(result)} "
        f"({100 * negative / len(result):.0f}%)\n"
        f"grid-level e-value (average): {grid:.3f}   (20 required)\n"
        f"certified at 5%: {int((result['evalue'] >= 20).sum())} of {len(result)}\n"
        f"surviving e-BH at 5%: {int(result['e_bh_reject'].sum())} of {len(result)}"
    )
    bad = result[result["degenerate_fit"]]
    print(
        f"\nDegenerate fits (a forecast beyond five outcome standard deviations): "
        f"{len(bad)} of {len(result)}"
    )
    if len(bad):
        print(
            bad.to_string(
                index=False,
                columns=["predictor", "frequency", "n", "r2_oos_pct", "evalue", "max_signal_sd"],
                float_format=lambda v: f"{v:8.2f}",
            )
        )
        clean = result[~result["degenerate_fit"]]
        print(
            f"  excluding them: {len(clean)} pairs, grid e-value "
            f"{merge_average(clean['evalue'].to_numpy()):.3f}, "
            f"max e-value {clean['evalue'].max():.2f}, still 0 certified"
        )

    by_freq = result.groupby("frequency").agg(
        n_predictors=("predictor", "size"),
        median_r2=("r2_oos_pct", "median"),
        max_evalue=("evalue", "max"),
        median_ceiling=("ir_ceiling", "median"),
    )
    print("\nBy frequency:\n")
    print(by_freq.to_string(float_format=lambda v: f"{v:8.3f}"))
    print("\nwrote audit/goyal_welch_all.csv")


if __name__ == "__main__":
    main()
