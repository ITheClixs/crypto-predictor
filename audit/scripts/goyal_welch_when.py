"""When did the procedure accumulate its evidence?

Clark-West averages a covariance over the whole sample. The certificate compounds, so its path
records *when* evidence arrived rather than only how much. That is a useful diagnostic and it
is not a changepoint estimator: a flat path can mean the effect vanished, shrank, changed sign,
grew noisier, or simply produced no further favourable draws, and the peak is a random,
post-hoc quantity with no interval attached.

Two corrections to an earlier version of this script are worth recording because both changed
the reported numbers.

*The path must be the raw process.* ``Certificate.wealth`` is the running maximum, which is
non-decreasing by construction, so reading contributions off it reports zero for every period
that failed to set a new high. On the underlying process several of these predictors have
strongly *negative* second halves rather than flat ones.

*The illustrated subset is not the population.* Six predictors chosen to display a pattern
gave a Spearman correlation between Clark-West and the e-value of -0.83, and an earlier draft
generalised that to a claim that the two rank predictors "close to inversely". Across all 117
predictor-frequency pairs the correlation is +0.54. The subset is illustrative only, and the
full-sample correlation is now computed here so the two cannot be confused again.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from alphacert import certify
from cryptoforecast.plots.style import PALETTE, REFERENCE_BLACK, finish

spec = importlib.util.spec_from_file_location("gw", "audit/scripts/goyal_welch_pilot.py")
gw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gw)

OUT = Path("reports/figures/fig11_goyal_welch.png")
SHOW = ("infl", "tbl", "tms", "d/y", "d/p", "e/p")


def main() -> None:
    import matplotlib.pyplot as plt

    frame = gw.load()
    premium = frame["premium"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rows = []
    for i, name in enumerate(SHOW):
        lagged = frame[name].shift(1).to_numpy()
        outcome, model, bench = gw.walk_forward(lagged, premium, gw.BURN)
        dates = frame["yyyymm"].to_numpy()[-outcome.size :]
        years = np.array([int(str(d)[:4]) + (int(str(d)[4:]) - 1) / 12 for d in dates])
        cert = certify(model - bench, outcome, return_bound=gw.ENVELOPE)
        # The raw process, not the running maximum: see the module docstring.
        ax.plot(years, cert.raw_wealth, lw=1.5, color=PALETTE[i % len(PALETTE)], label=name)
        peak = int(np.argmax(cert.raw_wealth))
        # A peak year is only interpretable when the process actually got above its starting
        # point. For a path that never exceeds one, the argmax is the high-water mark of a
        # decline and says nothing about when an edge existed; we record that explicitly
        # rather than printing a year that invites a break-date reading.
        ever_positive = bool(cert.raw_wealth.max() > 1.0)
        rows.append(
            {
                "predictor": name,
                "clark_west": gw.clark_west(outcome, model, bench),
                "evalue": cert.evalue,
                "ever_above_one": ever_positive,
                "peak_year": int(str(dates[peak])[:4]) if ever_positive else None,
                "nats_first_half": float(np.log(cert.raw_wealth)[outcome.size // 2]),
                "nats_second_half": float(
                    np.log(cert.raw_wealth)[-1] - np.log(cert.raw_wealth)[outcome.size // 2]
                ),
            }
        )
    ax.axhline(20.0, color=REFERENCE_BLACK, ls="--", lw=1.3, label="reject at 5%")
    ax.axhline(1.0, color=REFERENCE_BLACK, lw=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.35, 60)
    ax.set_xlabel("Year")
    ax.set_ylabel("Certificate process (raw, log scale)")
    ax.set_title(
        "Timing of sequential evidence accumulation, six illustrative predictors", pad=26
    )
    # Above the axes: every in-panel corner collides either with the reject line or with the
    # valuation-ratio paths, and those paths staying below one is the point of the figure.
    ax.legend(
        ncol=7, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=8.5
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    finish(fig, OUT)

    table = pd.DataFrame(rows)
    table.to_csv("audit/goyal_welch_when.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    # The full-sample correlation, so the illustrative subset can never stand in for it.
    from scipy import stats as _stats

    full = pd.read_csv("audit/goyal_welch_all.csv")
    clean = full[~full["degenerate_fit"]]
    shown = _stats.spearmanr(table["clark_west"], table["evalue"]).statistic
    print(
        f"\nSpearman(Clark-West, e-value):"
        f"\n  the six shown above : {shown:+.3f}   (illustrative subset)"
        f"\n  all {len(full)} pairs      : "
        f"{_stats.spearmanr(full['clark_west'], full['evalue']).statistic:+.3f}"
        f"\n  excluding degenerate: "
        f"{_stats.spearmanr(clean['clark_west'], clean['evalue']).statistic:+.3f}"
        f"\nThe two statistics are positively rank-correlated across the full set. The"
        f"\nnegative correlation in the displayed subset does not generalise."
    )
    print(f"\nwrote {OUT} and audit/goyal_welch_when.csv")


if __name__ == "__main__":
    main()
