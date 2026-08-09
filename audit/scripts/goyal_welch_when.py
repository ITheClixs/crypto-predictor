"""When did each predictor's edge exist? The wealth path dates it.

Clark-West averages a covariance over the whole sample, so a predictor that worked until 1957
and never again scores the same as one that has worked steadily throughout. The certificate
cannot: wealth is a product, so an edge that stops contributing stops compounding, and the
path shows exactly when it stopped. That makes the instrument a stability-weighted test of
predictability, which is the property Goyal and Welch's original complaint calls for.
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
        ax.plot(years, cert.wealth, lw=1.5, color=PALETTE[i % len(PALETTE)], label=name)
        peak = int(np.argmax(cert.wealth))
        rows.append(
            {
                "predictor": name,
                "clark_west": gw.clark_west(outcome, model, bench),
                "evalue": cert.evalue,
                "peak_year": int(str(dates[peak])[:4]),
                "nats_first_half": float(np.log(cert.wealth)[outcome.size // 2]),
                "nats_second_half": float(
                    np.log(cert.wealth)[-1] - np.log(cert.wealth)[outcome.size // 2]
                ),
            }
        )
    ax.axhline(20.0, color=REFERENCE_BLACK, ls="--", lw=1.3, label="reject at 5%")
    ax.axhline(1.0, color=REFERENCE_BLACK, lw=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 30)
    ax.set_xlabel("Year")
    ax.set_ylabel("Certificate $\\mathcal{E}_t$ (log scale)")
    ax.set_title("A century of equity-premium predictors, and when their edge existed")
    ax.legend(ncol=4, loc="upper left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    finish(fig, OUT)

    table = pd.DataFrame(rows)
    table.to_csv("audit/goyal_welch_when.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
    print(f"\nwrote {OUT} and audit/goyal_welch_when.csv")


if __name__ == "__main__":
    main()
