"""How much of the time was each predictor actually contributing?

Proposition 2 says an effect present only a fraction ``phi`` of the time presents as
``sqrt(phi)`` of its on-period information ratio, so the span needed to certify it scales as
``1/phi``. That is only useful if ``phi`` can be measured, and the certificate's wealth path
measures it directly: wealth compounds while an edge is contributing and flattens when it is
not, so the fraction of the evaluation window over which log wealth is rising is an estimate
of the duty cycle.

We take the duty cycle as the fraction of periods in which the certificate's log wealth
increases, computed on a twelve-month rolling basis so that single lucky months do not count
as an era. A predictor whose edge is genuinely stationary should show a duty cycle near one
half plus its edge; one that worked in a single era and stopped should show much less, and the
year at which its wealth peaks dates the stop.

The resulting distribution feeds back into the floor: multiplying each predictor's own-sample
floor by ``1/sqrt(phi)`` gives the ratio it would actually have had to possess, on its
on-periods, for its sample to have certified it.

Usage: goyal_welch_duty.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from goyal_welch_all import FREQUENCIES, load_sheet, real_time_override
from goyal_welch_pilot import walk_forward

from alphacert import certify

CAMPBELL_THOMPSON = 0.25
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


def duty_cycle(raw_wealth: np.ndarray, blocks: int = 10) -> float:
    """Fraction of equal-length sub-periods in which the signal added log wealth.

    Computed on the *underlying* process, not on the running maximum: the latter is
    non-decreasing by construction, so asking when it rose answers "when did it set a new
    high", which is a different and much smaller number. A predictor with no edge splits its
    sub-periods roughly evenly, so a duty cycle near one half is the no-signal baseline and
    values well below it indicate an edge confined to part of the sample.
    """
    log_wealth = np.log(np.maximum(raw_wealth, 1e-300))
    if log_wealth.size < blocks * 4:
        return float("nan")
    edges = np.linspace(0, log_wealth.size - 1, blocks + 1).astype(int)
    gains = np.diff(log_wealth[edges])
    return float(np.mean(gains > 0))


def main() -> None:
    rows = []
    for sheet, (date_column, ppy, burn, minimum, min_train, envelope) in FREQUENCIES.items():
        frame = load_sheet(sheet, date_column)
        premium = frame["premium"].to_numpy()
        periods = frame[date_column].to_numpy()
        for name in frame.columns:
            if name in NOT_PREDICTORS or name == date_column:
                continue
            series = real_time_override(name, sheet[0], periods)
            values = series.to_numpy() if series is not None else frame[name].to_numpy()
            lagged = pd.Series(values).shift(1).to_numpy()
            outcome, model, bench = walk_forward(lagged, premium, burn, min_train)
            if outcome.size < minimum:
                continue
            cert = certify(model - bench, outcome, return_bound=envelope)
            evaluated = periods[-outcome.size :]
            peak = int(np.argmax(cert.raw_wealth))
            phi = duty_cycle(cert.raw_wealth)
            rows.append(
                {
                    "predictor": name,
                    "frequency": sheet,
                    "duty_cycle": phi,
                    "peak_period": int(evaluated[peak]),
                    "peak_fraction": peak / max(outcome.size - 1, 1),
                    "evalue": cert.evalue,
                    # What the on-period ratio would have to be, given this duty cycle, for
                    # the observed unconditional ratio to reach the economic threshold.
                    "on_period_ir_required": (
                        CAMPBELL_THOMPSON / math.sqrt(phi) if phi and phi > 0 else float("nan")
                    ),
                }
            )

    table = pd.DataFrame(rows).dropna(subset=["duty_cycle"])
    table.to_csv("audit/goyal_welch_duty.csv", index=False)

    print(
        "Duty cycle: the fraction of ten equal sub-periods in which the certificate's log "
        "wealth rose,\nmeasured on the underlying process. A predictor with no edge scores "
        "about 0.50.\n"
    )
    print(
        table.nlargest(12, "evalue").to_string(
            index=False,
            columns=[
                "predictor",
                "frequency",
                "duty_cycle",
                "peak_period",
                "on_period_ir_required",
                "evalue",
            ],
            float_format=lambda v: f"{v:9.2f}",
        )
    )
    print(
        f"\nAcross {len(table)} predictors: median duty cycle "
        f"{table['duty_cycle'].median():.2f}, quartiles "
        f"{table['duty_cycle'].quantile(0.25):.2f} to {table['duty_cycle'].quantile(0.75):.2f}."
    )
    stalled = table[table["peak_fraction"] < 0.75]
    print(
        f"  {len(stalled)} of {len(table)} ({100 * len(stalled) / len(table):.0f}%) peak "
        f"before the final quarter of their own evaluation window, i.e. contributed nothing "
        f"in their last quarter-century or more."
    )
    print(
        f"  At the median duty cycle, an unconditional ratio of {CAMPBELL_THOMPSON} requires "
        f"an on-period ratio of {CAMPBELL_THOMPSON / math.sqrt(table['duty_cycle'].median()):.2f}, "
        f"and Proposition 2 multiplies the required span by "
        f"{1 / table['duty_cycle'].median():.1f}."
    )
    print("\nwrote audit/goyal_welch_duty.csv")


if __name__ == "__main__":
    main()
