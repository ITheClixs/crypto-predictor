"""What validity costs: the certificate against Clark-West, at matched size.

A test that is valid under weaker assumptions and at every stopping time cannot also be more
powerful than one that assumes more and looks once. The question is the exchange rate, and it
is not in the e-value literature for this comparison, so we measure it.

The design is the one in which Clark-West against the recursive mean is *correctly sized*, so
the comparison is at its most favourable to the incumbent: returns are i.i.d.\\ with a
realistic drift, the larger model is the recursive mean plus a signal plus the out-of-sample
noise of ``p`` estimated coefficients, and the features are exogenous, so the benchmark's
estimation error is uncorrelated with the signal and nothing distorts the incumbent's size.

Two comparisons are reported and they answer different questions.

*Nominal.* Each instrument at the threshold a practitioner would actually use: 1.645 for
Clark-West, ``1/alpha = 20`` for the certificate. This is the honest operational comparison,
and it includes the certificate's conservatism, which is real: Ville's inequality is tight
only for processes that jump straight to the threshold, and a smooth wealth path does not.

*Size-matched.* Both instruments at their empirical 5% critical value, which strips out the
conservatism and isolates the loss due to the construction itself.

The certificate is run in both of its stake modes, because the choice is the dominant term.
Clark-West pays nothing to learn a scale -- the t-statistic is self-normalising -- so pitting
it against a certificate that learns its Kelly fraction online compares two different
experiments. The pre-committed mode is the like-for-like one.

Usage: power_head_to_head.py [reps]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from alphacert import certify
from cryptoforecast.evaluate.stats import newey_west_lrv

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SIGMA = 0.03
N_TEST = 2186
WINDOW = 504
N_FEATURES = 12
DRIFT_SHARPE = 0.8
RATIOS = (1.0, 1.5, 2.0, 3.0)
#: Declared before the data are seen, as the pre-committed mode requires.
DESIGN_RATIO = 1.0
ENVELOPE = 0.6


def _expanding_mean(pre: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    running = np.concatenate([[0.0], np.cumsum(outcome)[:-1]])
    return (pre.sum() + running) / (pre.size + np.arange(outcome.size))


def _clark_west(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray) -> float:
    adjusted = 2.0 * (outcome - bench) * (model - bench)
    lrv = newey_west_lrv(adjusted, 0)
    return float(adjusted.mean() / math.sqrt(lrv / adjusted.size)) if lrv > 0 else float("nan")


def _draw(ratio: float, rep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng([555, rep, int(ratio * 100)])
    drift = DRIFT_SHARPE / math.sqrt(365.0) * SIGMA
    signal = ratio / math.sqrt(365.0) * SIGMA * rng.standard_normal(N_TEST)
    outcome = drift + signal + SIGMA * rng.standard_normal(N_TEST)
    pre = drift + SIGMA * rng.standard_normal(WINDOW)
    benchmark = _expanding_mean(pre, outcome)
    noise = SIGMA * math.sqrt(N_FEATURES / WINDOW) * rng.standard_normal(N_TEST)
    return outcome, benchmark, benchmark + signal + noise, signal + noise


def _wealth(signal: np.ndarray, outcome: np.ndarray, design: float | None) -> float:
    kwargs: dict[str, object] = {
        "payoff": "identity",
        "return_bound": ENVELOPE,
        "drift_resolution": 5e-4,
    }
    if design is not None:
        kwargs["design_ratio"] = design
    return float(certify(signal, outcome, **kwargs).wealth.max())  # type: ignore[arg-type]


def main() -> None:
    modes = {"learned": None, "pre-committed": DESIGN_RATIO}
    null = {"cw": [], **{k: [] for k in modes}}
    for rep in range(REPS):
        outcome, benchmark, model, signal = _draw(0.0, rep)
        null["cw"].append(_clark_west(outcome, model, benchmark))
        for name, design in modes.items():
            null[name].append(_wealth(signal, outcome, design))
    critical = {k: float(np.quantile(v, 0.95)) for k, v in null.items()}
    size = {
        "cw": float(np.mean(np.array(null["cw"]) > 1.645)),
        **{k: float(np.mean(np.array(null[k]) >= 20.0)) for k in modes},
    }

    rows = []
    for ratio in RATIOS:
        record: dict[str, object] = {"information_ratio": ratio}
        hits = dict.fromkeys(null, 0)
        matched = dict.fromkeys(null, 0)
        for rep in range(REPS):
            outcome, benchmark, model, signal = _draw(ratio, rep)
            statistic = _clark_west(outcome, model, benchmark)
            hits["cw"] += statistic > 1.645
            matched["cw"] += statistic > critical["cw"]
            for name, design in modes.items():
                wealth = _wealth(signal, outcome, design)
                hits[name] += wealth >= 20.0
                matched[name] += wealth >= critical[name]
        for key in null:
            record[f"{key}_nominal"] = hits[key] / REPS
            record[f"{key}_size_matched"] = matched[key] / REPS
        rows.append(record)

    result = pd.DataFrame(rows)
    result.to_csv("audit/power_head_to_head.csv", index=False)
    print(f"replications {REPS}; nominal thresholds CW 1.645, certificate 20\n")
    print("measured size at the nominal threshold:")
    for key, value in size.items():
        print(f"  {key:<14} {value:.3f}   (empirical 5% critical value {critical[key]:.3f})")
    print("\npower:\n")
    print(result.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nwrote audit/power_head_to_head.csv")


if __name__ == "__main__":
    main()
