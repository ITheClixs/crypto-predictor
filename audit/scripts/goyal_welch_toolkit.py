"""Does the standard toolkit say anything the certificate does not?

The certificate certifies nothing. That is only informative if the conventional tests, run
on the identical forecasts, also find nothing -- otherwise the non-result is a statement
about our instrument rather than about the data. So we run the toolkit a referee would ask
for, on exactly the walk-forward forecasts of ``goyal_welch_all``:

``DM``
    Diebold-Mariano on squared-error loss. Reported for completeness and *not* as a valid
    test: the models here are nested, and under nesting the loss differential is degenerate
    under the null, so the statistic is not asymptotically standard normal
    (Clark-McCracken). Its p-values are shown struck through in the paper for that reason.
``CW``
    Clark-West, the standard nested correction, compared against a one-sided normal.
``ENC-t``
    The Clark-McCracken encompassing statistic: does the benchmark forecast encompass the
    model's? For a one-step-ahead nested comparison this is *algebraically the same test* as
    Clark-West -- the Clark-West adjusted loss differential is exactly twice the encompassing
    product, and the scale cancels in the t-ratio. We compute it separately and assert the
    identity rather than presenting it as independent corroboration, because reporting the
    same statistic twice under two names would overstate the evidence.
``MCS``
    Hansen, Lunde and Nason's model confidence set over the benchmark and all predictors
    jointly, with the range statistic and a stationary bootstrap. This is the only one of
    the four that prices the multiplicity of comparing many models at once, and it answers
    the question the grid-level e-value answers: is any model distinguishable from the
    best?

The point of the table is not that these agree with the certificate. It is that they agree
on the *verdict* while supplying no interval, which is the paper's argument in one table.

Usage: goyal_welch_toolkit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from goyal_welch_all import load_sheet, real_time_override
from goyal_welch_pilot import PREDICTORS as CLASSIC_MONTHLY
from goyal_welch_pilot import walk_forward
from scipy import stats

BURN_IN = 20 * 12
MIN_TRAIN = 120
BOOTSTRAP = 2000
BLOCK = 12
SEED = 20260810


def diebold_mariano(loss_bench: np.ndarray, loss_model: np.ndarray) -> tuple[float, float]:
    """DM on squared-error loss, Newey-West with the automatic lag. Invalid under nesting."""
    d = loss_bench - loss_model
    n = d.size
    centred = d - d.mean()
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(centred @ centred) / n
    variance = gamma0
    for lag in range(1, lags + 1):
        cov = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    if variance <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(variance / n)
    return float(stat), float(1.0 - stats.norm.cdf(stat))


def clark_west(outcome: np.ndarray, bench: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """Clark-West: DM plus the adjustment for the noise in the extra parameters."""
    f = (outcome - bench) ** 2 - ((outcome - model) ** 2 - (bench - model) ** 2)
    n = f.size
    stat = f.mean() / (f.std(ddof=1) / np.sqrt(n))
    return float(stat), float(1.0 - stats.norm.cdf(stat))


def encompassing_t(
    outcome: np.ndarray, bench: np.ndarray, model: np.ndarray
) -> tuple[float, float]:
    """Clark-McCracken ENC-t: benchmark errors against the benchmark-minus-model gap.

    Under the null that the benchmark encompasses the model the mean of the product is zero.
    The limiting distribution is non-standard; we compare against the normal, which rejects
    too easily, so not rejecting is the conservative direction for our claim.
    """
    c = (outcome - bench) * (model - bench)
    n = c.size
    stat = c.mean() / (c.std(ddof=1) / np.sqrt(n))
    return float(stat), float(1.0 - stats.norm.cdf(stat))


def _stationary_bootstrap_indices(
    n: int, reps: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks, mean length ``block``."""
    p = 1.0 / block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=reps)
    restart = rng.random((reps, n)) < p
    fresh = rng.integers(0, n, size=(reps, n))
    for t in range(1, n):
        carried = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart[:, t], fresh[:, t], carried)
    return idx


def model_confidence_set(losses: np.ndarray, names: list[str], alpha: float = 0.10) -> list[str]:
    """Hansen-Lunde-Nason MCS with the range statistic and a stationary bootstrap.

    ``losses`` is (observations, models). Returns the surviving model names. The elimination
    rule drops the model with the largest standardised excess loss whenever the range
    statistic exceeds its bootstrap quantile.
    """
    rng = np.random.default_rng(SEED)
    n, _ = losses.shape
    idx = _stationary_bootstrap_indices(n, BOOTSTRAP, BLOCK, rng)
    alive = list(range(losses.shape[1]))

    while len(alive) > 1:
        sub = losses[:, alive]
        bar = sub.mean(axis=0)
        # Bootstrap the sampling error of each model's mean loss.
        boot = sub[idx].mean(axis=1)  # (reps, models)
        centred = boot - bar
        var = centred.var(axis=0, ddof=1)
        var = np.maximum(var, 1e-300)

        # Range statistic over all pairs, and its bootstrap distribution.
        diff = bar[:, None] - bar[None, :]
        pair_var = np.maximum(var[:, None] + var[None, :], 1e-300)
        observed = np.max(np.abs(diff) / np.sqrt(pair_var))
        boot_diff = centred[:, :, None] - centred[:, None, :]
        boot_stat = np.max(np.abs(boot_diff) / np.sqrt(pair_var)[None, :, :], axis=(1, 2))
        p_value = float(np.mean(boot_stat >= observed))
        if p_value > alpha:
            break

        # Eliminate the worst: largest standardised excess over the mean model loss.
        excess = (bar - bar.mean()) / np.sqrt(var)
        alive.pop(int(np.argmax(excess)))

    return [names[i] for i in alive]


def main() -> None:
    frame = load_sheet("Monthly", "yyyymm")
    premium = frame["premium"].to_numpy()
    periods = frame["yyyymm"].to_numpy()

    rows, loss_columns, names = [], [], []
    benchmark_loss = None

    for name in CLASSIC_MONTHLY:
        if name not in frame.columns:
            continue
        series = real_time_override(name, "M", periods)
        values = series.to_numpy() if series is not None else frame[name].to_numpy()
        lagged = pd.Series(values).shift(1).to_numpy()
        outcome, model, bench = walk_forward(lagged, premium, BURN_IN, MIN_TRAIN)

        loss_model = (outcome - model) ** 2
        loss_bench = (outcome - bench) ** 2
        if benchmark_loss is None:
            benchmark_loss = loss_bench
            loss_columns.append(loss_bench)
            names.append("benchmark")
        dm_stat, dm_p = diebold_mariano(loss_bench, loss_model)
        cw_stat, cw_p = clark_west(outcome, bench, model)
        enc_stat, enc_p = encompassing_t(outcome, bench, model)
        rows.append(
            {
                "predictor": name,
                "n": outcome.size,
                "dm": dm_stat,
                "dm_p": dm_p,
                "cw": cw_stat,
                "cw_p": cw_p,
                "enc_t": enc_stat,
                "enc_p": enc_p,
            }
        )
        loss_columns.append(loss_model)
        names.append(name)

    table = pd.DataFrame(rows)
    # ENC-t and Clark-West are the same statistic here; assert it rather than double-count.
    gap = float(np.max(np.abs(table["enc_t"] - table["cw"])))
    assert gap < 1e-9, f"ENC-t and Clark-West should coincide for a one-step nested pair; {gap}"
    table.to_csv("audit/goyal_welch_toolkit.csv", index=False)

    print(
        "The conventional toolkit on the identical walk-forward forecasts, "
        f"{len(table)} classic monthly predictors.\n"
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print(
        "\nENC-t equals Clark-West to machine precision, as it must for a one-step nested\n"
        "comparison: the Clark-West adjusted differential is twice the encompassing product.\n"
        "They are one test, not two.\n"
    )
    for label, column in (("DM", "dm_p"), ("Clark-West", "cw_p"), ("ENC-t", "enc_p")):
        for alpha in (0.05, 0.05 / len(table)):
            hits = int((table[column] < alpha).sum())
            tag = "uncorrected" if alpha == 0.05 else f"Bonferroni/{len(table)}"
            print(f"  {label:<11} rejects {hits:2d}/{len(table)} at 5% {tag}")

    losses = np.column_stack(loss_columns)
    survivors = model_confidence_set(losses, names, alpha=0.10)
    print(
        f"\nModel confidence set at 90%: {len(survivors)} of {losses.shape[1]} models survive.\n"
        f"  benchmark in set: {'benchmark' in survivors}\n"
        f"  {', '.join(survivors)}"
    )
    print("\nwrote audit/goyal_welch_toolkit.csv")


if __name__ == "__main__":
    main()
