"""Combining certificates: across a research grid, and across overlapping horizons.

E-values compose in ways p-values do not, and both facts this module rests on are
theorems rather than simulations:

*Averaging.* If ``E_1, ..., E_m`` are e-values then so is any convex combination, **with
no assumption whatsoever about their dependence**. A research grid of eighteen settings
that share features, assets, horizons and market regimes therefore needs no joint
bootstrap to be corrected: the average of the eighteen certificates is already a valid
e-value for the global null that none of them predicts anything.

*Selection.* The e-value analogue of Benjamini-Hochberg (Wang and Ramdas, 2022) controls
the false discovery rate under arbitrary dependence, again with no bootstrap. The usual
BH procedure needs valid p-values and a dependence condition; the e-value version needs
neither, which matters here because the p-values a nested test produces are not exactly
uniform under the null.

*Overlap.* An h-step forecast evaluated every day produces overlapping outcomes, so the
per-day payoffs are not a martingale difference sequence and no wealth process can be
built on them directly. Splitting the sample into the ``h`` non-overlapping phases fixes
that -- each phase *is* a martingale -- and averaging the ``h`` phase certificates
recombines them without needing their joint law. The same average is, read as a trading
rule, the equal-weighted portfolio over all ``h`` daily vintages: the statistically
correct combination and the economically natural implementation are the same object.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .certificate import Certificate, certify


def merge_average(
    evalues: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray | None = None
) -> float:
    """Weighted mean of e-values. Valid under arbitrary dependence."""
    e = np.asarray(evalues, dtype=float)
    if e.size == 0:
        raise ValueError("need at least one e-value")
    if np.any(e < 0):
        raise ValueError("e-values must be non-negative")
    if weights is None:
        return float(e.mean())
    w = np.asarray(weights, dtype=float)
    if w.shape != e.shape:
        raise ValueError("weights must align with e-values")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1")
    return float(np.dot(w, e))


def merge_product(evalues: Sequence[float] | np.ndarray) -> float:
    """Product of e-values. Valid only when each is an e-value given the others' past.

    Use for a sequence of independent replications, never for a research grid evaluated
    on one sample -- there the product overstates the evidence, badly.
    """
    e = np.asarray(evalues, dtype=float)
    if e.size == 0:
        raise ValueError("need at least one e-value")
    if np.any(e < 0):
        raise ValueError("e-values must be non-negative")
    return float(np.prod(e))


def e_bh(evalues: Sequence[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """e-Benjamini-Hochberg: boolean rejections controlling FDR under any dependence.

    Sort the e-values downwards and reject the largest ``k`` for the biggest ``k``
    satisfying ``e_(k) >= m / (alpha k)``.
    """
    e = np.asarray(evalues, dtype=float)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("evalues must be a non-empty one-dimensional array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    m = e.size
    order = np.argsort(-e)
    ranks = np.arange(1, m + 1)
    eligible = e[order] >= m / (alpha * ranks)
    hit = np.flatnonzero(eligible)
    rejected = np.zeros(m, dtype=bool)
    if hit.size:
        rejected[order[: hit[-1] + 1]] = True
    return rejected


def phase_indices(n: int, horizon: int) -> tuple[np.ndarray, ...]:
    """Index sets of the ``horizon`` non-overlapping sub-samples of ``range(n)``."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if n < 0:
        raise ValueError("n must be non-negative")
    return tuple(np.arange(phase, n, horizon) for phase in range(horizon))


def certify_overlapping(
    signal: np.ndarray,
    outcome: np.ndarray,
    horizon: int,
    **kwargs: object,
) -> tuple[float, tuple[Certificate, ...]]:
    """Certificate for h-step forecasts sampled every step.

    Returns the merged e-value and the per-phase certificates. The merge is an average,
    so it is valid however the phases depend on one another -- which they do, since they
    share models, features and overlapping return windows.
    """
    signal = np.asarray(signal, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if signal.shape != outcome.shape:
        raise ValueError("signal and outcome must align")
    certs = tuple(
        certify(signal[idx], outcome[idx], **kwargs)  # type: ignore[arg-type]
        for idx in phase_indices(signal.size, horizon)
        if idx.size
    )
    if not certs:
        raise ValueError("no phase contained an observation")
    return merge_average([c.evalue for c in certs]), certs
