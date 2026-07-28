"""Purged, embargoed walk-forward cross-validation.

Ordinary k-fold shuffles time and leaks the future into the past. For an
``h``-day forward target there is a subtler leak even in a chronological split:
a training label at position ``i`` is realized at ``i + h``, so training rows too
close to the test block have labels that overlap it. We therefore *purge* the
``h`` rows before each test block and add an ``embargo`` gap on top (López de
Prado, *Advances in Financial Machine Learning*, ch. 7).

Every split satisfies ``max(train) + horizon + embargo < min(test)``, which the
test suite checks directly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Split(NamedTuple):
    train: np.ndarray  # integer positions
    test: np.ndarray


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    horizon: int = 1,
    embargo: int = 0,
    mode: str = "expanding",
    min_train: int | None = None,
    step: int | None = None,
) -> list[Split]:
    """Generate chronological train/test splits over ``n_samples`` positions.

    Parameters mirror :class:`cryptoforecast.config.WalkForwardConfig`. Test blocks
    tile the timeline forward (``step`` defaults to ``test_size``, i.e. no overlap).
    Returns an empty list when there is not enough history for one valid fold.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    step = step or test_size
    min_train = min_train or train_size

    # A training row i is safe iff i + horizon < test_start - embargo, so the last
    # usable train position (exclusive) is test_start - horizon - embargo.
    purge = horizon + embargo

    splits: list[Split] = []
    test_start = train_size + purge  # first fold trains on ~train_size rows
    while test_start + 1 <= n_samples:
        train_end = test_start - purge
        train_start = 0 if mode == "expanding" else max(0, train_end - train_size)
        if train_end - train_start < min_train:
            test_start += step
            continue

        test_end = min(test_start + test_size, n_samples)
        train = np.arange(train_start, train_end)
        test = np.arange(test_start, test_end)
        if test.size > 0:
            splits.append(Split(train, test))
        test_start += step

    return splits
