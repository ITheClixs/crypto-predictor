"""Walk-forward splits must never leak the future into training."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from cryptoforecast.splits import walk_forward_splits


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("horizon", [1, 7])
@pytest.mark.parametrize("embargo", [0, 5])
def test_purge_and_embargo_gap(mode: str, horizon: int, embargo: int) -> None:
    splits = walk_forward_splits(
        1000, train_size=300, test_size=50, horizon=horizon, embargo=embargo, mode=mode
    )
    assert splits, "expected at least one fold"
    for train, test in splits:
        # The core guarantee: last train label lands strictly before the test block.
        assert train.max() + horizon + embargo < test.min()
        assert set(train).isdisjoint(set(test))
        assert train.min() >= 0 and test.max() < 1000


@pytest.mark.unit
def test_expanding_starts_at_zero() -> None:
    splits = walk_forward_splits(800, train_size=200, test_size=40, mode="expanding")
    assert all(train.min() == 0 for train, _ in splits)
    # Training window grows monotonically across folds.
    sizes = [len(train) for train, _ in splits]
    assert sizes == sorted(sizes)


@pytest.mark.unit
def test_rolling_caps_train_size() -> None:
    splits = walk_forward_splits(800, train_size=200, test_size=40, mode="rolling")
    assert all(len(train) <= 200 for train, _ in splits)


@pytest.mark.unit
def test_test_blocks_tile_forward_without_overlap() -> None:
    splits = walk_forward_splits(600, train_size=150, test_size=50, mode="expanding")
    starts = [test.min() for _, test in splits]
    assert starts == sorted(starts)
    for (_, a), (_, b) in pairwise(splits):
        assert a.max() < b.min()


@pytest.mark.unit
def test_min_train_enforced() -> None:
    splits = walk_forward_splits(500, train_size=100, test_size=50, min_train=100, mode="expanding")
    assert all(len(train) >= 100 for train, _ in splits)


@pytest.mark.unit
def test_returns_empty_when_too_short() -> None:
    assert walk_forward_splits(50, train_size=300, test_size=50) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "shuffled"}, "mode must be"),
        ({"horizon": 0}, "horizon must be >= 1"),
        ({"embargo": -1}, "embargo must be non-negative"),
    ],
)
def test_invalid_geometry_is_rejected(kwargs: dict, message: str) -> None:
    """A silently-accepted bad split would invalidate every number downstream."""
    with pytest.raises(ValueError, match=message):
        walk_forward_splits(500, train_size=100, test_size=50, **kwargs)


@pytest.mark.unit
def test_early_folds_are_skipped_rather_than_trained_on_too_little_history() -> None:
    """An expanding window starts below min_train; those folds are dropped, not shrunk."""
    splits = walk_forward_splits(
        500, train_size=100, test_size=50, min_train=150, mode="expanding", horizon=7, embargo=5
    )
    assert splits
    assert all(len(train) >= 150 for train, _ in splits)
    # The first accepted test block starts later than an unconstrained run would.
    unconstrained = walk_forward_splits(
        500, train_size=100, test_size=50, min_train=100, mode="expanding", horizon=7, embargo=5
    )
    assert splits[0].test.min() > unconstrained[0].test.min()


@pytest.mark.unit
def test_rolling_mode_cannot_satisfy_a_min_train_above_its_window() -> None:
    """A rolling window is capped at train_size, so an impossible floor yields no folds."""
    assert (
        walk_forward_splits(400, train_size=100, test_size=50, min_train=150, mode="rolling") == []
    )


@pytest.mark.unit
@pytest.mark.parametrize("horizon", [1, 7])
def test_purging_by_position_still_holds_in_calendar_time_when_bars_are_missing(
    horizon: int,
) -> None:
    """Rows are purged by position, but the leak they prevent is a calendar one.

    A label at row ``i`` is realized ``horizon`` *bars* later, and with bars
    missing that lands at least ``horizon`` calendar days later. Since gaps only
    stretch the timeline, purging ``h`` positions purges at least ``h`` days, so
    the position rule is conservative, never optimistic. This checks that on an
    index with holes punched in it.
    """
    rng = np.random.default_rng(0)
    calendar = pd.date_range("2020-01-01", periods=1200, freq="D")
    keep = np.sort(rng.choice(1200, size=900, replace=False))  # 25% of bars missing
    index = calendar[keep]

    splits = walk_forward_splits(
        len(index), train_size=300, test_size=50, horizon=horizon, embargo=5
    )
    assert splits
    for train, test in splits:
        last_label_date = index[min(train.max() + horizon, len(index) - 1)]
        # The realized date of the last training label precedes the test block.
        assert last_label_date < index[test.min()]
        # And the gap is at least the horizon in calendar days, not just in rows.
        assert (index[test.min()] - index[train.max()]).days >= horizon
