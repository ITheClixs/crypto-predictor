# Factor-Zoo Ceilings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an anytime-valid upper bound on the true Sharpe ratio of every published cross-sectional anomaly, using post-publication data only, so the field can retire factors on a number rather than a shrug.

**Architecture:** The equity-premium paper built a certificate that answers "is this signal worth anything?" and, more usefully, "how much could it still be worth?" That ceiling is applied here to a target where it decides something: the 212 published predictors in the Open Source Asset Pricing dataset, evaluated only on data that arrived *after* each was published. The existing `alphacert` package needs one new primitive — a certificate for a bare return stream, where there is no benchmark forecast and hence no drift nuisance — after which the pipeline is a re-parameterisation of what already runs on Goyal-Welch.

**Tech Stack:** Python 3.13, numpy, pandas, scipy, pytest, the existing `alphacert` package, the `openassetpricing` pip package for data, tectonic for the paper.

## Why this and not more work on the current paper

Stated plainly because the plan depends on the reader agreeing:

- The equity-premium paper's mathematics is **not novel**. Proposition 1 is a textbook power calculation. The certificate is a standard composite-null e-process (infimum over a nuisance), a construction that predates it. The empirical result confirms Welch–Goyal (2008).
- Its *quality* is high and it should be posted as-is. Further polishing has no return.
- The reusable asset is the **ceiling**: a time-uniform upper bound that requires no rejection. Applied to six equity-premium predictors it is a footnote. Applied to the factor zoo it answers the most-cited open question in empirical asset pricing.
- Anytime-validity is a **necessity** here rather than a decoration: post-publication samples grow every month, and anyone who re-checks a factor annually as new data lands has invalidated fixed-sample inference. This is the one applied setting where the machinery is required rather than merely available.

Predecessors this extends, and what each left open:

| Paper | What it established | What it did not supply |
|---|---|---|
| Harvey, Liu & Zhu (2016), ~4000 cites | A t-threshold (>3.0) for new factors | Any bound on factors already published |
| McLean & Pontiff (2016), ~2000 cites | 58% post-publication decay | How much is left, with an interval |
| Chen & Zimmermann (2022) | Most claimed findings are likely true | A per-factor upper bound, jointly corrected |

The gap is a per-factor, jointly-corrected, time-uniform **upper** bound. No e-value or anytime-valid treatment of the factor zoo exists (verified by literature search, 2026-08-10).

## Global Constraints

- Python 3.13; the repo venv is `venv/` (has matplotlib/openpyxl), `.venv/` does not. Use `venv/bin/python`.
- Run scripts with `PYTHONPATH=src:audit/scripts`.
- Every number that reaches the paper comes from exactly one script, named in the table caption. No hand-transcribed values.
- Test coverage 80% minimum; every new statistical function gets a calibration test under its own null and a power test under a known alternative.
- Immutable style: functions return new arrays, never mutate inputs.
- Files 200-400 lines typical, 800 max.
- Commit after each task with conventional-commit prefixes.
- **Diagnostics use `Certificate.raw_wealth`; tests use `Certificate.wealth`.** The running maximum is non-decreasing, so asking "when was this rising?" of `wealth` answers "when did it set a new high". This bug shipped once already.
- Any claim of the form "procedure A and procedure B rank X inversely" must be computed on the **full** population, never on a displayed subset. This bug also shipped once.
- No claim that a wealth-path peak dates a structural break. It is a noisy argmax with no interval.

## File Structure

**New library code (`src/alphacert/`):**
- `stream.py` — the new primitive. `certify_mean()` and `mean_ceiling()` for a bare return stream: null is `E[X_t] <= 0`, no nuisance drift, no benchmark. ~180 lines.

**New audit scripts (`audit/scripts/`):**
- `osap_load.py` — download/cache the Open Source Asset Pricing portfolio returns and publication dates; expose one loader. ~150 lines.
- `osap_ceilings.py` — the primary grid: per-factor post-publication certificate and ceiling. ~200 lines.
- `osap_joint.py` — merging, e-BH, and the joint statement across factors. ~150 lines.
- `osap_costs.py` — ceilings against realistic trading costs; the decision table. ~150 lines.
- `osap_tables.py` / `osap_plots.py` — LaTeX and figures. ~200 lines each.

**New tests (`tests/`):**
- `test_alphacert_stream.py` — calibration, power, monotonicity, coverage of the new primitive.
- `test_osap_load.py` — schema and date-alignment guards on the loader.

**Paper (`paper-zoo/`):** created in Task 10, mirroring `paper-equity/` structure.

---

## Task 1: The bare-return-stream certificate

The existing `certify()` requires a signal and eliminates a drift nuisance. A published factor's long-short portfolio return *is* the strategy return: the estimand is its mean, there is no benchmark forecast, and there is no nuisance to eliminate. Using `certify()` here would be wrong — it centres the signal, and a constant signal centres to zero.

**Files:**
- Create: `src/alphacert/stream.py`
- Modify: `src/alphacert/__init__.py`
- Test: `tests/test_alphacert_stream.py`

**Interfaces:**
- Consumes: `Certificate` from `alphacert.certificate`, `_EPS` from the same module.
- Produces:
  - `certify_mean(returns: np.ndarray, *, return_bound: float, cap: float = 0.9) -> Certificate`
  - `mean_ceiling(returns: np.ndarray, *, return_bound: float, alpha: float = 0.05) -> StreamCeiling`
  - `StreamCeiling` dataclass with fields `lower: float`, `upper: float`, `alpha: float` and method `sharpe_ceiling(scale: float, periods_per_year: float = 12.0) -> float`.

- [ ] **Step 1: Write the failing calibration test**

```python
# tests/test_alphacert_stream.py
"""Tests for the bare-return-stream certificate.

A published factor's long-short return is already a strategy return: the estimand is its
mean and there is no benchmark forecast to difference against. That makes the null simpler
than the one `certify` handles -- there is no drift nuisance, because the drift is the
thing being tested -- and a simpler null deserves its own primitive rather than a
mis-parameterised call to the general one.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacert import certify_mean, mean_ceiling

SIGMA = 0.04  # monthly long-short volatility, roughly what OSAP factors show
BOUND = 0.6   # a priori envelope on a monthly long-short return


def _stream(rng: np.random.Generator, n: int, annual_sharpe: float) -> np.ndarray:
    return annual_sharpe / np.sqrt(12.0) * SIGMA + SIGMA * rng.standard_normal(n)


@pytest.mark.unit
def test_a_zero_mean_stream_is_not_certified_more_than_nominally() -> None:
    """The property the whole paper rests on: no edge, no certificate."""
    rejections = 0
    reps = 200
    for rep in range(reps):
        rng = np.random.default_rng([41, rep])
        cert = certify_mean(_stream(rng, 600, 0.0), return_bound=BOUND)
        rejections += cert.rejects(0.05)
    assert rejections / reps <= 0.10, f"leaked at {rejections / reps:.3f}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_alphacert_stream.py -v`
Expected: FAIL with `ImportError: cannot import name 'certify_mean'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/alphacert/stream.py
"""A certificate for a bare return stream.

``certify`` tests whether a signal predicts an outcome, and eliminates the outcome's drift
as a nuisance. For an already-formed strategy -- a published factor's long-short portfolio
return, say -- there is no signal and no benchmark: the drift *is* the estimand. The null
is one-sided,

    H_0:  E[X_t | F_{t-1}]  <=  0,

and the certificate is a single betting martingale on the stream itself. No infimum is
needed, which makes this both simpler and strictly more powerful than routing a constant
signal through the general construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certificate import _EPS, Certificate


def _predictable_stake(payoff: np.ndarray, envelope: float, cap: float) -> np.ndarray:
    """Stakes from the realised payoffs strictly before each step.

    A predictable stake cannot break validity -- only power -- so this rule is chosen for
    stability rather than optimality: it is the running mean over the running second
    moment, truncated to keep every wealth factor positive.
    """
    n = payoff.size
    mean = np.zeros(n)
    second = np.full(n, envelope**2)
    if n > 1:
        cumulative = np.cumsum(payoff)
        squares = np.cumsum(payoff**2)
        counts = np.arange(1, n)
        mean[1:] = cumulative[:-1] / counts
        second[1:] = np.maximum(squares[:-1] / counts, _EPS)
    raw = mean / second
    ceiling = cap / max(envelope, _EPS)
    return np.clip(raw, 0.0, ceiling)


def certify_mean(
    returns: np.ndarray, *, return_bound: float, cap: float = 0.9
) -> Certificate:
    """Anytime-valid certificate for ``E[X_t] > 0`` on a bare return stream."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie in (0, 1)")
    if return_bound <= 0.0:
        raise ValueError("return_bound must be positive")
    if x.size and float(np.abs(x).max()) > return_bound:
        raise ValueError(
            f"return_bound={return_bound} is violated by a return of "
            f"{float(np.abs(x).max()):.4f}; the envelope must be set a priori and hold"
        )
    if x.size == 0:
        return Certificate(np.ones(0), np.ones(0), "stream", np.zeros(0), 0)

    lam = _predictable_stake(x, return_bound, cap)
    increments = np.log1p(lam * x)
    running = np.cumsum(increments)
    raw = np.exp(running)
    wealth = np.exp(np.maximum.accumulate(running))
    return Certificate(wealth, raw, "stream", np.zeros(0), int(np.count_nonzero(lam)))
```

Then add to `src/alphacert/__init__.py`:

```python
from .stream import StreamCeiling, certify_mean, mean_ceiling
```

and extend `__all__` with `"StreamCeiling"`, `"certify_mean"`, `"mean_ceiling"`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_alphacert_stream.py -v`
Expected: PASS. If the rejection rate exceeds 0.10, the stake rule is too aggressive — reduce `cap`, do not weaken the assertion.

- [ ] **Step 5: Add the power test**

```python
@pytest.mark.unit
def test_a_real_edge_is_certified() -> None:
    rng = np.random.default_rng(43)
    cert = certify_mean(_stream(rng, 1200, annual_sharpe=0.8), return_bound=BOUND)
    assert cert.rejects(0.05)
    assert cert.stopping_time(0.05) is not None


@pytest.mark.unit
def test_a_negative_edge_is_never_certified() -> None:
    """One-sided by design: a factor that lost money is not evidence that it works."""
    for rep in range(20):
        rng = np.random.default_rng([47, rep])
        assert not certify_mean(_stream(rng, 1200, -0.8), return_bound=BOUND).rejects(0.05)


@pytest.mark.unit
def test_the_certificate_cannot_see_the_future() -> None:
    rng = np.random.default_rng(53)
    x = _stream(rng, 400, 0.5)
    tampered = x.copy()
    tampered[-1] += 10.0 * SIGMA
    assert np.allclose(
        certify_mean(x, return_bound=BOUND).wealth[:-1],
        certify_mean(tampered, return_bound=BOUND).wealth[:-1],
    )


@pytest.mark.unit
def test_an_envelope_violation_is_refused_rather_than_absorbed() -> None:
    with pytest.raises(ValueError, match="envelope"):
        certify_mean(np.array([0.1, 0.9]), return_bound=0.5)
```

- [ ] **Step 6: Run and confirm all five pass**

Run: `venv/bin/python -m pytest tests/test_alphacert_stream.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/alphacert/stream.py src/alphacert/__init__.py tests/test_alphacert_stream.py
git commit -m "feat: certificate for a bare return stream

A published factor's long-short return is already a strategy return, so the
estimand is its mean and there is no drift nuisance to eliminate. Routing a
constant signal through certify() would centre it to zero; this is the correct
primitive for that case and is strictly more powerful, since it needs no
infimum over a nuisance grid."
```

---

## Task 2: The ceiling for a bare return stream

**Files:**
- Modify: `src/alphacert/stream.py`
- Test: `tests/test_alphacert_stream.py`

**Interfaces:**
- Consumes: `certify_mean` from Task 1.
- Produces: `StreamCeiling(lower, upper, alpha)` with `.sharpe_ceiling(scale, periods_per_year)`, and `mean_ceiling(returns, *, return_bound, alpha=0.05) -> StreamCeiling`.

- [ ] **Step 1: Write the failing coverage test**

```python
@pytest.mark.unit
def test_the_interval_covers_the_truth_at_least_as_often_as_promised() -> None:
    """Time-uniform coverage: the true mean should escape at most alpha of the time."""
    misses = 0
    reps = 200
    true_sharpe = 0.5
    for rep in range(reps):
        rng = np.random.default_rng([59, rep])
        x = _stream(rng, 600, true_sharpe)
        interval = mean_ceiling(x, return_bound=BOUND, alpha=0.05)
        truth = true_sharpe / np.sqrt(12.0) * SIGMA
        misses += not (interval.lower <= truth <= interval.upper)
    assert misses / reps <= 0.10, f"coverage failed at {misses / reps:.3f}"


@pytest.mark.unit
def test_the_ceiling_tightens_as_data_arrives() -> None:
    rng = np.random.default_rng(61)
    x = _stream(rng, 4000, 0.0)
    wide = mean_ceiling(x[:250], return_bound=BOUND).upper
    narrow = mean_ceiling(x, return_bound=BOUND).upper
    assert narrow < wide


@pytest.mark.unit
def test_sharpe_conversion_is_the_square_root_of_time_rule() -> None:
    interval = StreamCeiling(lower=-0.001, upper=0.004, alpha=0.05)
    assert interval.sharpe_ceiling(0.04, 12.0) == pytest.approx(
        0.004 / 0.04 * np.sqrt(12.0)
    )
    with pytest.raises(ValueError):
        interval.sharpe_ceiling(0.0, 12.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_alphacert_stream.py -k ceiling -v`
Expected: FAIL with `ImportError` / `NameError: StreamCeiling`

- [ ] **Step 3: Implement**

```python
# append to src/alphacert/stream.py

@dataclass(frozen=True)
class StreamCeiling:
    """Time-uniform confidence interval for the mean of a return stream."""

    lower: float
    upper: float
    alpha: float

    def sharpe_ceiling(self, scale: float, periods_per_year: float = 12.0) -> float:
        """Upper endpoint as an annualised Sharpe ratio."""
        if scale <= 0:
            raise ValueError("scale must be positive")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        return float(self.upper / scale * np.sqrt(periods_per_year))

    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


def _rules_out(x: np.ndarray, candidate: float, envelope: float, alpha: float) -> bool:
    """Does a two-sided betting martingale against ``E[X]=candidate`` ever reach 1/alpha?"""
    centred = x - candidate
    reach = envelope + abs(candidate)
    stake = 0.5 / max(reach, _EPS)
    up = np.cumsum(np.log1p(stake * centred))
    down = np.cumsum(np.log1p(-stake * centred))
    peak = float(np.max(np.logaddexp(up, down) - np.log(2.0)))
    return peak >= np.log(1.0 / alpha)


def _search_edge(
    x: np.ndarray, inside: float, outside: float, envelope: float, alpha: float
) -> float:
    """Bisect between a retained and a rejected candidate. The retained set is an interval."""
    for _ in range(60):
        middle = 0.5 * (inside + outside)
        if _rules_out(x, middle, envelope, alpha):
            outside = middle
        else:
            inside = middle
    return 0.5 * (inside + outside)


def mean_ceiling(
    returns: np.ndarray, *, return_bound: float, alpha: float = 0.05
) -> StreamCeiling:
    """Time-uniform interval for ``E[X_t]``, from which the Sharpe ceiling follows."""
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if x.size == 0:
        return StreamCeiling(-return_bound, return_bound, alpha)

    centre = float(x.mean())
    if _rules_out(x, centre, return_bound, alpha):
        # The sample mean itself is excluded; the interval is empty in the strict sense,
        # which we report as a degenerate point rather than silently widening.
        return StreamCeiling(centre, centre, alpha)
    span = return_bound
    return StreamCeiling(
        _search_edge(x, centre, centre - span, return_bound, alpha),
        _search_edge(x, centre, centre + span, return_bound, alpha),
        alpha,
    )
```

- [ ] **Step 4: Run and confirm**

Run: `venv/bin/python -m pytest tests/test_alphacert_stream.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/alphacert/stream.py tests/test_alphacert_stream.py
git commit -m "feat: time-uniform interval and Sharpe ceiling for a return stream"
```

---

## Task 3: Load the Open Source Asset Pricing data

212 published predictors with monthly long-short portfolio returns, plus each one's
publication year — which is what makes the post-publication split possible.

**Files:**
- Create: `audit/scripts/osap_load.py`
- Test: `tests/test_osap_load.py`
- Modify: `.gitignore` (cache directory)

**Interfaces:**
- Produces:
  - `load_returns() -> pd.DataFrame` with columns `signal`, `date` (pandas Timestamp, month end), `ret` (decimal monthly long-short return).
  - `load_metadata() -> pd.DataFrame` with columns `signal`, `year` (int, sample-end year of the original publication), `journal`, `sample_end`.
  - `post_publication(returns, metadata) -> pd.DataFrame` adding a boolean `post` column.
  - `CACHE: pathlib.Path` pointing at `data/osap/`.

- [ ] **Step 1: Install the data package and cache the download**

```bash
venv/bin/pip install openassetpricing
mkdir -p data/osap
printf 'data/osap/*.csv\ndata/osap/*.zip\n' >> .gitignore
```

- [ ] **Step 2: Write the failing schema test**

```python
# tests/test_osap_load.py
"""Guards on the factor-zoo loader.

Every downstream number depends on two things being right: that a return series is aligned
to the month it was earned, and that the post-publication flag is computed from the original
paper's sample end rather than its print date. Both are easy to get silently wrong and
neither is visible in a summary statistic, so both are asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit" / "scripts"))

from osap_load import post_publication  # noqa: E402


@pytest.mark.unit
def test_post_publication_flag_uses_the_original_sample_end() -> None:
    """A factor is out of sample only after the data its authors could have seen."""
    returns = pd.DataFrame(
        {
            "signal": ["A"] * 4,
            "date": pd.to_datetime(["1998-12-31", "2000-12-31", "2001-12-31", "2010-12-31"]),
            "ret": [0.01, 0.02, 0.03, 0.04],
        }
    )
    metadata = pd.DataFrame({"signal": ["A"], "sample_end": [2000]})
    flagged = post_publication(returns, metadata)
    assert flagged["post"].tolist() == [False, False, True, True]


@pytest.mark.unit
def test_a_signal_with_no_metadata_is_dropped_not_defaulted() -> None:
    """Silently treating an unknown publication date as 'all post' would invent data."""
    returns = pd.DataFrame(
        {"signal": ["A", "B"], "date": pd.to_datetime(["2010-12-31"] * 2), "ret": [0.0, 0.0]}
    )
    metadata = pd.DataFrame({"signal": ["A"], "sample_end": [2000]})
    assert post_publication(returns, metadata)["signal"].unique().tolist() == ["A"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_osap_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'osap_load'`

- [ ] **Step 4: Implement the loader**

```python
# audit/scripts/osap_load.py
"""Load the Open Source Asset Pricing portfolio returns and publication dates.

Chen and Zimmermann's dataset ships 212 published cross-sectional predictors with monthly
long-short portfolio returns and, critically for this study, the sample period of the paper
that introduced each one. That second field is what makes an honest out-of-sample split
possible: a factor's post-publication record is the part of its history its discoverers
could not have fitted to.

The download is cached under ``data/osap`` and is not committed; the loader is deterministic
given the cache, and the cache's checksum is recorded in the paper.

Usage: imported by the osap_* scripts; run directly to refresh the cache.
"""

from __future__ import annotations

import pathlib

import pandas as pd

CACHE = pathlib.Path("data/osap")
RETURNS_FILE = CACHE / "portfolio_returns.csv"
METADATA_FILE = CACHE / "signal_metadata.csv"


def refresh() -> None:
    """Download once into the cache. Network access lives here and nowhere else."""
    import openassetpricing as oap

    CACHE.mkdir(parents=True, exist_ok=True)
    client = oap.OpenAP()
    client.dl("port_op", RETURNS_FILE.as_posix(), "csv")
    client.dl("signal_doc", METADATA_FILE.as_posix(), "csv")


def load_returns() -> pd.DataFrame:
    """Monthly long-short returns, one row per signal-month, in decimal units."""
    if not RETURNS_FILE.exists():
        raise FileNotFoundError(f"{RETURNS_FILE} missing; run osap_load.py to refresh")
    frame = pd.read_csv(RETURNS_FILE)
    frame = frame.rename(columns=str.lower)
    frame["date"] = pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0)
    # OSAP ships long-short returns in percent; the certificate wants decimals, and an
    # envelope set in decimals against percent data would silently never bind.
    frame["ret"] = frame["ret"] / 100.0
    return frame[["signal", "date", "ret"]].dropna().reset_index(drop=True)


def load_metadata() -> pd.DataFrame:
    """One row per signal, carrying the original paper's sample end year."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"{METADATA_FILE} missing; run osap_load.py to refresh")
    frame = pd.read_csv(METADATA_FILE).rename(columns=str.lower)
    frame = frame.rename(columns={"acronym": "signal", "sampleendyear": "sample_end"})
    keep = [c for c in ("signal", "sample_end", "year", "journal") if c in frame.columns]
    return frame[keep].dropna(subset=["signal", "sample_end"]).reset_index(drop=True)


def post_publication(returns: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Flag each month as inside or after the discovering paper's own sample.

    An inner join, deliberately: a signal whose publication date is unknown cannot be split
    honestly, and defaulting it either way would fabricate an out-of-sample period.
    """
    merged = returns.merge(metadata[["signal", "sample_end"]], on="signal", how="inner")
    merged["post"] = merged["date"].dt.year > merged["sample_end"]
    return merged


if __name__ == "__main__":
    refresh()
    r, m = load_returns(), load_metadata()
    joined = post_publication(r, m)
    print(
        f"{joined['signal'].nunique()} signals, {len(joined)} signal-months, "
        f"{joined['post'].sum()} post-publication "
        f"({100 * joined['post'].mean():.1f}%)"
    )
```

- [ ] **Step 5: Run the tests and refresh the cache**

Run: `venv/bin/python -m pytest tests/test_osap_load.py -v`
Expected: 2 passed.
Then: `PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/osap_load.py`
Expected: a signal count near 212 and a post-publication share near 40%.

**If the column names differ from those assumed above**, print `frame.columns.tolist()` and adjust the two `rename` calls only — do not change the test expectations, which encode the required semantics rather than the vendor's spelling.

- [ ] **Step 6: Record the cache checksum**

```bash
shasum -a 256 data/osap/*.csv | tee audit/osap_checksums.txt
git add audit/osap_checksums.txt .gitignore audit/scripts/osap_load.py tests/test_osap_load.py
git commit -m "feat: load Open Source Asset Pricing returns with publication-date splits"
```

---

## Task 4: The primary grid — a ceiling for every published factor

**Files:**
- Create: `audit/scripts/osap_ceilings.py`
- Output: `audit/osap_ceilings.csv`

**Interfaces:**
- Consumes: `load_returns`, `load_metadata`, `post_publication` (Task 3); `certify_mean`, `mean_ceiling` (Tasks 1-2).
- Produces: `audit/osap_ceilings.csv` with columns `signal`, `n_post`, `years_post`, `mean_ret`, `sd_ret`, `sharpe_in`, `sharpe_post`, `evalue`, `p_value`, `sharpe_ceiling`, `ceiling_excludes_zero`, `decayed`.

- [ ] **Step 1: Write the script**

```python
# audit/scripts/osap_ceilings.py
"""What is the largest Sharpe ratio each published factor could still have?

For every signal in the Open Source Asset Pricing set we take only the months *after* the
sample period of the paper that introduced it, and ask two questions of that record:

1. does the post-publication return stream certify a positive mean at all; and
2. what is the time-uniform upper bound on that mean, expressed as an annualised Sharpe?

The second question is the one the literature has never answered per factor. A factor whose
ceiling sits below plausible trading costs can be retired on a number rather than on taste,
and that statement holds at every future month without recomputation -- which matters because
these samples grow, and anyone rechecking annually under a fixed-sample test has invalidated
their own inference.

The envelope is set a priori at 60% per month, comfortably above any long-short decile
spread in the data, and is asserted rather than fitted.

Usage: osap_ceilings.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from osap_load import load_metadata, load_returns, post_publication

from alphacert import certify_mean, mean_ceiling

ENVELOPE = 0.60
MIN_MONTHS = 60
PERIODS_PER_YEAR = 12.0


def annualised_sharpe(x: np.ndarray) -> float:
    if x.size < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def main() -> None:
    frame = post_publication(load_returns(), load_metadata())
    rows = []
    for signal, block in frame.groupby("signal", sort=True):
        block = block.sort_values("date")
        post = block.loc[block["post"], "ret"].to_numpy()
        pre = block.loc[~block["post"], "ret"].to_numpy()
        if post.size < MIN_MONTHS:
            continue
        cert = certify_mean(post, return_bound=ENVELOPE)
        interval = mean_ceiling(post, return_bound=ENVELOPE, alpha=0.05)
        scale = float(post.std(ddof=1))
        rows.append(
            {
                "signal": signal,
                "n_post": int(post.size),
                "years_post": post.size / PERIODS_PER_YEAR,
                "mean_ret": float(post.mean()),
                "sd_ret": scale,
                "sharpe_in": annualised_sharpe(pre),
                "sharpe_post": annualised_sharpe(post),
                "evalue": cert.evalue,
                "p_value": cert.p_value,
                "sharpe_ceiling": interval.sharpe_ceiling(scale, PERIODS_PER_YEAR),
                "ceiling_excludes_zero": interval.excludes_zero(),
            }
        )

    table = pd.DataFrame(rows)
    table["decayed"] = table["sharpe_post"] < table["sharpe_in"]
    table.to_csv("audit/osap_ceilings.csv", index=False)

    print(f"{len(table)} factors with at least {MIN_MONTHS} post-publication months.\n")
    print(
        table.nlargest(15, "evalue").to_string(
            index=False,
            columns=["signal", "years_post", "sharpe_in", "sharpe_post", "evalue",
                     "sharpe_ceiling"],
            float_format=lambda v: f"{v:8.2f}",
        )
    )
    print(
        f"\nCertified at 5% individually: {int((table['p_value'] < 0.05).sum())} of {len(table)}"
        f"\nMedian in-sample Sharpe:        {table['sharpe_in'].median():.3f}"
        f"\nMedian post-publication Sharpe: {table['sharpe_post'].median():.3f}"
        f"\nDecayed: {int(table['decayed'].sum())} of {len(table)} "
        f"({100 * table['decayed'].mean():.0f}%)"
        f"\nMedian Sharpe ceiling:          {table['sharpe_ceiling'].median():.3f}"
    )
    print("\nwrote audit/osap_ceilings.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/osap_ceilings.py`
Expected: a table of ~180-200 factors. The median post-publication Sharpe should sit well below the median in-sample Sharpe — if it does not, the post-publication flag is inverted and Task 3 is wrong.

- [ ] **Step 3: Sanity-check against McLean & Pontiff**

The published decay figure is roughly 58% of the in-sample return. Compute the ratio directly:

```bash
PYTHONPATH=src:audit/scripts venv/bin/python -c "
import pandas as pd
t = pd.read_csv('audit/osap_ceilings.csv')
print('post/in ratio of medians:', round(t.sharpe_post.median() / t.sharpe_in.median(), 3))
print('median of per-factor ratios:', round((t.sharpe_post / t.sharpe_in).median(), 3))
"
```

Expected: a ratio in the 0.3-0.6 range. A ratio near 1.0 means the split failed; a negative
median means the sign convention is wrong. **Do not proceed past a failed check** — write
down the observed number and reconcile it with the published one before continuing.

- [ ] **Step 4: Commit**

```bash
git add audit/scripts/osap_ceilings.py audit/osap_ceilings.csv
git commit -m "feat: post-publication Sharpe ceiling for every OSAP factor"
```

---

## Task 5: The joint statement across the zoo

Per-factor intervals are marginal. The zoo's whole problem is that hundreds were searched, so
the paper needs a statement that survives that.

**Files:**
- Create: `audit/scripts/osap_joint.py`
- Output: `audit/osap_joint.csv`

**Interfaces:**
- Consumes: `audit/osap_ceilings.csv`; `merge_average`, `e_bh` from `alphacert.merge`.
- Produces: `audit/osap_joint.csv` with columns `procedure`, `alpha`, `rejections`, `note`.

- [ ] **Step 1: Write the script**

```python
# audit/scripts/osap_joint.py
"""Does anything in the zoo survive a joint account?

Each factor's e-value is valid on its own. The literature's difficulty is that hundreds of
factors were searched, so the question worth answering is joint. E-values make this cheap
and assumption-free: they average to an e-value under *arbitrary* dependence, which matters
here because factor returns are heavily correlated and a bootstrap over 200 dependent series
would need a dependence model nobody agrees on.

Three statements are reported:

``individual``
    how many factors clear 1/alpha on their own, which is the number a naive reader would
    quote;
``merged``
    the average e-value over the whole set, which asks whether the zoo *as a body* carries
    evidence;
``e-BH``
    false-discovery control at 5%, which asks which specific factors survive.

Usage: osap_joint.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacert.merge import e_bh, merge_average

ALPHA = 0.05


def main() -> None:
    table = pd.read_csv("audit/osap_ceilings.csv")
    evalues = table["evalue"].to_numpy()

    individual = int((evalues >= 1.0 / ALPHA).sum())
    merged = merge_average(evalues)
    survivors = e_bh(evalues, alpha=ALPHA)
    survivor_names = table.loc[np.asarray(survivors, dtype=bool), "signal"].tolist()

    rows = [
        {
            "procedure": "individual at 5%",
            "alpha": ALPHA,
            "rejections": individual,
            "note": "marginal; ignores that the set was searched",
        },
        {
            "procedure": "merged (average e-value)",
            "alpha": ALPHA,
            "rejections": int(merged >= 1.0 / ALPHA),
            "note": f"merged e-value {merged:.3f} against {1 / ALPHA:.0f} required",
        },
        {
            "procedure": "e-BH at 5%",
            "alpha": ALPHA,
            "rejections": len(survivor_names),
            "note": ", ".join(survivor_names) if survivor_names else "none",
        },
    ]
    pd.DataFrame(rows).to_csv("audit/osap_joint.csv", index=False)

    print(f"{len(table)} factors.\n")
    for row in rows:
        print(f"  {row['procedure']:<26} {row['rejections']:>4}   {row['note']}")
    print(
        "\nE-values are merged by averaging, which is valid under arbitrary dependence "
        "\nand therefore needs no model of the correlation between factor returns."
    )
    print("\nwrote audit/osap_joint.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/osap_joint.py`

- [ ] **Step 3: Commit**

```bash
git add audit/scripts/osap_joint.py audit/osap_joint.csv
git commit -m "feat: joint e-value account across the factor zoo"
```

---

## Task 6: Ceilings against trading costs — the decision table

This is the task that makes the paper actionable rather than descriptive, and it is the
reason a practitioner would cite it.

**Files:**
- Create: `audit/scripts/osap_costs.py`
- Output: `audit/osap_costs.csv`

**Interfaces:**
- Consumes: `audit/osap_ceilings.csv`.
- Produces: `audit/osap_costs.csv` with columns `cost_bps`, `turnover`, `hurdle_sharpe`, `factors_below_ceiling`, `share_below`.

- [ ] **Step 1: Write the script**

```python
# audit/scripts/osap_costs.py
"""How many published factors have a ceiling below the cost of trading them?

A ceiling is only a decision if it is compared with something. The natural comparison for a
long-short equity factor is its implementation cost: a factor whose *upper* bound on true
Sharpe sits below the Sharpe it would need to cover costs cannot be profitable, whatever its
true value, and can be retired without further data.

Costs are parameterised rather than asserted, because the honest range is wide and depends
on the institution. We report a grid of round-trip costs and annual turnovers and convert
each to a Sharpe hurdle,

    hurdle  =  (cost per round trip) * (round trips per year) / (annual volatility),

taking each factor's own realised volatility. No claim is made that any particular cell is
the right one; the table's purpose is to let a reader locate their own cell.

Usage: osap_costs.py
"""

from __future__ import annotations

import pandas as pd

COSTS_BPS = (10, 25, 50, 100)
TURNOVERS = (2.0, 6.0, 12.0)


def main() -> None:
    table = pd.read_csv("audit/osap_ceilings.csv")
    annual_vol = table["sd_ret"] * (12.0**0.5)

    rows = []
    for cost in COSTS_BPS:
        for turnover in TURNOVERS:
            hurdle = (cost / 10_000.0) * turnover / annual_vol
            below = table["sharpe_ceiling"] < hurdle
            rows.append(
                {
                    "cost_bps": cost,
                    "turnover": turnover,
                    "hurdle_sharpe": float(hurdle.median()),
                    "factors_below_ceiling": int(below.sum()),
                    "share_below": float(below.mean()),
                }
            )

    grid = pd.DataFrame(rows)
    grid.to_csv("audit/osap_costs.csv", index=False)
    print(
        "Factors whose 95% anytime-valid Sharpe ceiling lies below their own cost hurdle.\n"
        "A factor in this set cannot be profitable at that cost, whatever its true Sharpe.\n"
    )
    print(grid.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
    print(f"\nOf {len(table)} factors. wrote audit/osap_costs.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and commit**

```bash
PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/osap_costs.py
git add audit/scripts/osap_costs.py audit/osap_costs.csv
git commit -m "feat: ceilings against trading-cost hurdles, the decision table"
```

---

## Task 7: The negative controls

Before any of this reaches a paper it needs the checks that would catch the failure modes the
equity paper hit. Each of these has a right answer known in advance.

**Files:**
- Create: `audit/scripts/osap_controls.py`
- Output: `audit/osap_controls.csv`

**Interfaces:**
- Consumes: `load_returns`, `post_publication`, `certify_mean`, `mean_ceiling`.
- Produces: `audit/osap_controls.csv` with columns `control`, `expected`, `observed`, `passes`.

- [ ] **Step 1: Write the script**

```python
# audit/scripts/osap_controls.py
"""Four checks with answers known before running them.

A pipeline that produces a plausible number is not thereby correct. These controls each have
a right answer fixed in advance, and the paper reports them whether or not they flatter it.

``sign_flip``
    Negating every return must destroy certification. The test is one-sided, so a factor that
    loses money is not evidence that it works. Expected: zero certifications.
``permutation``
    Shuffling each factor's post-publication returns destroys any time structure but keeps
    the mean. Certification should be roughly unchanged, since the null concerns the mean --
    a large change would mean the stake rule is exploiting ordering, not edge.
``synthetic_null``
    Gaussian streams with zero mean, matched to each factor's length and volatility. Expected
    rejection rate at or below 5%.
``synthetic_power``
    The same but with a true annualised Sharpe of 0.5. Reports the detection rate, which is
    the honest statement of what this design can see.

Usage: osap_controls.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from osap_ceilings import ENVELOPE, MIN_MONTHS
from osap_load import load_metadata, load_returns, post_publication

from alphacert import certify_mean

ALPHA = 0.05
SEED = 20260810


def main() -> None:
    frame = post_publication(load_returns(), load_metadata())
    streams = []
    for signal, block in frame.groupby("signal", sort=True):
        post = block.sort_values("date").loc[block["post"], "ret"].to_numpy()
        if post.size >= MIN_MONTHS:
            streams.append((signal, post))

    rng = np.random.default_rng(SEED)
    n = len(streams)

    flipped = sum(certify_mean(-x, return_bound=ENVELOPE).rejects(ALPHA) for _, x in streams)
    permuted = sum(
        certify_mean(rng.permutation(x), return_bound=ENVELOPE).rejects(ALPHA)
        for _, x in streams
    )
    actual = sum(certify_mean(x, return_bound=ENVELOPE).rejects(ALPHA) for _, x in streams)
    null = sum(
        certify_mean(
            x.std(ddof=1) * rng.standard_normal(x.size), return_bound=ENVELOPE
        ).rejects(ALPHA)
        for _, x in streams
    )
    power = sum(
        certify_mean(
            0.5 / np.sqrt(12.0) * x.std(ddof=1) + x.std(ddof=1) * rng.standard_normal(x.size),
            return_bound=ENVELOPE,
        ).rejects(ALPHA)
        for _, x in streams
    )

    rows = [
        {"control": "sign_flip", "expected": "0", "observed": flipped, "passes": flipped == 0},
        {
            "control": "permutation",
            "expected": f"near {actual}",
            "observed": permuted,
            "passes": abs(permuted - actual) <= max(3, 0.25 * max(actual, 1)),
        },
        {
            "control": "synthetic_null",
            "expected": f"<= {int(np.ceil(0.05 * n))}",
            "observed": null,
            "passes": null <= np.ceil(0.10 * n),
        },
        {
            "control": "synthetic_power_sharpe_0.5",
            "expected": "reported, not asserted",
            "observed": power,
            "passes": True,
        },
    ]
    pd.DataFrame(rows).to_csv("audit/osap_controls.csv", index=False)
    print(f"Controls over {n} factors, alpha = {ALPHA}.\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"\nDetection rate at a true annualised Sharpe of 0.5: {power}/{n} "
        f"({100 * power / n:.0f}%). This is the design's power, and it belongs in the paper "
        f"\nwhether or not it is flattering."
    )
    print("\nwrote audit/osap_controls.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it. Do not proceed if `sign_flip` or `synthetic_null` fails.**

Run: `PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/osap_controls.py`

- [ ] **Step 3: Commit**

```bash
git add audit/scripts/osap_controls.py audit/osap_controls.csv
git commit -m "test: negative controls for the factor-zoo pipeline"
```

---

## Task 8: Figures

**Files:**
- Create: `audit/scripts/osap_plots.py`
- Output: `reports/figures/fig20_zoo_ceilings.pdf`, `fig21_zoo_decay.pdf`

**Interfaces:**
- Consumes: `audit/osap_ceilings.csv`, `audit/osap_costs.csv`; `PALETTE`, `REFERENCE_BLACK`, `finish` from `cryptoforecast.plots.style`.

- [ ] **Step 1: Write the plotting script**

Figure 1 (`fig20_zoo_ceilings.pdf`), two panels:
- Left: each factor's post-publication Sharpe (x) against its ceiling (y), with the 45-degree
  line and a horizontal band for the median cost hurdle from Task 6. The story is the vertical
  gap: how much room is left above what was realised.
- Right: the distribution of ceilings as a sorted step plot, with the cost hurdle marked, so a
  reader can read off "N factors have a ceiling below X".

Figure 2 (`fig21_zoo_decay.pdf`): in-sample Sharpe (x) against post-publication Sharpe (y),
one point per factor, 45-degree line, with the McLean-Pontiff decay slope fitted and stated in
the caption. This is the recognisable picture the field already knows, which earns the right to
show the ceiling picture next to it.

```python
# audit/scripts/osap_plots.py -- skeleton; fill both panels as described above
"""Figures for the factor-zoo ceilings.

Every series plotted here is read from a committed CSV rather than recomputed, so a figure
can never disagree with the table beside it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptoforecast.plots.style import PALETTE, REFERENCE_BLACK, finish

CEILINGS = "audit/osap_ceilings.csv"


def ceilings_figure() -> None:
    table = pd.read_csv(CEILINGS)
    costs = pd.read_csv("audit/osap_costs.csv")
    hurdle = float(costs.query("cost_bps == 25 and turnover == 6.0")["hurdle_sharpe"].iloc[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    left.scatter(
        table["sharpe_post"], table["sharpe_ceiling"], s=14, alpha=0.75, color=PALETTE[0]
    )
    lo = float(min(table["sharpe_post"].min(), 0.0))
    hi = float(table["sharpe_ceiling"].max())
    left.plot([lo, hi], [lo, hi], color=REFERENCE_BLACK, lw=0.9, ls="--")
    left.axhline(hurdle, color=PALETTE[3], lw=1.2)
    left.set_xlabel("Realised post-publication Sharpe")
    left.set_ylabel("95% anytime-valid Sharpe ceiling")

    ordered = np.sort(table["sharpe_ceiling"].to_numpy())
    right.step(np.arange(1, ordered.size + 1), ordered, where="post", color=PALETTE[0])
    right.axhline(hurdle, color=PALETTE[3], lw=1.2)
    right.set_xlabel("Factors, ordered by ceiling")
    right.set_ylabel("95% anytime-valid Sharpe ceiling")

    finish(fig, "reports/figures/fig20_zoo_ceilings.pdf")


def decay_figure() -> None:
    table = pd.read_csv(CEILINGS).dropna(subset=["sharpe_in", "sharpe_post"])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.scatter(table["sharpe_in"], table["sharpe_post"], s=14, alpha=0.75, color=PALETTE[1])
    span = [float(table["sharpe_in"].min()), float(table["sharpe_in"].max())]
    ax.plot(span, span, color=REFERENCE_BLACK, lw=0.9, ls="--")
    slope = float(np.polyfit(table["sharpe_in"], table["sharpe_post"], 1)[0])
    ax.plot(span, [slope * s for s in span], color=PALETTE[3], lw=1.4)
    ax.set_xlabel("In-sample Sharpe (discovering paper's window)")
    ax.set_ylabel("Post-publication Sharpe")
    ax.set_title(f"Post-publication decay, fitted slope {slope:.2f}")
    finish(fig, "reports/figures/fig21_zoo_decay.pdf")
    print(f"decay slope: {slope:.3f}")


if __name__ == "__main__":
    ceilings_figure()
    decay_figure()
```

- [ ] **Step 2: Run, then open both PNGs and check them against the CSVs before committing.**

A figure that disagrees with its table has shipped in this repo before. Read the images.

- [ ] **Step 3: Commit**

```bash
git add audit/scripts/osap_plots.py reports/figures/fig20_zoo_ceilings.* reports/figures/fig21_zoo_decay.*
git commit -m "feat: factor-zoo ceiling and decay figures"
```

---

## Task 9: LaTeX tables from the CSVs

**Files:**
- Create: `audit/scripts/osap_tables.py`
- Output: `paper-zoo/tables/*.tex`

**Interfaces:**
- Consumes: the four `audit/osap_*.csv` files.
- Produces: `paper-zoo/tables/headline.tex`, `joint.tex`, `costs.tex`, `controls.tex`, `appendix_all.tex`.

- [ ] **Step 1: Emit complete `tabular` environments, not bare rows**

This is not a style preference. `\input`-ing bare rows inside a `tabular` produces
`Misplaced \noalign` and cost hours the last time. Each generated file must open with
`\begin{tabular}` and close with `\end{tabular}`.

```python
def emit(frame: pd.DataFrame, columns: list[str], alignment: str, path: Path) -> None:
    """Write a complete tabular environment. Bare rows break \\input; do not emit them."""
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule",
             " & ".join(columns) + " \\\\", "\\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(format_cell(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 2: Run, then compile a scratch document that `\input`s each file** to confirm
each compiles standalone before the paper depends on it.

- [ ] **Step 3: Commit**

```bash
git add audit/scripts/osap_tables.py paper-zoo/tables
git commit -m "feat: LaTeX tables generated from the audit CSVs"
```

---

## Task 10: The paper

**Files:**
- Create: `paper-zoo/paper.tex`, `paper-zoo/references.bib`
- Copy: `paper-equity/neurips_2023.sty` → `paper-zoo/`

- [ ] **Step 1: Draft against this skeleton**

Working title: **"The Factor Zoo Has a Ceiling: Anytime-Valid Upper Bounds on 200 Published Anomalies."**

1. *Introduction.* The zoo's question has always been "which of these are real?" The
   answerable version is "how large could each still be?" State the headline: N factors have a
   95% anytime-valid Sharpe ceiling below a 25bp/6× cost hurdle and can be retired on a number.
2. *Related work.* Harvey-Liu-Zhu (a threshold for new factors, nothing for old ones);
   McLean-Pontiff (decay, no bound); Chen-Zimmermann (most are real, no per-factor bound);
   the e-value literature (Ramdas, Grünwald, Vovk, Wang-Ramdas, Larsson-Ramdas-Ruf).
3. *Why anytime-valid is required here, not merely available.* Post-publication samples grow
   monthly. A researcher who rechecks annually under a fixed-sample test has no valid error
   control. This is the argument that earns the machinery; make it early and concretely.
4. *The instrument.* Self-contained, as in `paper-equity`: filtration, the betting martingale,
   the predictable stake, the envelope, the interval by bisection, the Sharpe conversion. State
   that the drift-nuisance construction of the companion paper is *not* needed here and why.
5. *Data and design.* OSAP, the publication-date split, the minimum-months rule, the envelope,
   the checksum.
6. *Results.* Decay (reproducing McLean-Pontiff, earning trust); ceilings; the joint account;
   the cost table; the controls.
7. *Limitations.* Long-short portfolio returns are not net of costs, shorting constraints, the
   OSAP replication choices are not ours, the ceiling is marginal per factor unless stated
   jointly, and the power figure from Task 7 in plain sight.
8. *Conclusion.* The field can retire factors on a number.

- [ ] **Step 2: Compile clean and verify no undefined references**

```bash
cd paper-zoo && tectonic -X compile paper.tex 2>&1 | grep -cE "^error"
cd .. && venv/bin/python -c "
import pypdf, re
r = pypdf.PdfReader('paper-zoo/paper.pdf'); t = ''.join(p.extract_text() for p in r.pages)
print('pages', len(r.pages), '| undefined:', len(re.findall(r'\?\?|\[\?\]', t)))"
```

- [ ] **Step 3: Run the claim audit before posting**

Every numeric claim in the tex must be greppable to a CSV. Build
`audit/ZOO_CLAIM_LEDGER.csv` with columns `claim`, `value`, `source_script`, `source_csv`,
`verified`, one row per number in the paper, and check each by hand.

- [ ] **Step 4: Commit**

```bash
git add paper-zoo audit/ZOO_CLAIM_LEDGER.csv
git commit -m "docs: the factor-zoo ceilings paper"
```

---

## Track B (gated, do not start before Task 7 passes): the numeraire e-variable

**What it is not:** a new theorem. Larsson, Ramdas & Ruf (*Annals of Statistics*, 2025)
established existence and optimality of the numeraire e-variable for an arbitrary composite
null, and its duality with the reverse information projection. That work is done and cannot be
reclaimed.

**What is genuinely open:** *computing* it for the specific composite null this project uses —
"the outcome is a martingale difference around some unknown drift" — and measuring what it
buys. The current construction takes an infimum over a drift grid, which is valid but lossy.
The numeraire is the log-optimal e-variable for that same null. The contribution would be:

1. A characterisation of the numeraire for the drift-nuisance forecast-comparison null, as the
   solution of a per-step convex programme.
2. An algorithm that solves it at each step at a cost comparable to the current grid search.
3. A measured power gain against the already-established 1.90× anytime-validity cost baseline
   in `src/alphacert/design.py`.

**Honest expected value:** a solid methods paper (JMLR, EJS, Bernoulli tier), not a landmark.
Cite it as "a computable instantiation", never as a new optimality theory.

**Gate:** do not begin until the factor-zoo pipeline passes Task 7's controls. If the numeraire
turns out to beat the infimum by less than ~15% in required sample size, write it up as a
one-section remark inside the zoo paper and stop. That threshold is the go/no-go and should be
measured before any prose is written.

---

## Self-review

**Spec coverage.** Every element of the stated goal maps to a task: the new primitive (1-2),
data (3), per-factor ceilings (4), the joint statement (5), the decision-relevant comparison
(6), the checks that would have caught this project's two shipped bugs (7), figures and tables
(8-9), the paper (10). Track B is explicitly gated and is not required for the paper to be
complete.

**Placeholders.** Task 9's `emit` shows the signature and the critical `\begin{tabular}`
requirement but not every column's formatting, and Task 10 is prose rather than code. Both are
deliberate — the formatting depends on the observed column names from Task 3, which cannot be
known until the vendor data is in hand. Every other step contains runnable content.

**Type consistency.** `certify_mean` returns `Certificate` (existing type, so `.evalue`,
`.p_value`, `.rejects`, `.raw_wealth` are all available). `mean_ceiling` returns
`StreamCeiling`, used in Task 4 as `.sharpe_ceiling(scale, 12.0)` and `.excludes_zero()`, both
defined in Task 2. `ENVELOPE` and `MIN_MONTHS` are defined in `osap_ceilings.py` (Task 4) and
imported by `osap_controls.py` (Task 7). Column names written in Task 4 are the ones read in
Tasks 5, 6 and 8.

**Known risk.** Task 3 assumes OSAP's column spellings. If they differ, only the `rename` calls
change; the tests encode semantics and must not be relaxed to accommodate the vendor.
