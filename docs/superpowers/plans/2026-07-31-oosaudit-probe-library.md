# oosaudit Probe Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a library that detects seven specific leakage and misspecification failures in
out-of-sample financial forecasting pipelines, validated by a control matrix of synthetic pipelines
with known ground truth.

**Architecture:** Probes never see a pipeline. They see a `Trace` — the observable record of one
execution. A `Replayable` produces a `Trace` given a `Mutation`, so probes that must re-run on
perturbed inputs share one interface with probes that only observe. Harnesses declare which mutations
they support and probes declare which they require, so unsupported combinations return
`not_applicable` by construction rather than being silently skipped.

**Tech Stack:** Python 3.12+, numpy, pandas, scipy, scikit-learn, xgboost, pytest, ruff, mypy.

## Global Constraints

- New repository `oosaudit`. Source under `src/oosaudit/`, tests under `tests/`.
- Files stay in the 200–400 line band; 800 is a hard ceiling.
- All dataclasses frozen. No in-place mutation of `Trace`, `Verdict`, or `Mutation` objects.
- Type annotations on every function signature. `mypy` must pass with no errors.
- `ruff` clean. Formatting via `black`.
- Coverage ≥ 80% overall, 100% on `src/oosaudit/probes/` — the probes are the product.
- Every probe is a pure function of a `Replayable`. No probe imports a reference pipeline.
- No probe may emit `fail` for a `measurement` finding class.
- Reference pipelines are deterministic: every one takes `seed: int` and uses
  `np.random.default_rng(seed)`.

**Scope:** Phases 1–4 of the spec. Phase 5 (subprocess harness, manifest schema, CLI) is a separate
plan, written after two real RFS packages have been wrapped.

**Deviation from spec, deliberate:** the spec lists probe 4 as requiring no mutation. A `Trace` does
not expose the fitted model, so the model's zeroed-feature prediction is not recoverable from
observation alone. This plan adds a fifth mutation, `zero_features`, which replaces every feature
column with its training-window mean. Probe 4 then compares the declared benchmark against the trace
produced under that mutation.

---

## File Structure

| file | responsibility |
|---|---|
| `src/oosaudit/trace.py` | `Fold`, `Meta`, `Trace`, validation |
| `src/oosaudit/mutations.py` | `Mutation` protocol and the five mutations |
| `src/oosaudit/verdict.py` | `Verdict`, severity, report rendering |
| `src/oosaudit/probes/base.py` | `Probe` protocol, capability negotiation |
| `src/oosaudit/probes/leakage.py` | probes 1–3 (`defect`) |
| `src/oosaudit/probes/specification.py` | probes 4–6 (`specification`) |
| `src/oosaudit/probes/calibration.py` | probe 7 (`measurement`) |
| `src/oosaudit/registry.py` | `ALL_PROBES` |
| `src/oosaudit/harness/native.py` | `Pipeline` protocol → `Replayable` |
| `src/oosaudit/references/*.py` | eight synthetic reference pipelines |
| `src/oosaudit/audit.py` | runner: determinism check, error handling, report assembly |

---

### Task 1: Repository scaffold and the Trace core

**Files:**
- Create: `pyproject.toml`, `src/oosaudit/__init__.py`, `src/oosaudit/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Fold(origin, train_idx, test_idx)`, `Meta(horizon, declared_test, hac_lags, seed)`,
  `Trace(folds, features, forecasts, benchmark, y_true, y_time, prices, meta)`,
  `Trace.validate() -> None` raising `ValueError`, and `Trace.bar_period -> pd.Timedelta`.

- [ ] **Step 1: Create the package scaffold**

`pyproject.toml`:

```toml
[project]
name = "oosaudit"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26", "pandas>=2.1", "scipy>=1.11", "scikit-learn>=1.4", "xgboost>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5", "mypy>=1.10", "black>=24.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["unit: fast isolated test", "matrix: control-matrix test"]

[tool.mypy]
files = ["src"]
strict = true

[tool.ruff]
line-length = 100

[tool.black]
line-length = 100
```

Create empty `src/oosaudit/__init__.py`.

- [ ] **Step 2: Write the failing test**

`tests/test_trace.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oosaudit.trace import Fold, Meta, Trace


def build_trace(n: int = 100, horizon: int = 1) -> Trace:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)}, index=idx)
    y_true = pd.Series(rng.normal(scale=0.02, size=n), index=idx)
    folds = (Fold(origin=idx[49], train_idx=np.arange(0, 50), test_idx=np.arange(51, n)),)
    test_idx = idx[51:]
    return Trace(
        folds=folds,
        features=features,
        forecasts=pd.Series(rng.normal(scale=0.001, size=len(test_idx)), index=test_idx),
        benchmark=pd.Series(0.0, index=test_idx),
        y_true=y_true.loc[test_idx],
        y_time=pd.Series(test_idx + pd.Timedelta(days=horizon), index=test_idx),
        prices=pd.Series(100.0 + np.arange(n), index=idx),
        meta=Meta(horizon=horizon, declared_test="clark_west", hac_lags=0, seed=7),
    )


@pytest.mark.unit
def test_valid_trace_passes_validation() -> None:
    build_trace().validate()


@pytest.mark.unit
def test_bar_period_is_the_median_index_spacing() -> None:
    assert build_trace().bar_period == pd.Timedelta(days=1)


@pytest.mark.unit
def test_forecast_index_must_match_y_true_index() -> None:
    t = build_trace()
    broken = Trace(**{**t.__dict__, "forecasts": t.forecasts.iloc[:-1]})
    with pytest.raises(ValueError, match="forecasts and y_true must share an index"):
        broken.validate()


@pytest.mark.unit
def test_label_resolution_must_not_precede_its_origin() -> None:
    t = build_trace()
    broken = Trace(**{**t.__dict__, "y_time": pd.Series(t.y_time.index, index=t.y_time.index)})
    with pytest.raises(ValueError, match="resolves at or before"):
        broken.validate()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_trace.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.trace'`

- [ ] **Step 4: Implement `trace.py`**

```python
"""The observable record of one pipeline execution.

Probes are pure functions of a Trace. Nothing in this module knows what a
pipeline is, which is the point: it decouples adapting messy third-party code
from the checks that are actually valuable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One forecast origin with its train and test row positions."""

    origin: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class Meta:
    """What the pipeline says about itself.

    ``declared_test`` is the statistic the pipeline reports, not one we choose:
    probe 5 escalates from a specification finding to a defect when that
    statistic assumes independent observations and the labels overlap.
    """

    horizon: int
    declared_test: str  # "clark_west" | "diebold_mariano" | "pesaran_timmermann" | "none"
    hac_lags: int | None
    seed: int | None


@dataclass(frozen=True)
class Trace:
    folds: tuple[Fold, ...]
    features: pd.DataFrame
    forecasts: pd.Series
    benchmark: pd.Series
    y_true: pd.Series
    y_time: pd.Series
    prices: pd.Series
    meta: Meta

    @property
    def bar_period(self) -> pd.Timedelta:
        """Median spacing of the feature index, used for calendar-time purge checks."""
        deltas = pd.Series(self.features.index).diff().dropna()
        return pd.Timedelta(deltas.median())

    def validate(self) -> None:
        if not self.folds:
            raise ValueError("a trace must contain at least one fold")
        if self.meta.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.meta.horizon}")
        if not self.features.index.is_monotonic_increasing:
            raise ValueError("features index must be sorted")
        if not self.forecasts.index.equals(self.y_true.index):
            raise ValueError("forecasts and y_true must share an index")
        if not self.benchmark.index.equals(self.forecasts.index):
            raise ValueError("benchmark and forecasts must share an index")
        if not self.y_time.index.equals(self.y_true.index):
            raise ValueError("y_time and y_true must share an index")
        early = self.y_time.to_numpy() <= self.y_time.index.to_numpy()
        if early.any():
            raise ValueError(
                f"{int(early.sum())} labels resolve at or before their own origin; "
                "the target is not forward-looking"
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_trace.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/oosaudit tests/test_trace.py
git commit -m "feat: Trace core with validation"
```

---

### Task 2: The five mutations

**Files:**
- Create: `src/oosaudit/mutations.py`
- Test: `tests/test_mutations.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `Mutation` protocol with `name: str` and `apply(raw: pd.DataFrame) -> pd.DataFrame`; the
  concrete mutations `PerturbAfter(cut, seed)`, `DropBars(fraction, seed)`,
  `CorruptRows(positions, seed)`, `ResampleReturns(block, seed, price_col)`, `ZeroFeatures()`; and the
  name constants `PERTURB_AFTER`, `DROP_BARS`, `CORRUPT_ROWS`, `RESAMPLE_RETURNS`, `ZERO_FEATURES`.

- [ ] **Step 1: Write the failing test**

`tests/test_mutations.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oosaudit.mutations import DropBars, PerturbAfter, ResampleReturns


def raw(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close,
                         "Volume": rng.uniform(1e6, 2e6, n)}, index=idx)


@pytest.mark.unit
def test_perturb_after_leaves_the_past_untouched() -> None:
    df = raw()
    cut = df.index[100]
    out = PerturbAfter(cut=cut, seed=3).apply(df)
    pd.testing.assert_frame_equal(df.loc[:cut], out.loc[:cut])
    assert not np.allclose(df.loc[cut:].to_numpy(), out.loc[cut:].to_numpy())


@pytest.mark.unit
def test_drop_bars_removes_the_requested_fraction_and_keeps_order() -> None:
    df = raw()
    out = DropBars(fraction=0.25, seed=5).apply(df)
    assert len(out) == 150
    assert out.index.is_monotonic_increasing
    assert out.index.isin(df.index).all()


@pytest.mark.unit
def test_resample_returns_preserves_length_and_starting_price() -> None:
    df = raw()
    out = ResampleReturns(block=1, seed=9).apply(df)
    assert len(out) == len(df)
    assert out["Close"].iloc[0] == pytest.approx(df["Close"].iloc[0])
    assert (out["Close"] > 0).all()


@pytest.mark.unit
def test_resample_returns_destroys_the_original_path() -> None:
    df = raw()
    out = ResampleReturns(block=1, seed=9).apply(df)
    assert not np.allclose(df["Close"].to_numpy(), out["Close"].to_numpy())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_mutations.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.mutations'`

- [ ] **Step 3: Implement `mutations.py`**

```python
"""Declared, enumerable transformations of a pipeline's raw input data.

Mutations are what let a probe ask a counterfactual question -- "would this
feature have changed if the future had been different?" -- rather than only
observing what a pipeline did once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

PERTURB_AFTER = "perturb_after"
DROP_BARS = "drop_bars"
CORRUPT_ROWS = "corrupt_rows"
RESAMPLE_RETURNS = "resample_returns"
ZERO_FEATURES = "zero_features"


@runtime_checkable
class Mutation(Protocol):
    name: str

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class PerturbAfter:
    """Randomly rescale every field strictly after ``cut``.

    A causal feature at or before ``cut`` cannot move. One that does has read
    the future.
    """

    cut: pd.Timestamp
    seed: int = 17
    name: str = PERTURB_AFTER

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        future = out.index > self.cut
        rng = np.random.default_rng(self.seed)
        for col in out.columns:
            out.loc[future, col] = out.loc[future, col] * rng.uniform(0.5, 1.5, int(future.sum()))
        return out


@dataclass(frozen=True)
class DropBars:
    """Remove a fraction of bars, so row position and calendar time diverge."""

    fraction: float
    seed: int = 23
    name: str = DROP_BARS

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        keep_n = len(raw) - int(len(raw) * self.fraction)
        keep = np.sort(rng.choice(len(raw), size=keep_n, replace=False))
        return raw.iloc[keep]


@dataclass(frozen=True)
class CorruptRows:
    """Replace the price at the given row positions with noise.

    Used to prove a model never saw rows it claims to have purged.
    """

    positions: tuple[int, ...]
    seed: int = 31
    name: str = CORRUPT_ROWS

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        rng = np.random.default_rng(self.seed)
        rows = [p for p in self.positions if 0 <= p < len(out)]
        for col in out.columns:
            out.iloc[rows, out.columns.get_loc(col)] = rng.uniform(0.5, 1.5, len(rows)) * float(
                out[col].median()
            )
        return out


@dataclass(frozen=True)
class ResampleReturns:
    """Rebuild the price path from resampled returns.

    Future returns become independent of every past-information feature by
    construction, while drift, tails, sample length and the marginal return
    distribution survive. ``block=1`` is an iid bootstrap and measures size;
    ``block>1`` preserves dependence and therefore bounds sensitivity to it
    rather than measuring size.
    """

    block: int = 1
    seed: int = 41
    price_col: str = "Close"
    name: str = RESAMPLE_RETURNS

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        px = raw[self.price_col].to_numpy(dtype=float)
        rets = np.diff(np.log(px))
        n = rets.size
        rng = np.random.default_rng(self.seed)
        if self.block <= 1:
            draws = rets[rng.integers(0, n, n)]
        else:
            starts = rng.integers(0, n, math.ceil(n / self.block))
            offsets = np.arange(self.block)
            idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
            draws = rets[idx]
        path = px[0] * np.exp(np.cumsum(np.concatenate([[0.0], draws])))
        out = raw.copy()
        scale = path / px
        for col in out.columns:
            if col == "Volume":
                continue
            out[col] = out[col].to_numpy() * scale
        return out


@dataclass(frozen=True)
class ZeroFeatures:
    """Marker mutation: the harness replaces each feature with its training mean.

    Applied at the feature stage rather than the raw stage, so ``apply`` is the
    identity here and the harness honours the name.
    """

    name: str = ZERO_FEATURES

    def apply(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_mutations.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/oosaudit/mutations.py tests/test_mutations.py
git commit -m "feat: the five input mutations"
```

---

### Task 3: Verdict schema and report rendering

**Files:**
- Create: `src/oosaudit/verdict.py`
- Test: `tests/test_verdict.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Verdict(probe, finding_class, outcome, severity, evidence, detail, rebuttal)`,
  `Verdict.__post_init__` enforcing the measurement rule, and
  `render_report(verdicts: Sequence[Verdict]) -> str`.

- [ ] **Step 1: Write the failing test**

`tests/test_verdict.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.verdict import Verdict, render_report


@pytest.mark.unit
def test_measurement_findings_may_not_fail() -> None:
    with pytest.raises(ValueError, match="measurement findings cannot fail"):
        Verdict(
            probe="resampled_null_size",
            finding_class="measurement",
            outcome="fail",
            severity=None,
            evidence={"size": 0.22},
            detail="",
            rebuttal="",
        )


@pytest.mark.unit
def test_specification_findings_require_a_rebuttal() -> None:
    with pytest.raises(ValueError, match="requires a rebuttal"):
        Verdict(
            probe="benchmark_mismatch",
            finding_class="specification",
            outcome="fail",
            severity="critical",
            evidence={"max_abs_diff": 0.0017},
            detail="benchmark is the zero forecast",
            rebuttal="",
        )


@pytest.mark.unit
def test_report_lists_failures_first_and_shows_evidence() -> None:
    passing = Verdict("lookahead", "defect", "pass", None, {"max_abs_diff": 0.0}, "clean", "")
    failing = Verdict(
        "benchmark_mismatch", "specification", "fail", "critical",
        {"max_abs_diff": 0.0017}, "benchmark is the zero forecast",
        "flips if a joint zero-drift null was intended",
    )
    out = render_report([passing, failing])
    assert out.index("benchmark_mismatch") < out.index("lookahead")
    assert "0.0017" in out
    assert "flips if a joint zero-drift null was intended" in out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_verdict.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.verdict'`

- [ ] **Step 3: Implement `verdict.py`**

```python
"""What a probe returns, and how a run is rendered for a human.

``finding_class`` is the load-bearing field. A ``defect`` is a bug report and
the author's intent is irrelevant. A ``specification`` finding is a choice the
author may defend, so it carries a rebuttal naming what response would flip it
-- written before the result is seen. A ``measurement`` returns a number and
never passes judgement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FindingClass = Literal["defect", "specification", "measurement"]
Outcome = Literal["pass", "fail", "measured", "not_applicable", "inconclusive", "error"]
Severity = Literal["critical", "major", "moderate", "minor"]

_ORDER: dict[str, int] = {
    "fail": 0, "inconclusive": 1, "error": 2, "measured": 3, "not_applicable": 4, "pass": 5,
}


@dataclass(frozen=True)
class Verdict:
    probe: str
    finding_class: FindingClass
    outcome: Outcome
    severity: Severity | None
    evidence: dict[str, float]
    detail: str
    rebuttal: str

    def __post_init__(self) -> None:
        if self.finding_class == "measurement" and self.outcome == "fail":
            raise ValueError(
                "measurement findings cannot fail; they report a number, not a judgement"
            )
        if self.finding_class == "specification" and self.outcome == "fail" and not self.rebuttal:
            raise ValueError(
                f"{self.probe}: a failing specification finding requires a rebuttal naming "
                "what author response would flip it"
            )


def render_report(verdicts: Sequence[Verdict]) -> str:
    ordered = sorted(verdicts, key=lambda v: (_ORDER.get(v.outcome, 9), v.probe))
    lines = ["# oosaudit report", ""]
    for v in ordered:
        head = f"## {v.probe} — {v.outcome.upper()}"
        if v.severity:
            head += f" ({v.severity})"
        lines += [head, "", f"class: {v.finding_class}", ""]
        if v.detail:
            lines += [v.detail, ""]
        if v.evidence:
            for key, value in v.evidence.items():
                lines.append(f"- {key}: {value:g}")
            lines.append("")
        if v.rebuttal:
            lines += [f"Rebuttal: {v.rebuttal}", ""]
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_verdict.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/oosaudit/verdict.py tests/test_verdict.py
git commit -m "feat: verdict schema with defect/specification/measurement classes"
```

---

### Task 4: Pipeline protocol, native harness, and the clean reference pipeline

**Files:**
- Create: `src/oosaudit/harness/__init__.py`, `src/oosaudit/harness/native.py`,
  `src/oosaudit/references/__init__.py`, `src/oosaudit/references/clean.py`
- Test: `tests/test_native_harness.py`

**Interfaces:**
- Consumes: `Trace`, `Fold`, `Meta` (Task 1); mutation names and `ZeroFeatures` (Task 2).
- Produces: `Pipeline` protocol with `load()`, `features(df)`, `fit(X, y)`, `predict(X)`,
  `target(df)`, `benchmark(y_train, X_test)`, and attributes `horizon: int`, `declared_test: str`,
  `hac_lags: int | None`, `seed: int`; `NativeReplayable(pipeline, train_size, test_size, embargo)`
  with `supported_mutations: frozenset[str]` and `run(mutation: Mutation | None = None) -> Trace`;
  `CleanPipeline(seed=7, horizon=1, n=600)`.

- [ ] **Step 1: Write the failing test**

`tests/test_native_harness.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.mutations import PERTURB_AFTER, RESAMPLE_RETURNS, PerturbAfter
from oosaudit.references.clean import CleanPipeline


@pytest.mark.unit
def test_baseline_run_produces_a_valid_trace() -> None:
    trace = NativeReplayable(CleanPipeline()).run()
    trace.validate()
    assert len(trace.folds) >= 2
    assert len(trace.forecasts) > 0


@pytest.mark.unit
def test_run_is_deterministic() -> None:
    r = NativeReplayable(CleanPipeline())
    first, second = r.run(), r.run()
    assert first.forecasts.equals(second.forecasts)


@pytest.mark.unit
def test_declares_the_mutations_it_supports() -> None:
    supported = NativeReplayable(CleanPipeline()).supported_mutations
    assert PERTURB_AFTER in supported
    assert RESAMPLE_RETURNS in supported


@pytest.mark.unit
def test_a_mutation_changes_the_trace() -> None:
    r = NativeReplayable(CleanPipeline())
    base = r.run()
    cut = base.features.index[len(base.features) // 2]
    mutated = r.run(PerturbAfter(cut=cut, seed=3))
    assert not base.forecasts.equals(mutated.forecasts)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_native_harness.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.harness'`

- [ ] **Step 3: Implement the `Pipeline` protocol and native harness**

`src/oosaudit/harness/__init__.py`: empty file.

`src/oosaudit/harness/native.py`:

```python
"""Drive a pipeline that implements the Protocol directly, and record a Trace.

This is the deep tier. It has full visibility of the fit/predict boundary, so
it supports every mutation. The thin subprocess tier, built later, supports a
subset and says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..mutations import (
    CORRUPT_ROWS,
    DROP_BARS,
    PERTURB_AFTER,
    RESAMPLE_RETURNS,
    ZERO_FEATURES,
    Mutation,
)
from ..trace import Fold, Meta, Trace


class NativeReplayable:
    supported_mutations = frozenset(
        {PERTURB_AFTER, DROP_BARS, CORRUPT_ROWS, RESAMPLE_RETURNS, ZERO_FEATURES}
    )

    def __init__(
        self, pipeline: object, train_size: int = 250, test_size: int = 50, embargo: int = 2
    ) -> None:
        self.pipeline = pipeline
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo

    def run(self, mutation: Mutation | None = None) -> Trace:
        p = self.pipeline
        raw = p.load()  # type: ignore[attr-defined]
        if mutation is not None and mutation.name != ZERO_FEATURES:
            raw = mutation.apply(raw)

        features = p.features(raw)  # type: ignore[attr-defined]
        target = p.target(raw)  # type: ignore[attr-defined]
        keep = features.dropna(how="any").index.intersection(target.dropna().index)
        features, target = features.loc[keep], target.loc[keep]

        zero_features = mutation is not None and mutation.name == ZERO_FEATURES
        folds, preds, benches = [], [], []
        h = int(p.horizon)  # type: ignore[attr-defined]
        start = self.train_size
        while start + self.test_size <= len(features):
            train_idx = np.arange(0, start - h - self.embargo)
            test_idx = np.arange(start, start + self.test_size)
            if train_idx.size < 30:
                start += self.test_size
                continue
            X_tr, y_tr = features.iloc[train_idx], target.iloc[train_idx]
            X_te = features.iloc[test_idx]
            if zero_features:
                X_te = pd.DataFrame(
                    np.tile(X_tr.mean().to_numpy(), (len(X_te), 1)),
                    index=X_te.index,
                    columns=X_te.columns,
                )
            p.fit(X_tr, y_tr)  # type: ignore[attr-defined]
            preds.append(pd.Series(p.predict(X_te), index=X_te.index))  # type: ignore[attr-defined]
            benches.append(
                pd.Series(p.benchmark(y_tr, X_te), index=X_te.index)  # type: ignore[attr-defined]
            )
            folds.append(
                Fold(origin=features.index[start - 1], train_idx=train_idx, test_idx=test_idx)
            )
            start += self.test_size

        forecasts = pd.concat(preds)
        benchmark = pd.concat(benches)
        y_true = target.loc[forecasts.index]
        positions = features.index.get_indexer(forecasts.index)
        resolves = np.minimum(positions + h, len(features.index) - 1)
        y_time = pd.Series(features.index[resolves], index=forecasts.index)

        trace = Trace(
            folds=tuple(folds),
            features=features,
            forecasts=forecasts,
            benchmark=benchmark,
            y_true=y_true,
            y_time=y_time,
            prices=raw["Close"].loc[features.index],
            meta=Meta(
                horizon=h,
                declared_test=str(p.declared_test),  # type: ignore[attr-defined]
                hac_lags=p.hac_lags,  # type: ignore[attr-defined]
                seed=int(p.seed),  # type: ignore[attr-defined]
            ),
        )
        trace.validate()
        return trace
```

- [ ] **Step 4: Implement the clean reference pipeline**

`src/oosaudit/references/__init__.py`: empty file.

`src/oosaudit/references/clean.py`:

```python
"""Reference pipeline zero: causal features, purged splits, drift benchmark.

Every probe must pass on this one. A probe that fires here is broken, not
insightful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class CleanPipeline:
    declared_test = "clark_west"
    hac_lags = None

    def __init__(self, seed: int = 7, horizon: int = 1, n: int = 600) -> None:
        self.seed = seed
        self.horizon = horizon
        self.n = n
        self._model: Ridge | None = None

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        idx = pd.date_range("2019-01-01", periods=self.n, freq="D")
        rets = rng.normal(0.0008, 0.02, self.n)
        close = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.uniform(1e6, 2e6, self.n),
            },
            index=idx,
        )

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        return pd.DataFrame(
            {
                "ret_1": np.log(c / c.shift(1)),
                "ret_5": np.log(c / c.shift(5)),
                "vol_10": np.log(c / c.shift(1)).rolling(10).std(),
            },
            index=df.index,
        )

    def target(self, df: pd.DataFrame) -> pd.Series:
        c = df["Close"]
        return np.log(c.shift(-self.horizon) / c)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = Ridge(alpha=1.0).fit(X.to_numpy(float), y.to_numpy(float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None
        return np.asarray(self._model.predict(X.to_numpy(float)), dtype=float)

    def benchmark(self, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """Recursively estimated training mean -- the model's slopes-zero restriction."""
        return np.full(len(X_test), float(y_train.mean()))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_native_harness.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/harness src/oosaudit/references tests/test_native_harness.py
git commit -m "feat: Pipeline protocol, native harness, clean reference pipeline"
```

---

### Task 5: Probe base protocol and capability negotiation

**Files:**
- Create: `src/oosaudit/probes/__init__.py`, `src/oosaudit/probes/base.py`
- Test: `tests/test_probe_base.py`

**Interfaces:**
- Consumes: `Verdict` (Task 3); `Mutation` names (Task 2).
- Produces: `Probe` protocol with `name: str`, `finding_class`, `required_mutations: frozenset[str]`,
  `run(replayable) -> Verdict`; and `negotiate(probe, replayable) -> Verdict | None` returning a
  `not_applicable` verdict when a required mutation is unsupported and `None` when the probe may run.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_base.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.probes.base import negotiate
from oosaudit.verdict import Verdict


class FakeProbe:
    name = "fake"
    finding_class = "defect"
    required_mutations = frozenset({"perturb_after"})

    def run(self, replayable: object) -> Verdict:
        return Verdict("fake", "defect", "pass", None, {}, "", "")


class Supporting:
    supported_mutations = frozenset({"perturb_after"})


class NotSupporting:
    supported_mutations = frozenset({"drop_bars"})


@pytest.mark.unit
def test_negotiate_returns_none_when_the_harness_supports_the_probe() -> None:
    assert negotiate(FakeProbe(), Supporting()) is None


@pytest.mark.unit
def test_negotiate_reports_not_applicable_naming_the_missing_mutation() -> None:
    verdict = negotiate(FakeProbe(), NotSupporting())
    assert verdict is not None
    assert verdict.outcome == "not_applicable"
    assert "perturb_after" in verdict.detail
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_base.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.probes'`

- [ ] **Step 3: Implement `probes/base.py`**

`src/oosaudit/probes/__init__.py`: empty file.

```python
"""The probe contract and the capability negotiation that keeps coverage honest.

A probe declares which mutations it needs; a harness declares which it
supports. When they do not meet, the result is an explicit not_applicable
verdict naming the gap -- never a silent skip. That is the mechanism by which
a thin harness reports running five probes instead of seven.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..verdict import FindingClass, Verdict


@runtime_checkable
class Probe(Protocol):
    name: str
    finding_class: FindingClass
    required_mutations: frozenset[str]

    def run(self, replayable: object) -> Verdict: ...


def negotiate(probe: Probe, replayable: object) -> Verdict | None:
    """Return a not_applicable verdict if the harness cannot serve this probe."""
    supported: frozenset[str] = getattr(replayable, "supported_mutations", frozenset())
    missing = sorted(probe.required_mutations - supported)
    if not missing:
        return None
    return Verdict(
        probe=probe.name,
        finding_class=probe.finding_class,
        outcome="not_applicable",
        severity=None,
        evidence={},
        detail=f"harness does not support required mutation(s): {', '.join(missing)}",
        rebuttal="",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_probe_base.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/oosaudit/probes tests/test_probe_base.py
git commit -m "feat: probe protocol and capability negotiation"
```

---

### Task 6: Probe 1 — look-ahead perturbation

**Files:**
- Create: `src/oosaudit/probes/leakage.py`, `src/oosaudit/references/leaky_feature.py`
- Test: `tests/test_probe_lookahead.py`

**Interfaces:**
- Consumes: `negotiate` (Task 5), `PerturbAfter`/`PERTURB_AFTER` (Task 2), `NativeReplayable` and
  `CleanPipeline` (Task 4).
- Produces: `LookaheadProbe(cut_fraction=0.5, tol=1e-12)`; `LeakyFeaturePipeline(seed=7, horizon=1,
  n=600)` subclassing `CleanPipeline` and overriding `features`.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_lookahead.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.leakage import LookaheadProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.leaky_feature import LeakyFeaturePipeline


@pytest.mark.unit
def test_passes_on_the_clean_pipeline() -> None:
    v = LookaheadProbe().run(NativeReplayable(CleanPipeline()))
    assert v.outcome == "pass"
    assert v.finding_class == "defect"


@pytest.mark.unit
def test_fires_on_a_centred_rolling_window() -> None:
    v = LookaheadProbe().run(NativeReplayable(LeakyFeaturePipeline()))
    assert v.outcome == "fail"
    assert v.severity == "critical"
    assert v.evidence["max_abs_diff"] > 0
    assert v.evidence["n_columns_affected"] >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_lookahead.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.probes.leakage'`

- [ ] **Step 3: Implement the leaky reference pipeline**

`src/oosaudit/references/leaky_feature.py`:

```python
"""Broken reference: one feature uses a centred window and so reads the future."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import CleanPipeline


class LeakyFeaturePipeline(CleanPipeline):
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"]
        r = np.log(c / c.shift(1))
        return pd.DataFrame(
            {
                "ret_1": r,
                "ret_5": np.log(c / c.shift(5)),
                # center=True reads forward. This is the bug the probe must catch.
                "vol_10": r.rolling(10, center=True).std(),
            },
            index=df.index,
        )
```

- [ ] **Step 4: Implement `probes/leakage.py` with probe 1**

```python
"""Probes for objective leakage defects.

Every probe here is finding_class "defect": the code either reads information
it should not, or it does not. Author intent is irrelevant to the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mutations import PERTURB_AFTER, PerturbAfter
from ..verdict import Verdict
from .base import negotiate


@dataclass(frozen=True)
class LookaheadProbe:
    """Perturbing the future must not change any feature value in the past."""

    cut_fraction: float = 0.5
    tol: float = 1e-12
    name: str = "lookahead"
    finding_class: str = "defect"
    required_mutations: frozenset[str] = frozenset({PERTURB_AFTER})

    def run(self, replayable: object) -> Verdict:
        skip = negotiate(self, replayable)
        if skip is not None:
            return skip

        base = replayable.run()  # type: ignore[attr-defined]
        cut = base.features.index[int(len(base.features) * self.cut_fraction)]
        after = replayable.run(PerturbAfter(cut=cut, seed=17))  # type: ignore[attr-defined]

        shared = base.features.index.intersection(after.features.index)
        past = shared[shared <= cut]
        left = base.features.loc[past]
        right = after.features.loc[past]
        per_column = (left - right).abs().max()
        max_diff = float(per_column.max())
        affected = int((per_column > self.tol).sum())

        evidence = {
            "max_abs_diff": max_diff,
            "n_columns_affected": float(affected),
            "n_past_rows": float(len(past)),
        }
        if max_diff <= self.tol:
            return Verdict(
                self.name, "defect", "pass", None, evidence,
                "no feature at or before the cut moved when the future was perturbed", "",
            )
        names = ", ".join(map(str, per_column[per_column > self.tol].index))
        return Verdict(
            self.name, "defect", "fail", "critical", evidence,
            f"features change in the past when the future is perturbed: {names}", "",
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_probe_lookahead.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/probes/leakage.py src/oosaudit/references/leaky_feature.py tests/test_probe_lookahead.py
git commit -m "feat: probe 1, look-ahead perturbation"
```

---

### Task 7: Probe 2 — missing-bar purge

**Files:**
- Modify: `src/oosaudit/probes/leakage.py` (append)
- Create: `src/oosaudit/references/unpurged_split.py`
- Test: `tests/test_probe_purge.py`

**Interfaces:**
- Consumes: `DropBars`/`DROP_BARS` (Task 2), `negotiate` (Task 5), `NativeReplayable` (Task 4).
- Produces: `MissingBarPurgeProbe(drop_fraction=0.25)`;
  `UnpurgedSplitPipeline(seed=7, horizon=5, n=600)`.

The clean harness purges `h + embargo` rows. The broken reference is driven by a `NativeReplayable`
constructed with `embargo=0` and a pipeline whose horizon exceeds the gap, so the calendar gap closes
once bars are dropped.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_purge.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.leakage import MissingBarPurgeProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.unpurged_split import UnpurgedSplitPipeline


@pytest.mark.unit
def test_passes_when_the_gap_holds_in_calendar_time() -> None:
    r = NativeReplayable(CleanPipeline(horizon=5), embargo=5)
    v = MissingBarPurgeProbe().run(r)
    assert v.outcome == "pass"


@pytest.mark.unit
def test_fires_when_dropping_bars_closes_the_calendar_gap() -> None:
    r = NativeReplayable(UnpurgedSplitPipeline(horizon=5), embargo=0)
    v = MissingBarPurgeProbe().run(r)
    assert v.outcome == "fail"
    assert v.evidence["min_gap_bars"] < 5.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_purge.py -v`
Expected: FAIL, `ImportError: cannot import name 'MissingBarPurgeProbe'`

- [ ] **Step 3: Implement the reference pipeline**

`src/oosaudit/references/unpurged_split.py`:

```python
"""Broken reference: a horizon long enough that a position-based purge is not a
calendar purge once bars go missing."""

from __future__ import annotations

from .clean import CleanPipeline


class UnpurgedSplitPipeline(CleanPipeline):
    """Identical to CleanPipeline; the defect lives in how it is split.

    Drive it with ``NativeReplayable(..., embargo=0)`` so the only gap is the
    horizon itself, measured in row positions.
    """
```

- [ ] **Step 4: Append probe 2 to `probes/leakage.py`**

```python
@dataclass(frozen=True)
class MissingBarPurgeProbe:
    """The train/test gap must hold in calendar time, not merely in row positions."""

    drop_fraction: float = 0.25
    name: str = "missing_bar_purge"
    finding_class: str = "defect"
    required_mutations: frozenset[str] = frozenset({DROP_BARS})

    def run(self, replayable: object) -> Verdict:
        skip = negotiate(self, replayable)
        if skip is not None:
            return skip

        trace = replayable.run(  # type: ignore[attr-defined]
            DropBars(fraction=self.drop_fraction, seed=23)
        )
        period = trace.bar_period
        required = trace.meta.horizon
        gaps = []
        for fold in trace.folds:
            train_end = trace.features.index[fold.train_idx[-1]]
            test_start = trace.features.index[fold.test_idx[0]]
            gaps.append((test_start - train_end) / period)
        min_gap = float(np.min(gaps))

        evidence = {
            "min_gap_bars": min_gap,
            "required_bars": float(required),
            "n_folds": float(len(gaps)),
        }
        if min_gap >= required:
            return Verdict(
                self.name, "defect", "pass", None, evidence,
                "the train/test gap spans the horizon in calendar time under missing bars", "",
            )
        return Verdict(
            self.name, "defect", "fail", "major", evidence,
            f"smallest gap is {min_gap:.2f} bar-periods against a horizon of {required}; "
            "purging is by row position and does not survive missing bars", "",
        )
```

Add `DropBars, DROP_BARS` to the existing `..mutations` import at the top of the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_probe_purge.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/probes/leakage.py src/oosaudit/references/unpurged_split.py tests/test_probe_purge.py
git commit -m "feat: probe 2, missing-bar calendar purge"
```

---

### Task 8: Probe 3 — internal-holdout corruption

**Files:**
- Modify: `src/oosaudit/probes/leakage.py` (append)
- Create: `src/oosaudit/references/unpurged_holdout.py`
- Test: `tests/test_probe_holdout.py`

**Interfaces:**
- Consumes: `CorruptRows`/`CORRUPT_ROWS` (Task 2), `negotiate` (Task 5).
- Produces: `InternalHoldoutProbe()`; `UnpurgedHoldoutPipeline(seed=7, horizon=5, n=600,
  purge=0)` exposing `internal_holdout_positions(n_train: int) -> tuple[int, ...]`.

This probe requires the pipeline to disclose its internal validation slice. A pipeline that does not
expose `internal_holdout_positions` gets `not_applicable` — which is the honest answer, and is why the
spec records this probe as deep-tier-only.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_holdout.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.leakage import InternalHoldoutProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.unpurged_holdout import UnpurgedHoldoutPipeline


@pytest.mark.unit
def test_not_applicable_when_the_pipeline_hides_its_internal_split() -> None:
    v = InternalHoldoutProbe().run(NativeReplayable(CleanPipeline()))
    assert v.outcome == "not_applicable"
    assert "internal_holdout_positions" in v.detail


@pytest.mark.unit
def test_passes_when_the_holdout_is_purged() -> None:
    r = NativeReplayable(UnpurgedHoldoutPipeline(horizon=5, purge=5))
    assert InternalHoldoutProbe().run(r).outcome == "pass"


@pytest.mark.unit
def test_fires_when_the_holdout_is_not_purged() -> None:
    r = NativeReplayable(UnpurgedHoldoutPipeline(horizon=5, purge=0))
    v = InternalHoldoutProbe().run(r)
    assert v.outcome == "fail"
    assert v.severity == "major"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_holdout.py -v`
Expected: FAIL, `ImportError: cannot import name 'InternalHoldoutProbe'`

- [ ] **Step 3: Implement the reference pipeline**

`src/oosaudit/references/unpurged_holdout.py`:

```python
"""Reference with a tunable internal early-stopping holdout.

``purge=0`` leaves the last h fitting rows carrying labels that resolve inside
the validation slice, so early stopping is chosen on partly observed outcomes.
``purge=h`` fixes it. The probe must tell the two apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .clean import CleanPipeline


class UnpurgedHoldoutPipeline(CleanPipeline):
    def __init__(
        self, seed: int = 7, horizon: int = 5, n: int = 600, purge: int = 0,
        val_fraction: float = 0.2,
    ) -> None:
        super().__init__(seed=seed, horizon=horizon, n=n)
        self.purge = purge
        self.val_fraction = val_fraction
        self._alpha = 1.0

    def internal_holdout_positions(self, n_train: int) -> tuple[int, ...]:
        """Rows that must not influence the fit: the validation slice and its purge."""
        k = int(n_train * self.val_fraction)
        return tuple(range(n_train - k - self.purge, n_train))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        n = len(X)
        k = int(n * self.val_fraction)
        fit_end = n - k - self.purge
        Xf, yf = X.iloc[:fit_end], y.iloc[:fit_end]
        Xv, yv = X.iloc[n - k :], y.iloc[n - k :]
        best, best_err = 1.0, np.inf
        for alpha in (0.1, 1.0, 10.0):
            model = Ridge(alpha=alpha).fit(Xf.to_numpy(float), yf.to_numpy(float))
            err = float(np.mean((yv.to_numpy(float) - model.predict(Xv.to_numpy(float))) ** 2))
            if err < best_err:
                best, best_err = alpha, err
        self._alpha = best
        self._model = Ridge(alpha=best).fit(Xf.to_numpy(float), yf.to_numpy(float))
```

- [ ] **Step 4: Append probe 3 to `probes/leakage.py`**

```python
@dataclass(frozen=True)
class InternalHoldoutProbe:
    """Rows a model claims to purge from its internal validation must not move the fit.

    Black-box this is undecidable: corrupting the final h training rows changes
    predictions whether or not an internal holdout exists, because those rows
    are legitimate training data either way. The probe therefore requires the
    pipeline to disclose its internal split, and reports not_applicable when it
    does not.
    """

    name: str = "internal_holdout"
    finding_class: str = "defect"
    required_mutations: frozenset[str] = frozenset({CORRUPT_ROWS})

    def run(self, replayable: object) -> Verdict:
        skip = negotiate(self, replayable)
        if skip is not None:
            return skip

        pipeline = getattr(replayable, "pipeline", None)
        disclose = getattr(pipeline, "internal_holdout_positions", None)
        if disclose is None:
            return Verdict(
                self.name, "defect", "not_applicable", None, {},
                "pipeline does not expose internal_holdout_positions; the internal "
                "validation split is not observable, so this probe cannot decide", "",
            )

        base = replayable.run()  # type: ignore[attr-defined]
        train_len = int(base.folds[0].train_idx.size)
        positions = tuple(disclose(train_len))
        corrupted = replayable.run(CorruptRows(positions=positions, seed=31))  # type: ignore[attr-defined]

        shared = base.forecasts.index.intersection(corrupted.forecasts.index)
        diff = float((base.forecasts.loc[shared] - corrupted.forecasts.loc[shared]).abs().max())
        evidence = {"max_abs_forecast_diff": diff, "n_rows_corrupted": float(len(positions))}
        if diff <= 1e-12:
            return Verdict(
                self.name, "defect", "pass", None, evidence,
                "corrupting the purged internal-holdout rows left the fit unchanged", "",
            )
        return Verdict(
            self.name, "defect", "fail", "major", evidence,
            "corrupting rows that should have been purged from the internal validation "
            f"slice changed forecasts by up to {diff:.3g}", "",
        )
```

Add `CorruptRows, CORRUPT_ROWS` to the `..mutations` import.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_probe_holdout.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/probes/leakage.py src/oosaudit/references/unpurged_holdout.py tests/test_probe_holdout.py
git commit -m "feat: probe 3, internal-holdout corruption"
```

---

### Task 9: Probe 4 — benchmark / restriction mismatch

**Files:**
- Create: `src/oosaudit/probes/specification.py`, `src/oosaudit/references/zero_benchmark_drift.py`
- Test: `tests/test_probe_benchmark.py`

**Interfaces:**
- Consumes: `ZeroFeatures`/`ZERO_FEATURES` (Task 2), `negotiate` (Task 5).
- Produces: `BenchmarkMismatchProbe(tol=1e-6)`;
  `ZeroBenchmarkDriftPipeline(seed=7, horizon=1, n=600)` overriding `benchmark` to return zeros.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_benchmark.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.specification import BenchmarkMismatchProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.zero_benchmark_drift import ZeroBenchmarkDriftPipeline


@pytest.mark.unit
def test_passes_when_the_benchmark_is_the_zeroed_feature_restriction() -> None:
    v = BenchmarkMismatchProbe().run(NativeReplayable(CleanPipeline()))
    assert v.outcome == "pass"


@pytest.mark.unit
def test_fires_on_a_zero_benchmark_under_drift_and_reports_the_drift_share() -> None:
    v = BenchmarkMismatchProbe().run(NativeReplayable(ZeroBenchmarkDriftPipeline()))
    assert v.outcome == "fail"
    assert v.finding_class == "specification"
    assert v.severity == "critical"
    assert v.rebuttal
    assert v.evidence["max_abs_diff"] > 0
    assert 0.0 < v.evidence["drift_share"] <= 1.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_benchmark.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.probes.specification'`

- [ ] **Step 3: Implement the reference pipeline**

`src/oosaudit/references/zero_benchmark_drift.py`:

```python
"""Broken reference: a zero-return benchmark on a series with a real drift.

The estimator fits an intercept, so its zeroed-feature restriction is the
training mean, not zero. Testing against zero therefore tests a joint null of
no drift and no conditional predictability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import CleanPipeline


class ZeroBenchmarkDriftPipeline(CleanPipeline):
    def benchmark(self, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X_test), dtype=float)
```

- [ ] **Step 4: Implement `probes/specification.py` with probe 4**

```python
"""Probes for choices the author may defend.

Every failing verdict here carries a rebuttal naming what author response would
flip it. That field is written before the result is seen; a specification
finding for which no rebuttal can be stated is not published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mutations import ZERO_FEATURES, ZeroFeatures
from ..verdict import Verdict
from .base import negotiate


@dataclass(frozen=True)
class BenchmarkMismatchProbe:
    """Is the declared benchmark the model's own zeroed-feature restriction?

    Clark-West reduces to a t-test on 2(y - b)(m - b), so the benchmark defines
    the hypothesis rather than merely the baseline. When the benchmark is the
    zero forecast and the estimator fits an intercept, the statistic's
    expectation carries a drift term, 2 E[y] E[m], which is nonzero however
    uninformative the features are.
    """

    tol: float = 1e-6
    name: str = "benchmark_mismatch"
    finding_class: str = "specification"
    required_mutations: frozenset[str] = frozenset({ZERO_FEATURES})

    def run(self, replayable: object) -> Verdict:
        skip = negotiate(self, replayable)
        if skip is not None:
            return skip

        base = replayable.run()  # type: ignore[attr-defined]
        restricted = replayable.run(ZeroFeatures())  # type: ignore[attr-defined]

        shared = base.benchmark.index.intersection(restricted.forecasts.index)
        declared = base.benchmark.loc[shared].to_numpy(float)
        implied = restricted.forecasts.loc[shared].to_numpy(float)
        max_diff = float(np.max(np.abs(declared - implied)))

        y = base.y_true.loc[shared].to_numpy(float)
        m = base.forecasts.loc[shared].to_numpy(float)
        drift = float(np.mean(y) * np.mean(m))
        cov = float(np.cov(y, m, ddof=1)[0, 1])
        denom = drift + cov
        drift_share = float(drift / denom) if denom != 0 else float("nan")

        evidence = {
            "max_abs_diff": max_diff,
            "mean_abs_diff": float(np.mean(np.abs(declared - implied))),
            "drift_share": drift_share,
        }
        if max_diff <= self.tol:
            return Verdict(
                self.name, "specification", "pass", None, evidence,
                "the declared benchmark equals the model's zeroed-feature restriction", "",
            )
        return Verdict(
            self.name, "specification", "fail", "critical", evidence,
            f"the declared benchmark differs from the model's zeroed-feature restriction by "
            f"up to {max_diff:.3g}; {drift_share:.0%} of the adjusted differential is drift "
            "rather than covariance",
            "flips if the authors intended the joint martingale-difference null of "
            "Clark & West (2006), in which zero drift is part of the hypothesis",
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_probe_benchmark.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/probes/specification.py src/oosaudit/references/zero_benchmark_drift.py tests/test_probe_benchmark.py
git commit -m "feat: probe 4, benchmark/restriction mismatch"
```

---

### Task 10: Probe 5 — label-overlap inflation

**Files:**
- Modify: `src/oosaudit/probes/specification.py` (append)
- Create: `src/oosaudit/references/overlapping_labels.py`, `src/oosaudit/stats.py`
- Test: `tests/test_probe_overlap.py`

**Interfaces:**
- Consumes: nothing new from mutations; reads `Trace` only.
- Produces: `pesaran_timmermann(y_true, y_pred) -> tuple[float, float]` in `stats.py`;
  `LabelOverlapProbe()`; `OverlappingLabelsPipeline(seed=7, horizon=7, n=900)` with
  `declared_test = "pesaran_timmermann"`.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_overlap.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.specification import LabelOverlapProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.overlapping_labels import OverlappingLabelsPipeline
from oosaudit.stats import pesaran_timmermann


@pytest.mark.unit
def test_pesaran_timmermann_inflates_by_sqrt_h_under_repetition() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 0.03, 300)
    pred = -0.3 * y + rng.normal(0.0, 0.03, 300)
    lone, _ = pesaran_timmermann(pd.Series(y), pred)
    many, _ = pesaran_timmermann(pd.Series(np.repeat(y, 7)), np.repeat(pred, 7))
    assert abs(many) == pytest.approx(abs(lone) * np.sqrt(7), rel=0.05)


@pytest.mark.unit
def test_passes_at_horizon_one_where_labels_do_not_overlap() -> None:
    v = LabelOverlapProbe().run(NativeReplayable(CleanPipeline(horizon=1)))
    assert v.outcome == "pass"


@pytest.mark.unit
def test_escalates_to_defect_for_an_independence_assuming_test() -> None:
    v = LabelOverlapProbe().run(NativeReplayable(OverlappingLabelsPipeline()))
    assert v.outcome == "fail"
    assert v.finding_class == "defect"
    assert v.evidence["n_effective"] < v.evidence["n_reported"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_overlap.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.stats'`

- [ ] **Step 3: Implement `stats.py`**

```python
"""Statistics the probes need to recompute a pipeline's own claims.

Nothing here is novel; it exists so a probe can re-run a declared test under
non-overlapping sampling and compare.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

INDEPENDENCE_ASSUMING_TESTS = frozenset({"pesaran_timmermann"})


def pesaran_timmermann(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """Sign-predictability test. Every variance term divides by n, so the caller
    must pass non-overlapping observations."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    n = yt.size
    if n < 8:
        return float("nan"), float("nan")

    up_true = (yt > 0).astype(float)
    up_pred = (yp > 0).astype(float)
    p_y, p_x = up_true.mean(), up_pred.mean()
    hit = float(np.mean(up_true == up_pred))
    indep = p_y * p_x + (1 - p_y) * (1 - p_x)

    var_hit = indep * (1 - indep) / n
    var_indep = (
        ((2 * p_y - 1) ** 2) * p_x * (1 - p_x) / n
        + ((2 * p_x - 1) ** 2) * p_y * (1 - p_y) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n**2
    )
    denom = var_hit - var_indep
    if denom <= 0:
        return float("nan"), float("nan")
    stat = (hit - indep) / math.sqrt(denom)
    return float(stat), float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
```

- [ ] **Step 4: Implement the reference pipeline**

`src/oosaudit/references/overlapping_labels.py`:

```python
"""Broken reference: a 7-day target scored with a test that assumes independence."""

from __future__ import annotations

from .clean import CleanPipeline


class OverlappingLabelsPipeline(CleanPipeline):
    declared_test = "pesaran_timmermann"

    def __init__(self, seed: int = 7, horizon: int = 7, n: int = 900) -> None:
        super().__init__(seed=seed, horizon=horizon, n=n)
```

- [ ] **Step 5: Append probe 5 to `probes/specification.py`**

```python
@dataclass(frozen=True)
class LabelOverlapProbe:
    """An h-step label overlaps h-1 times; a test dividing by n inflates by ~sqrt(h)."""

    name: str = "label_overlap"
    finding_class: str = "specification"
    required_mutations: frozenset[str] = frozenset()

    def run(self, replayable: object) -> Verdict:
        base = replayable.run()  # type: ignore[attr-defined]
        h = base.meta.horizon
        n = len(base.forecasts)
        evidence = {"horizon": float(h), "n_reported": float(n), "n_effective": float(n // max(h, 1))}

        if h <= 1:
            return Verdict(
                self.name, "specification", "pass", None, evidence,
                "labels do not overlap at this horizon", "",
            )

        full_stat, full_p = pesaran_timmermann(base.y_true, base.forecasts.to_numpy())
        stats_, ps = [], []
        for phase in range(h):
            s, p = pesaran_timmermann(
                base.y_true.iloc[phase::h], base.forecasts.to_numpy()[phase::h]
            )
            if np.isfinite(s):
                stats_.append(s)
                ps.append(p)
        evidence |= {
            "stat_full": full_stat,
            "p_full": full_p,
            "stat_non_overlapping": float(np.mean(stats_)) if stats_ else float("nan"),
            "p_non_overlapping": float(np.mean(ps)) if ps else float("nan"),
        }

        assumes_independence = base.meta.declared_test in INDEPENDENCE_ASSUMING_TESTS
        hac_absent = base.meta.hac_lags is None or base.meta.hac_lags < h - 1
        if not (assumes_independence or hac_absent):
            return Verdict(
                self.name, "specification", "pass", None, evidence,
                "labels overlap but the declared test carries an adequate HAC bandwidth", "",
            )
        if assumes_independence:
            return Verdict(
                self.name, "defect", "fail", "major", evidence,
                f"the declared test '{base.meta.declared_test}' assumes independent "
                f"observations while {h}-step labels overlap {h - 1} times; the statistic "
                f"moves from {full_stat:.2f} to {evidence['stat_non_overlapping']:.2f} on "
                "non-overlapping subsamples", "",
            )
        return Verdict(
            self.name, "specification", "fail", "major", evidence,
            f"labels overlap {h - 1} times and the declared HAC bandwidth is "
            f"{base.meta.hac_lags}, below h-1", 
            "flips if the authors' bandwidth choice is defended on other grounds, "
            "for example a prewhitened or automatically selected bandwidth",
        )
```

Extend the imports at the top of `specification.py` with
`from ..stats import INDEPENDENCE_ASSUMING_TESTS, pesaran_timmermann`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_probe_overlap.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/oosaudit/stats.py src/oosaudit/probes/specification.py src/oosaudit/references/overlapping_labels.py tests/test_probe_overlap.py
git commit -m "feat: probe 5, label-overlap inflation"
```

---

### Task 11: Probe 6 — execution timing

**Files:**
- Modify: `src/oosaudit/probes/specification.py` (append)
- Create: `src/oosaudit/references/same_close_execution.py`
- Test: `tests/test_probe_execution.py`

**Interfaces:**
- Consumes: `Trace.prices`.
- Produces: `ExecutionTimingProbe(cost_bps=17.0, sharpe_drop_threshold=0.25)`;
  `SameCloseExecutionPipeline(seed=11, horizon=1, n=900)`.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_execution.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.specification import ExecutionTimingProbe
from oosaudit.references.same_close_execution import SameCloseExecutionPipeline


@pytest.mark.unit
def test_reports_both_sharpes_and_the_delta() -> None:
    v = ExecutionTimingProbe().run(NativeReplayable(SameCloseExecutionPipeline()))
    assert {"sharpe_same_close", "sharpe_delayed", "delta_sharpe"} <= set(v.evidence)


@pytest.mark.unit
def test_fires_when_a_one_bar_delay_destroys_performance() -> None:
    v = ExecutionTimingProbe().run(NativeReplayable(SameCloseExecutionPipeline()))
    assert v.outcome == "fail"
    assert v.finding_class == "specification"
    assert v.rebuttal
    assert v.evidence["delta_sharpe"] < 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_execution.py -v`
Expected: FAIL, `ImportError: cannot import name 'ExecutionTimingProbe'`

- [ ] **Step 3: Implement the reference pipeline**

`src/oosaudit/references/same_close_execution.py`:

```python
"""Broken reference: a forecast that only works if you trade at the very close
you needed in order to compute it.

The feature is the current bar's own return, and the target is the next bar's,
built so the edge lives entirely in the same-close fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import CleanPipeline


class SameCloseExecutionPipeline(CleanPipeline):
    def __init__(self, seed: int = 11, horizon: int = 1, n: int = 900) -> None:
        super().__init__(seed=seed, horizon=horizon, n=n)

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        idx = pd.date_range("2019-01-01", periods=self.n, freq="D")
        eps = rng.normal(0.0, 0.02, self.n)
        # One-bar reversal: today's shock predicts tomorrow, and nothing beyond.
        rets = np.zeros(self.n)
        rets[1:] = 0.0005 - 0.45 * eps[:-1] + eps[1:]
        close = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.uniform(1e6, 2e6, self.n),
            },
            index=idx,
        )
```

- [ ] **Step 4: Append probe 6 to `probes/specification.py`**

```python
@dataclass(frozen=True)
class ExecutionTimingProbe:
    """Re-run the implied strategy with entry delayed by one bar."""

    cost_bps: float = 17.0
    sharpe_drop_threshold: float = 0.25
    name: str = "execution_timing"
    finding_class: str = "specification"
    required_mutations: frozenset[str] = frozenset()

    @staticmethod
    def _sharpe(positions: np.ndarray, rets: np.ndarray, cost: float, ppy: float) -> float:
        turn = np.abs(np.diff(np.concatenate([[0.0], positions])))
        net = positions * rets - turn * cost
        if net.size < 2 or net.std(ddof=1) == 0:
            return float("nan")
        return float(np.sqrt(ppy) * net.mean() / net.std(ddof=1))

    def run(self, replayable: object) -> Verdict:
        base = replayable.run()  # type: ignore[attr-defined]
        h = base.meta.horizon
        cost = self.cost_bps * 1e-4
        ppy = 365.0 / h

        px = base.prices
        idx = base.forecasts.index[::h]
        pos = np.sign(base.forecasts.loc[idx].to_numpy(float))
        positions = px.index.get_indexer(idx)

        def realised(shift: int) -> np.ndarray:
            entry, exit_ = positions + shift, positions + shift + h
            ok = exit_ < len(px)
            return np.log(px.to_numpy()[exit_[ok]] / px.to_numpy()[entry[ok]]), ok

        r0, ok0 = realised(0)
        r1, ok1 = realised(1)
        s0 = self._sharpe(pos[ok0], np.expm1(r0), cost, ppy)
        s1 = self._sharpe(pos[ok1], np.expm1(r1), cost, ppy)

        evidence = {
            "sharpe_same_close": s0,
            "sharpe_delayed": s1,
            "delta_sharpe": s1 - s0,
            "sign_flip": float(np.sign(s0) != np.sign(s1)),
        }
        if not np.isfinite(s0 - s1) or (s1 - s0) > -self.sharpe_drop_threshold:
            return Verdict(
                self.name, "specification", "pass", None, evidence,
                "a one-bar execution delay does not materially change performance", "",
            )
        return Verdict(
            self.name, "specification", "fail", "major", evidence,
            f"net Sharpe falls from {s0:.2f} to {s1:.2f} when entry is delayed one bar; "
            "the result depends on filling at the same close the features are computed from",
            "flips if execution is genuinely at the close, for example via "
            "market-on-close orders, or if the features exclude the current bar",
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_probe_execution.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/probes/specification.py src/oosaudit/references/same_close_execution.py tests/test_probe_execution.py
git commit -m "feat: probe 6, execution timing"
```

---

### Task 12: Probe 7 — resampled-null size

**Files:**
- Create: `src/oosaudit/probes/calibration.py`, `src/oosaudit/references/uncalibrated_inference.py`
- Modify: `src/oosaudit/stats.py` (append `clark_west`)
- Test: `tests/test_probe_calibration.py`

**Interfaces:**
- Consumes: `ResampleReturns`/`RESAMPLE_RETURNS` (Task 2).
- Produces: `clark_west(y_true, pred_model, pred_bench, lags) -> tuple[float, float]` in `stats.py`;
  `ResampledNullSizeProbe(replications=60, block=1, nominal=0.05)`;
  `UncalibratedInferencePipeline(seed=7, horizon=1, n=600)`.

Default `replications=60` keeps the control-matrix test under a minute. The spec's screening default
of 200 and published default of 2000 are caller choices.

- [ ] **Step 1: Write the failing test**

`tests/test_probe_calibration.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.calibration import ResampledNullSizeProbe
from oosaudit.references.clean import CleanPipeline
from oosaudit.references.uncalibrated_inference import UncalibratedInferencePipeline


@pytest.mark.unit
def test_never_emits_a_failure() -> None:
    v = ResampledNullSizeProbe(replications=20).run(NativeReplayable(CleanPipeline()))
    assert v.outcome == "measured"
    assert v.finding_class == "measurement"


@pytest.mark.matrix
def test_zero_benchmark_measures_a_larger_size_than_the_drift_benchmark() -> None:
    clean = ResampledNullSizeProbe(replications=60).run(NativeReplayable(CleanPipeline()))
    broken = ResampledNullSizeProbe(replications=60).run(
        NativeReplayable(UncalibratedInferencePipeline())
    )
    assert broken.evidence["rejection_rate"] > clean.evidence["rejection_rate"]
    assert clean.evidence["mc_standard_error"] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_probe_calibration.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.probes.calibration'`

- [ ] **Step 3: Append `clark_west` to `stats.py`**

```python
def newey_west_lrv(x: np.ndarray, lags: int) -> float:
    """Bartlett long-run variance, every autocovariance normalised by n."""
    n = x.size
    centered = x - x.mean()
    lrv = float(np.mean(centered**2))
    for k in range(1, lags + 1):
        cov = float(np.sum(centered[k:] * centered[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lags + 1)) * cov
    return lrv


def clark_west(
    y_true: np.ndarray, pred_model: np.ndarray, pred_bench: np.ndarray, lags: int = 0
) -> tuple[float, float]:
    """MSPE-adjusted nested comparison, one-sided against a standard normal.

    Uses the identity f_t = 2 (y - b)(m - b), which makes plain that the
    benchmark defines the hypothesis rather than merely the baseline.
    """
    f = 2.0 * (y_true - pred_bench) * (pred_model - pred_bench)
    f = f[np.isfinite(f)]
    if f.size < 8:
        return float("nan"), float("nan")
    lrv = newey_west_lrv(f, lags)
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = float(f.mean() / math.sqrt(lrv / f.size))
    return stat, float(1.0 - stats.norm.cdf(stat))
```

- [ ] **Step 4: Implement the reference pipeline**

`src/oosaudit/references/uncalibrated_inference.py`:

```python
"""Broken reference: the pipeline tests against a zero benchmark on a drifting
series, so its own declared test is oversized for the null it is read as."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import CleanPipeline


class UncalibratedInferencePipeline(CleanPipeline):
    def benchmark(self, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X_test), dtype=float)
```

- [ ] **Step 5: Implement `probes/calibration.py`**

```python
"""Measure the size of a pipeline's own declared test under a constructed null.

Resampling the return series into a synthetic path makes future returns
independent of every past-information feature by construction, while drift,
tails, feature persistence, sample length and the estimators survive. Forecasts
are genuinely re-estimated at every origin, so the estimation-noise term the
adjusted statistic exists to remove is present -- which a simulation using an
exogenous noise forecast cannot deliver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mutations import RESAMPLE_RETURNS, ResampleReturns
from ..stats import clark_west
from ..verdict import Verdict
from .base import negotiate


@dataclass(frozen=True)
class ResampledNullSizeProbe:
    replications: int = 60
    block: int = 1
    nominal: float = 0.05
    name: str = "resampled_null_size"
    finding_class: str = "measurement"
    required_mutations: frozenset[str] = frozenset({RESAMPLE_RETURNS})

    def run(self, replayable: object) -> Verdict:
        skip = negotiate(self, replayable)
        if skip is not None:
            return skip

        rejects = 0
        usable = 0
        for i in range(self.replications):
            trace = replayable.run(  # type: ignore[attr-defined]
                ResampleReturns(block=self.block, seed=1000 + i)
            )
            lags = trace.meta.hac_lags if trace.meta.hac_lags is not None else 0
            _, p = clark_west(
                trace.y_true.to_numpy(float),
                trace.forecasts.to_numpy(float),
                trace.benchmark.to_numpy(float),
                lags=lags,
            )
            if np.isfinite(p):
                usable += 1
                rejects += int(p < self.nominal)

        rate = rejects / usable if usable else float("nan")
        se = math.sqrt(rate * (1 - rate) / usable) if usable else float("nan")
        scheme = "iid (measures size)" if self.block <= 1 else "block (bounds dependence)"
        return Verdict(
            self.name, "measurement", "measured", None,
            {
                "rejection_rate": rate,
                "nominal": self.nominal,
                "mc_standard_error": se,
                "replications_usable": float(usable),
                "block": float(self.block),
            },
            f"rejection rate {rate:.1%} against a nominal {self.nominal:.0%} under a "
            f"{scheme} resampled null over {usable} usable replications", "",
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_probe_calibration.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add src/oosaudit/probes/calibration.py src/oosaudit/references/uncalibrated_inference.py src/oosaudit/stats.py tests/test_probe_calibration.py
git commit -m "feat: probe 7, resampled-null size measurement"
```

---

### Task 13: The control matrix

**Files:**
- Create: `src/oosaudit/registry.py`
- Test: `tests/test_control_matrix.py`

**Interfaces:**
- Consumes: every probe and every reference pipeline.
- Produces: `ALL_PROBES: tuple[Probe, ...]`, `REFERENCE_PIPELINES: dict[str, Callable[[], object]]`,
  and `EXPECTED_FIRING: dict[str, str]` mapping reference name → the one probe that must fail on it.

This is the suite's central claim: each probe fires on its own break, stays silent on the clean
pipeline, and stays silent on the other six breaks. Sensitivity and specificity carry equal weight.

- [ ] **Step 1: Write the failing test**

`tests/test_control_matrix.py`:

```python
from __future__ import annotations

import pytest

from oosaudit.harness.native import NativeReplayable
from oosaudit.registry import ALL_PROBES, EXPECTED_FIRING, REFERENCE_PIPELINES


@pytest.mark.matrix
@pytest.mark.parametrize("reference", sorted(REFERENCE_PIPELINES))
def test_only_the_intended_probe_fires(reference: str) -> None:
    replayable = NativeReplayable(REFERENCE_PIPELINES[reference]())
    expected = EXPECTED_FIRING[reference]
    failed = {
        p.name for p in ALL_PROBES if p.run(replayable).outcome == "fail"
    }
    assert failed == ({expected} if expected else set()), (
        f"{reference}: expected {expected or 'no'} failure, got {sorted(failed)}"
    )


@pytest.mark.matrix
def test_every_probe_is_exercised_by_exactly_one_reference() -> None:
    firing = [v for v in EXPECTED_FIRING.values() if v]
    assert sorted(firing) == sorted({p.name for p in ALL_PROBES} - {"resampled_null_size"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_control_matrix.py -v -m matrix`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.registry'`

- [ ] **Step 3: Implement `registry.py`**

```python
"""The probe roster and the control matrix that validates it.

EXPECTED_FIRING is the suite's own ground truth: each broken reference must
trip exactly one probe. A probe that fires elsewhere is not more sensitive, it
is less specific, and against a real corpus that produces false accusations.

resampled_null_size is absent from the mapping because it is a measurement and
never emits a failure.
"""

from __future__ import annotations

from collections.abc import Callable

from .probes.calibration import ResampledNullSizeProbe
from .probes.leakage import InternalHoldoutProbe, LookaheadProbe, MissingBarPurgeProbe
from .probes.specification import (
    BenchmarkMismatchProbe,
    ExecutionTimingProbe,
    LabelOverlapProbe,
)
from .references.clean import CleanPipeline
from .references.leaky_feature import LeakyFeaturePipeline
from .references.overlapping_labels import OverlappingLabelsPipeline
from .references.same_close_execution import SameCloseExecutionPipeline
from .references.uncalibrated_inference import UncalibratedInferencePipeline
from .references.unpurged_holdout import UnpurgedHoldoutPipeline
from .references.unpurged_split import UnpurgedSplitPipeline
from .references.zero_benchmark_drift import ZeroBenchmarkDriftPipeline

ALL_PROBES = (
    LookaheadProbe(),
    MissingBarPurgeProbe(),
    InternalHoldoutProbe(),
    BenchmarkMismatchProbe(),
    LabelOverlapProbe(),
    ExecutionTimingProbe(),
    ResampledNullSizeProbe(replications=20),
)

REFERENCE_PIPELINES: dict[str, Callable[[], object]] = {
    "clean": CleanPipeline,
    "leaky_feature": LeakyFeaturePipeline,
    "unpurged_split": lambda: UnpurgedSplitPipeline(horizon=5),
    "unpurged_holdout": lambda: UnpurgedHoldoutPipeline(horizon=5, purge=0),
    "zero_benchmark_drift": ZeroBenchmarkDriftPipeline,
    "overlapping_labels": OverlappingLabelsPipeline,
    "same_close_execution": SameCloseExecutionPipeline,
    "uncalibrated_inference": UncalibratedInferencePipeline,
}

EXPECTED_FIRING: dict[str, str] = {
    "clean": "",
    "leaky_feature": "lookahead",
    "unpurged_split": "missing_bar_purge",
    "unpurged_holdout": "internal_holdout",
    "zero_benchmark_drift": "benchmark_mismatch",
    "overlapping_labels": "label_overlap",
    "same_close_execution": "execution_timing",
    "uncalibrated_inference": "benchmark_mismatch",
}
```

- [ ] **Step 4: Run the matrix and reconcile**

Run: `pytest tests/test_control_matrix.py -v -m matrix`

`uncalibrated_inference` shares its break with `zero_benchmark_drift` — both use a zero benchmark —
so `EXPECTED_FIRING` maps both to `benchmark_mismatch`, and the second assertion excludes
`resampled_null_size` for that reason. If any other probe fires on a reference it should not, fix the
**probe's** specificity rather than loosening the expectation. Loosening the matrix to make it pass is
the failure mode this whole suite exists to detect.

- [ ] **Step 5: Commit**

```bash
git add src/oosaudit/registry.py tests/test_control_matrix.py
git commit -m "test: control matrix, each probe fires only on its own break"
```

---

### Task 14: The audit runner

**Files:**
- Create: `src/oosaudit/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: `ALL_PROBES` (Task 13), `render_report` (Task 3).
- Produces: `AuditResult(verdicts, deterministic, report)`;
  `audit(replayable, probes=ALL_PROBES, tolerance=1e-12) -> AuditResult`.

- [ ] **Step 1: Write the failing test**

`tests/test_audit.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from oosaudit.audit import audit
from oosaudit.harness.native import NativeReplayable
from oosaudit.probes.leakage import LookaheadProbe
from oosaudit.references.clean import CleanPipeline


class NonDeterministic(CleanPipeline):
    def predict(self, X):  # type: ignore[no-untyped-def]
        base = super().predict(X)
        return base + np.random.default_rng().normal(0, 1e-6, len(base))


class Exploding(CleanPipeline):
    def load(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("package will not run")


@pytest.mark.unit
def test_clean_pipeline_produces_no_failures() -> None:
    result = audit(NativeReplayable(CleanPipeline()), probes=[LookaheadProbe()])
    assert result.deterministic
    assert [v.outcome for v in result.verdicts] == ["pass"]
    assert "lookahead" in result.report


@pytest.mark.unit
def test_non_determinism_forces_mutation_probes_to_inconclusive() -> None:
    result = audit(NativeReplayable(NonDeterministic()), probes=[LookaheadProbe()])
    assert not result.deterministic
    assert result.verdicts[0].outcome == "inconclusive"


@pytest.mark.unit
def test_a_pipeline_that_will_not_run_yields_error_not_a_crash() -> None:
    result = audit(NativeReplayable(Exploding()), probes=[LookaheadProbe()])
    assert result.verdicts[0].outcome == "error"
    assert "will not run" in result.verdicts[0].detail
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'oosaudit.audit'`

- [ ] **Step 3: Implement `audit.py`**

```python
"""Run every probe against one replayable and assemble the report.

Two rules here are not defensive polish. A pipeline that will not execute still
counts -- coverage statistics that quietly drop the packages that would not
build are the exact dishonesty this suite exists to catch. And a pipeline whose
baseline is not reproducible forces every mutation-based probe to inconclusive,
because a run-to-run difference is otherwise indistinguishable from leakage,
which would produce false accusations against named authors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .registry import ALL_PROBES
from .verdict import Verdict, render_report


@dataclass(frozen=True)
class AuditResult:
    verdicts: tuple[Verdict, ...]
    deterministic: bool
    report: str


def _is_deterministic(replayable: object, tolerance: float) -> bool:
    first = replayable.run()  # type: ignore[attr-defined]
    second = replayable.run()  # type: ignore[attr-defined]
    if not first.forecasts.index.equals(second.forecasts.index):
        return False
    diff = np.abs(first.forecasts.to_numpy() - second.forecasts.to_numpy()).max()
    return bool(diff <= tolerance)


def audit(
    replayable: object,
    probes: Sequence[object] = ALL_PROBES,
    tolerance: float = 1e-12,
) -> AuditResult:
    try:
        deterministic = _is_deterministic(replayable, tolerance)
    except Exception as exc:  # the package will not execute at all
        verdicts = tuple(
            Verdict(p.name, p.finding_class, "error", None, {}, str(exc), "")  # type: ignore[attr-defined]
            for p in probes
        )
        return AuditResult(verdicts, False, render_report(verdicts))

    out: list[Verdict] = []
    for probe in probes:
        needs_rerun = bool(probe.required_mutations)  # type: ignore[attr-defined]
        if not deterministic and needs_rerun:
            out.append(
                Verdict(
                    probe.name,  # type: ignore[attr-defined]
                    probe.finding_class,  # type: ignore[attr-defined]
                    "inconclusive",
                    None,
                    {},
                    "baseline runs differ, so a mutation-induced change cannot be "
                    "distinguished from run-to-run noise",
                    "",
                )
            )
            continue
        try:
            out.append(probe.run(replayable))  # type: ignore[attr-defined]
        except Exception as exc:
            out.append(
                Verdict(
                    probe.name,  # type: ignore[attr-defined]
                    probe.finding_class,  # type: ignore[attr-defined]
                    "error", None, {}, f"probe raised: {exc}", "",
                )
            )

    verdicts = tuple(out)
    return AuditResult(verdicts, deterministic, render_report(verdicts))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_audit.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the whole suite with coverage and the quality gate**

Run:
```bash
pytest --cov=oosaudit --cov-report=term-missing
ruff check src tests
mypy
```
Expected: all pass; `src/oosaudit/probes/` at 100%; overall ≥ 80%.

- [ ] **Step 6: Commit**

```bash
git add src/oosaudit/audit.py tests/test_audit.py
git commit -m "feat: audit runner with determinism and error handling"
```

---

## Self-review

**Spec coverage.** Trace/Replayable/Mutation core → Task 1–2. Verdict schema with three finding
classes and mandatory rebuttal → Task 3. Native harness and `Pipeline` protocol → Task 4. Capability
negotiation producing `not_applicable` by construction → Task 5. Probes 1–7 → Tasks 6–12. Eight
reference pipelines → Tasks 4, 6, 7, 8, 9, 10, 11, 12. Control matrix → Task 13. Determinism check,
unbuildable packages staying in the denominator, probe-raises handling → Task 14. Phases 1–4 of the
spec's phasing table are covered; phase 5 is out of scope by the scope check at the top.

**Deviations, both deliberate and flagged inline.** Probe 4 uses a new `zero_features` mutation
because a `Trace` does not expose the fitted model. Probe 3 requires the pipeline to disclose
`internal_holdout_positions` and returns `not_applicable` otherwise, which is the spec's documented
deep-tier-only limitation made concrete.

**Type consistency.** `Verdict` field order is fixed at Task 3 and every probe constructs it
positionally in that order. `run(mutation=None)` is the single `Replayable` entry point throughout.
`required_mutations` is a `frozenset[str]` on every probe, matched against `supported_mutations` on
every harness. `pesaran_timmermann` and `clark_west` both return `tuple[float, float]` as
`(statistic, p_value)`.

**Known cross-reference.** Task 7's `MissingBarPurgeProbe` uses `np`, and Task 6 already imports
`numpy as np` at the top of `leakage.py`; no second import is needed.
