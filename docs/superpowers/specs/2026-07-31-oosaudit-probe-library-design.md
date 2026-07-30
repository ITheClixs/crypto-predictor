# oosaudit — probe library design

**Date:** 2026-07-31
**Status:** approved, ready for implementation planning
**Scope:** sub-project 1 of 3. This spec covers the probe library only.

---

## Context

The `crypto-predictor` audit (`audit/FORENSIC_REVIEW.md`) found four critical defects in a
competently engineered, fully reproducible, 154-test return-predictability study. Every headline
result died. None of the defects was visible in code review, and none was caught by a test suite that
checked every statistical formula against hand-computed reference values.

The literature gate (`audit/RESEARCH_PIVOT_MEMO.md`) then established that the *econometrics* of those
defects is largely known: Clark & West (2006) define their null as a zero-mean martingale difference;
the recursive-mean benchmark has been standard since Goyal–Welch and Campbell–Thompson (2008); and
Pincheira, Hardy & Muñoz (2021) already documented Clark–West's long-horizon size distortion and
shipped the Wild Clark–West fix. What is *not* established is that applied pipelines routinely commit
these errors, or that they can be detected mechanically.

The transferable output of the audit is therefore not the crypto result and not the size number. It is
the set of cheap checks that caught them. This project packages those checks so they can be run
against pipelines other than our own.

**Target:** NeurIPS Datasets & Benchmarks. The contribution is the artifact plus a corpus audit, not a
theorem.

## Goals

1. Seven probes, each detecting one specific failure mode, each unit-testable in isolation.
2. Runnable against third-party pipelines without rewriting them.
3. Verdicts that distinguish objective defects from specification disagreements, so the corpus paper
   does not pick fights it cannot defend.
4. Honest coverage accounting: probes that cannot run say so, and packages that will not execute stay
   in the denominator.

## Non-goals

- Not a forecasting library. It ships no models.
- Not a reproduction tool. It does not check whether a package reproduces its published numbers.
- Not crypto-specific. The corpus is equity and options ML from journal replication archives; the
  crypto study is one worked example.
- No leaderboard, no hidden test set, no submission server.

---

## Corpus decision (settled, drives this design)

Established by the feasibility survey on 2026-07-31:

- RFS Dataverse holds **433 replication packages**, queryable via the Harvard Dataverse API with
  `subtree=rfs`. Policy in force for conditional accepts since 2020-07-01; each package is verified by
  a Data Editor before release. Review of Finance and the Journal of Finance run parallel policies.
- Keyword counts within that collection: ~20 "machine learning", ~27 "return predictability", 9
  "neural network", **0 "cryptocurrency"**.
- On-target packages confirmed by title, including *Option Return Predictability with Machine Learning
  and Big Data* (code and data), *Machine Learning and the Implementable Efficient Frontier*,
  *Confident Risk Premiums using Machine Learning Uncertainties*, *Man versus Machine Learning
  Revisited*, *Universal Portfolio Shrinkage*.
- Crypto ML publishes in journals without code mandates. Every crypto implementation found is a
  third-party replication, which cannot support a claim about a published result.
- Proprietary data is replaced by synthetic or pseudo equivalents in released packages (confirmed:
  *Replication Code and Pseudo-Data for "The Elasticity of Quantitative Investment"*). This does not
  block us: probes 1–5 test properties of **code**, not of data, and run on synthetic inputs.
- RFS ships `template-stata`, `template-python`, `template-r`. The corpus is trilingual.

**Consequences.** The corpus is equity/asset-pricing, not crypto. The adapter is Python-only; the
inclusion rule is declared ex ante and coverage is reported as "N of M candidate ML packages are
Python; we audit those."

---

## Architecture

Probes never see a pipeline. They see a `Trace`.

```
unmodified package ──harness──> Replayable ──rerun(mutation)──> Trace ──probe──> Verdict
```

**`Trace`** — the observable record of one pipeline execution:

| field | meaning |
|---|---|
| `origins` | forecast origin timestamps |
| `train_idx`, `test_idx` | index sets per origin |
| `features` | feature matrix as seen at each origin |
| `forecasts` | model forecasts, per test observation |
| `benchmark` | the pipeline's own declared benchmark forecast series |
| `y_true`, `y_time` | realised target and the timestamp at which it resolves |
| `prices` | price series, for execution-timing re-mapping |
| `meta` | horizon, declared test statistic, HAC bandwidth, seed |

**`Replayable`** — anything that can produce a `Trace` given a `Mutation`. Four probes must re-run on
perturbed inputs rather than observe once.

**`Mutation`** — a declared, enumerable transformation of the raw input data:
`perturb_after(t)`, `drop_bars(frac)`, `corrupt_rows(idx)`, `resample_returns(scheme)`.

### Why this split

- Probes become pure functions of a `Trace`, unit-testable against hand-built fixtures with no
  third-party code involved.
- Both tiers emit the same `Replayable`, so there is one probe core, not two.
- Capability negotiation is structural: a harness declares supported mutations, a probe declares
  required ones, and unsupported combinations return `not_applicable` **by construction**. Tier
  degradation cannot be silent.

### Two harnesses

| harness | tier | mechanism | probes |
|---|---|---|---|
| `harness/subprocess.py` | thin | manifest declares entry points; suite substitutes input files and invokes the package's own scripts; artifacts exchanged as Parquet/CSV | 1, 2, 4, 5, 6, 7 |
| `harness/native.py` | deep | package re-expressed against a Python `Protocol`; suite drives fit/predict directly | all 7 |

Screening strategy: run the thin harness across the corpus; reach for the deep port only where a probe
fires and the finding needs to survive an author's rebuttal.

```python
class Pipeline(Protocol):
    def load(self) -> pd.DataFrame: ...
    def features(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
```

---

## Verdict schema

```python
@dataclass(frozen=True)
class Verdict:
    probe: str
    finding_class: Literal["defect", "specification", "measurement"]
    outcome: Literal["pass", "fail", "measured", "not_applicable", "inconclusive", "error"]
    severity: Literal["critical", "major", "moderate", "minor"] | None
    evidence: dict[str, float]
    detail: str
    rebuttal: str
```

**`finding_class` is the load-bearing field.**

- `defect` — the code either leaks or it does not. A firing probe is a bug report, and the author's
  intent is irrelevant.
- `specification` — the author may have meant the choice you think is wrong. Publishable only with a
  `rebuttal`.
- `measurement` — returns a number, not a judgement. Never emits `fail`.

**`rebuttal` pre-registers the author's defence before the result is seen.** Examples: "flips if the
authors intended a joint zero-drift null"; "flips if entry was at the next open". A `specification`
finding for which no `rebuttal` can be written is not published.

`evidence` always carries the measured numbers, so a reader can disagree with the verdict while
accepting the measurement.

---

## The seven probes

### 1. Look-ahead perturbation — `defect`, critical
Mutation `perturb_after(t)`: randomly rescale every raw field strictly after cut `t`.
**Passes iff** every feature value at origins ≤ `t` is bit-identical across the two traces.
Catches accidental centred windows, sign errors in shifts, non-causal smoothers.

### 2. Missing-bar purge — `defect`, major
Mutation `drop_bars(frac)`: remove a fraction of bars at random.
**Passes iff** the train/test gap still spans at least `h` in **calendar** time for every fold.
Catches purging implemented by row position where the leak is calendar-based.

### 3. Internal-holdout corruption — `defect`, major, **deep tier only**
Mutation `corrupt_rows(idx)`: replace labels in rows that should be purged from an internal
validation split.
**Passes iff** the fitted model is unchanged.

> **Known limitation, accepted.** Black-box, this probe cannot distinguish "purged its internal
> holdout correctly" from "has no internal holdout." Corrupting the final `h` training rows changes
> predictions either way, because those rows are legitimate training data in both cases. The probe
> requires visibility of the internal split and is therefore `not_applicable` in the thin harness. If
> the deep tier proves too expensive to reach in sub-project 2, this probe is cut rather than weakened.

### 4. Benchmark / restriction mismatch — `specification`, critical
No mutation. Compare the pipeline's declared `benchmark` series against the model's own prediction
with feature effects zeroed.
**Evidence:** max and mean absolute difference between the two series; drift share of the
Clark–West numerator, `E[y]·E[ŷ] / (E[y]·E[ŷ] + Cov(y, ŷ))`.
**Fails when** the declared benchmark is the zero forecast while the estimator fits an intercept.
**Rebuttal:** "the authors intended the joint martingale-difference null of Clark & West (2006)."

### 5. Label-overlap inflation — `specification`, escalating to `defect`, major
No mutation. Compare the target horizon `h` against forecast spacing and against the test statistic
declared in `meta`.
**Escalates to `defect`** when the declared statistic assumes independent observations and labels
overlap; stays `specification` when a HAC correction with a defensible bandwidth is present.
**Evidence:** statistic and p-value on the full sample versus non-overlapping subsamples across all
`h` phase offsets.

### 6. Execution timing — `specification`, major
No data mutation; re-map positions to entry delayed by one bar using `prices`.
**Evidence:** Δ Sharpe, count of sign flips, and whether any headline result crosses zero.
**Rebuttal:** "executed on the close via market-on-close orders."

### 7. Resampled-null size — `measurement`
Mutation `resample_returns(scheme)`: iid or block bootstrap of the return series into a synthetic
path, so future returns are independent of every past-information feature by construction while drift,
tails, feature persistence, sample length and the estimators are retained.
Re-runs the pipeline and its own declared test `R` times.
**Evidence:** rejection rate at the nominal level, with Monte Carlo standard error; per-scheme, since
iid measures size while block resampling bounds dependence sensitivity and does not.
Needs a cheap mode: `R` configurable, default 200 for screening, 2000 for a published number.

---

## Audit flow

1. Read the package manifest — entry point, data path, declared benchmark, horizon, declared test.
2. Baseline run, twice, → `Trace₀`, `Trace₀'`.
3. Determinism check (below).
4. Per probe: negotiate capability → apply mutations → emit `Verdict`.
5. Render a human report and a machine-readable JSON record.

## Error handling

| condition | outcome |
|---|---|
| package will not execute | `error`; **stays in the coverage denominator** |
| harness lacks a required mutation | `not_applicable`, with the reason recorded |
| probe raises | `error`; the audit continues |
| baseline traces differ beyond tolerance | every mutation-based probe → `inconclusive` |

The determinism check is not defensive polish. Unseeded pipelines are common, and without it a
run-to-run difference is indistinguishable from leakage — which would produce false accusations
against named authors.

Dropping unbuildable packages from the denominator is the specific dishonesty this suite exists to
detect, so the coverage report must show attempted, executed, and audited counts separately.

---

## Testing

Ship **eight synthetic reference pipelines**: one clean, plus one deliberately broken per probe — a
leaky centred feature, an unpurged split, an unpurged internal holdout, a zero benchmark under drift,
an overlapping-label sign test, same-close execution, and uncalibrated inference.

**Requirement:** each probe fails on its own broken pipeline, passes on the clean one, and passes on
the other six. Specificity is weighted equally with sensitivity — a probe that fires on everything is
useless against a real corpus.

The resulting 7×8 matrix is simultaneously the regression suite and Table 1 of the paper, and it is
the answer to "how do you know your probes work."

Beyond the matrix: unit tests per probe against hand-built `Trace` fixtures; a golden-file test for
report rendering; and a determinism-check test using a deliberately unseeded reference pipeline.

Coverage target 80% per repo convention, with the probe modules at 100% — they are the product.

---

## Module layout

Files stay in the 200–400 line band.

| file | holds |
|---|---|
| `trace.py` | `Trace`, `Replayable`, `Mutation`; validation |
| `verdict.py` | schema, severity, report rendering |
| `probes/leakage.py` | probes 1–3 |
| `probes/specification.py` | probes 4–6 |
| `probes/calibration.py` | probe 7 |
| `harness/native.py` | `Protocol` → `Replayable` |
| `harness/subprocess.py` | manifest → `Replayable` |
| `references/` | the eight synthetic reference pipelines |
| `cli.py` | `oosaudit run <manifest>`, `oosaudit report <run-id>` |

New repository. `crypto-predictor` becomes reference pipeline zero and a worked example, not the host.

---

## Implementation phasing

The risk table calls for validating the thin harness against real packages "before building probes
4–7", while probes are testable against the synthetic reference pipelines with no harness at all.
Resolving that: **probes are built against `harness/native.py` and the reference pipelines only.** The
subprocess harness is a later phase and does not gate probe development.

| phase | delivers | done when |
|---|---|---|
| 1 | `trace.py`, `verdict.py`, one reference pipeline (clean) | a hand-built `Trace` validates; report renders |
| 2 | probes 1–3, `harness/native.py`, broken references for each | 3×4 control matrix passes |
| 3 | probes 4–6, remaining broken references | 6×7 control matrix passes |
| 4 | probe 7, cheap mode | reproduces the audit's 14–22% / 5–6% split on reference pipeline zero |
| 5 | `harness/subprocess.py`, manifest schema, `cli.py` | two real RFS packages produce a valid `Trace` |

Phase 5 is where the manifest schema gets pinned, and where the design is most likely to need
revision. If it fails against both trial packages, phases 1–4 remain useful and the corpus strategy
reopens rather than the library being rewritten.

## Risks

| risk | mitigation |
|---|---|
| Probe 3 is deep-tier-only and may be uncuttable-but-unreachable | Accepted and documented above; cut rather than weaken |
| Thin harness cannot intercept enough of a messy package to build a usable `Trace` | Validate against two real RFS packages before building probes 4–7 |
| Python-only excludes most of the corpus | Report coverage honestly; the language split is measured in sub-project 2, not assumed |
| Corpus too small after filtering to Python ML packages | Sub-project 2 opens with an enumeration pass; if N < 10, the paper's framing reopens before any porting effort is spent |
| Specification findings provoke author disputes | `rebuttal` is mandatory and pre-registered; deep-tier port before publishing any named finding |

## Open questions deferred to later sub-projects

- Exact manifest schema — pinned once two real RFS packages have been wrapped.
- Whether to adopt Wild Clark–West as a reference-correct statistic inside probe 7, pending a full
  read of Pincheira, Hardy & Muñoz (2021).
- Corpus inclusion rule and its pre-registration — sub-project 2.
