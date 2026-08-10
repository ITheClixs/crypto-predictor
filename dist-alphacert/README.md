# alphacert

Anytime-valid certificates and ceilings for out-of-sample predictive ability.

A *certificate* is a non-negative wealth process `E_t` with `E_0 = 1` whose expectation never
exceeds 1 under the null. Ville's inequality then gives, simultaneously at every `t`,

```
P( there exists t with E_t >= 1/alpha )  <=  alpha
```

so `1 / max_t E_t` is a p-value valid at any stopping time. You may monitor continuously, stop
when the evidence looks good, and resume, with no multiplicity correction and no pre-committed
sample size.

## Install

```bash
pip install ./dist-alphacert
```

Depends on `numpy` and `scipy`. Nothing else.

## The problem it solves

Nested forecast-comparison tests handle the outcome's unconditional drift by plugging in an
estimate of it. When the benchmark is the zero forecast, the statistic loads on that drift and
rejects for assets that merely went up. Measured, on a signal that is **pure noise**:

| Asset's annualised Sharpe | Clark–West vs. zero | Clark–West vs. mean | `certify` |
|---|---|---|---|
| 0.0 | 0.053 | 0.058 | 0.003 |
| 0.8 | 0.118 | 0.058 | 0.003 |
| 1.6 | 0.560 | 0.055 | 0.010 |
| 2.0 | **0.810** | 0.040 | 0.008 |

`certify` removes the drift instead of estimating it, by taking an infimum over the nuisance of
a convex combination of a signal-betting martingale and a drift-centring martingale. The bound
holds uniformly in the drift.

## Quickstart

```python
import numpy as np
from alphacert import certify, mean_ceiling, anytime_validity_cost

rng = np.random.default_rng(0)
y = 0.001 + 0.03 * rng.standard_normal(2000)     # drifting, unpredictable

certify(rng.standard_normal(2000), y).rejects(0.05)   # False -- noise is not certified

signal = 0.02 * rng.standard_normal(2000)
certify(signal, y + signal).rejects(0.05)             # True  -- a real edge is

# What a return stream could still be worth, whether or not anything rejects.
returns = 0.04 * rng.standard_normal(600)
mean_ceiling(returns, return_bound=0.5).sharpe_ceiling(0.04)   # 0.469

anytime_validity_cost()   # 1.903 -- the sample-size price of validity at every stopping time
```

## What's in it

| Module | Purpose |
|---|---|
| `certificate` | `certify()` — the drift-robust e-process for signal-vs-benchmark |
| `stream` | `certify_mean()`, `mean_ceiling()` — for an already-formed strategy's returns, where the drift *is* the estimand |
| `bounds` | `value_ceiling()` — time-uniform upper bound on incremental value |
| `merge` | `merge_average()`, `merge_product()`, `e_bh()` — merging and FDR control under arbitrary dependence |
| `design` | `detection_horizon()`, `certifiable_ratio()`, `anytime_validity_cost()` — how much data a given effect needs |
| `payoffs` | `IDENTITY` (default), `TANH`, `SIGN` |

## What it costs

Stated because a method that reports only its advantages is not usable for deciding whether to
adopt it. Against a **correctly specified** Clark–West test, at matched empirical size:

| True annualised IR | Clark–West | `certify` (learned stakes) | `certify` (pre-committed) |
|---|---|---|---|
| 1.0 | 0.210 | 0.143 | 0.163 |
| 1.5 | 0.495 | 0.335 | 0.390 |
| 2.0 | 0.850 | 0.710 | 0.758 |
| 3.0 | 1.000 | 0.983 | 0.998 |

Clark–West wins wherever it is well specified. Use `alphacert` when the drift is genuinely a
nuisance, when monitoring is sequential, or when you want a bound rather than a verdict.

Roughly a factor of `1.90` of that gap is the price of anytime-validity itself and is not
specific to this construction — `anytime_validity_cost()` computes it.

## Two traps, both guarded by tests

**The drift grid must resolve the drift.** The infimum is numerical, and the spacing must be
finer than the sampling error `sigma / sqrt(n)`, not merely finer than `sigma`. Too coarse and
the numerical infimum sits above the true one and leaks type-I error — measured at 0.274
against a nominal 0.05 before this was diagnosed. Use `recommended_resolution()`.

**`wealth` is the running maximum; `raw_wealth` is the process.** Ville's inequality bounds the
supremum, so a *test* must use `wealth`. But `wealth` is non-decreasing by construction, so
asking "when was this rising?" of it answers "when did it set a new high" — any *diagnostic*
must use `raw_wealth`. Getting this backwards produced a wrong duty-cycle estimate once; there
are now tests pinning both.

## Tests

```bash
pytest tests/test_alphacert_*.py -q     # ~750 lines, calibration and power included
```

Calibration is verified under the null at several drifts, under skewed innovations, under a
GARCH null, and against a documented failure mode. Power is measured rather than asserted.

## License

PolyForm Noncommercial 1.0.0 — free for research, teaching and personal use; commercial use
requires a separate licence.
