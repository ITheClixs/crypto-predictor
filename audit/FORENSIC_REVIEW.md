# Forensic review — *On the Out-of-Sample Predictability of Short-Horizon Cryptocurrency Returns*

Audited commit: `83881c7`. Manuscript: `paper/paper.tex`. Results audited: `reports/results.csv`
(reproduced exactly from the committed parquet cache, `data/cache/`, sample end 2026-07-18).

Every number below was computed by a script in `audit/scripts/`. Nothing is asserted from reading
the manuscript alone.

## Verdict

The paper's central quantitative claim — *five of eighteen settings reject no-predictability against
0.9 expected from noise* — does not survive. The test used to produce those five rejections rejects
**14–22 % of the time on data containing no predictability at all**, so the correct comparison is
5 observed against 2.6–4.0 expected. There is no headline finding left.

Two further claims are mathematically false rather than merely fragile: the nesting claim
(Section VII below) and the claim that the validating simulation holds the null by construction
(Section VIII). Two reported significance results (the only significant sign-timing findings in the
paper, p ≈ 0.001) are artifacts of treating overlapping 7-day labels as independent and become
p ≈ 0.24–0.27 when they are not.

Severity summary:

| # | Issue | Severity |
|---|-------|----------|
| 1 | Zero-forecast benchmark is not the slopes-zero restriction of any model used | CRITICAL |
| 2 | The "Clark–West validation" simulation tests a false null; no estimation occurs in it | CRITICAL |
| 3 | Clark–West against the zero benchmark is 14–22 % oversized under a genuine null | CRITICAL |
| 4 | Pesaran–Timmermann treats overlapping h-step labels as independent | CRITICAL |
| 5 | R² and the hypothesis test use different benchmarks; the R² benchmark is the weaker one | MAJOR |
| 6 | HAC bandwidth h−1 gives zero lags at h = 1, where four of five rejections live | MAJOR |
| 7 | Backtest executes at the same close the features are computed from | MAJOR |
| 8 | Multiplicity family of 18 excludes most of the search | MAJOR |
| 9 | Sample end date unpinned; forecasts never persisted | MODERATE |
| 10 | Exposure-decomposition claim asserted, not measured | MODERATE |

---

## Issue 1 — The random walk is not nested in any model the study fits

**CLAIM IN THE PAPER.** *"Every one of the three collapses to the random walk when its slopes are
zero. They are nested extensions of the benchmark"* (§Forecasters). Also `README.md:224`.

**WHAT THE CODE IMPLEMENTS.** `models/linear.py` builds `Pipeline([StandardScaler, Ridge(alpha=1)])`
and the same for `ElasticNet`. Neither passes `fit_intercept=False`, so both fit an unpenalised
intercept on standardised features. `models/trees.py` constructs `XGBRegressor` without
`base_score`, and XGBoost 3.0.4 estimates it from the training labels.

Measured on BTC, h = 1, fold 0 (`audit/scripts/` intercept check):

```
ridge        fit_intercept=True  intercept_=+1.712990e-03  train mean(y)=+1.712990e-03  ratio=1.000000
elastic_net  fit_intercept=True  intercept_=+1.712990e-03  train mean(y)=+1.712990e-03  ratio=1.000000
gbm          booster base_score=+1.975228e-03             train mean(y)=+1.712990e-03
```

The intercept equals the training-window mean of the target to machine precision. Zeroing the
feature effects therefore yields `ŷ = ȳ_train`, not `ŷ = 0`.

**RELEVANT THEORY.** Clark–West requires model 2 to reduce to model 1 when the extra parameters are
set to zero. Testing against the zero forecast is a legitimate special case (Clark & West 2006,
martingale-difference version) but it tests a *joint* null: `β₀ = 0 and β = 0`, i.e. zero
conditional mean including zero drift. Testing feature-based predictability requires
`β = 0` with `β₀` recursively estimated — the recursive historical mean the paper already computes
for its R².

**WHY IT MATTERS.** BTC's realised drift over the sample is +0.00102 per day (37 % annualised). The
Clark–West adjusted differential has the exact closed form

```
f_t = 2 (y_t − ŷ_b,t)(ŷ_m,t − ŷ_b,t)   →   f_t = 2 y_t ŷ_m,t   when ŷ_b = 0
```

so `E[f_t] = 2(E[y]E[ŷ] + Cov(y, ŷ))`. The first term is pure drift × long bias and is nonzero
whatever the features do. Measured drift share of the numerator (`audit/scripts/retest.py`):
9 % for BTC h=1 ridge, but 33 % for BTC h=1 GBM, 41–61 % across BTC h=7, and 96–115 % for the
benchmark forecasters. The paper's own results table contains the tell: `historical_mean`, which
uses *no* features, scores CW = +1.03 (BTC h=1) and +1.01 (BTC h=7) against the zero benchmark.

**SEVERITY.** CRITICAL. Not because the drift term explains all of the BTC h=1 result — it does not —
but because the estimand is not the one the paper interprets, and the composition of the rejecting
set changes completely when it is fixed.

**POSSIBLE AUTHOR DEFENCE.** "Clark & West (2006) explicitly cover the zero benchmark, so the test
is valid." **Does it succeed?** Partly. The test is valid *for the martingale-difference null*. It
does not succeed as a defence of the paper's interpretation, which is about features.

**REQUIRED REPAIR.** Test against the recursively estimated mean. Recomputed
(`audit/scripts/retest.py`, HAC lag h−1 as in the paper):

| setting | CW vs zero (paper) | p | CW vs recursive mean | p |
|---|---|---|---|---|
| BTC 1 ridge | 2.62 | 0.004 | 2.66 | 0.004 |
| BTC 1 elastic_net | 2.57 | 0.005 | 2.67 | 0.004 |
| BTC 1 gbm | 2.02 | 0.022 | 1.75 | 0.040 |
| BTC 7 ridge | 1.64 | 0.050 | 1.36 | 0.086 |
| BTC 7 elastic_net | 1.76 | 0.039 | 1.54 | 0.061 |
| ETH 1 ridge | 1.86 | 0.031 | 1.76 | 0.039 |
| ETH 7 ridge | −0.79 | 0.785 | −1.62 | 0.948 |
| SOL 1 gbm | 0.82 | 0.207 | **1.87** | **0.031** |
| SOL 7 ridge | −1.43 | 0.923 | **2.01** | **0.022** |
| SOL 7 elastic_net | −1.41 | 0.921 | **2.28** | **0.011** |

Raw 5 % rejections: 5/18 → 7/18. **The identity of the rejecting settings changes.** SOL, which the
paper reports as having no signal at either horizon, becomes the strongest rejecter at h = 7.

**CLAIMS AFFECTED.** Abstract ("5 of 18"); §Results-tests narrative *"Four of the five are BTC and
four are at the one-day horizon. That concentration is consistent with a weak short-horizon effect
in the most liquid asset"* — this sentence is an artifact of the benchmark choice and does not
survive. Conclusion ("a faint statistical signal in BTC at the one-day horizon").

**DOES THE CURRENT CONCLUSION SURVIVE?** No. The direction of the correction is not
conclusion-preserving in the convenient way — the corrected test finds *more* raw rejections, in
*different* places, and the geographic story is gone.

---

## Issue 2 — The simulation that validates the paper's main methodological argument tests a false null

**CLAIM IN THE PAPER.** *"We verify the bias by simulation rather than asserting it. Over 200
replications in which the larger model forecasts pure noise, so that the null holds by
construction, the mean Diebold–Mariano statistic is positive and it rejects in favour of the
benchmark far above the nominal rate, while Clark–West remains centred on zero and holds its size."*
(§Clark–West). Repeated in `README.md:313-316` and in the Reproducibility paragraph.

**WHAT THE CODE IMPLEMENTS.** `tests/test_stats.py:103`:

```python
y = pd.Series(rng.normal(0.0, 0.03, 250))
useless = rng.normal(0.0, 0.01, 250)  # independent of y
bench = np.zeros(250)
```

**RELEVANT THEORY / DERIVATION.**

```
MSPE_bench = E[(y−0)²]  = σ_y²          = 9.00e-4
MSPE_model = E[(y−u)²]  = σ_y² + σ_u²   = 1.00e-3
```

The population MSPEs differ by `σ_u² = 1e-4`. **The null of equal predictive accuracy is false by
construction, not true by construction.** Monte Carlo over 5 000 replications
(`audit/scripts/sim_audit.py`) confirms the mean sample loss differential is 1.000e-4, exactly σ_u².

Second defect: `useless` is exogenous noise, not an estimated forecast. No parameter is estimated
anywhere in the simulation. The estimation-noise term that Clark–West exists to remove is therefore
absent, so the experiment cannot measure the size of either statistic under a nested-*estimation*
null — which is the only null that matters for the paper.

Third: Clark–West's apparent centring is pure algebra. With `ŷ_b = 0`, `f_t = 2 y_t u_t`, and
`E[2yu] = 0` for *any* independent `u`, however bad a forecast it is. Numerical check over 200 000
draws: `E[2yu] = −9.7e-7 ± 1.3e-6`. A statistic that is centred for every independent forecast
cannot be evidence that the statistic has correct size.

**MEASURED CONSEQUENCE.** DM rejects 75.0 % of the time in this simulation. The paper calls that
rejecting "far above the nominal rate". It is *power against a false null*, correctly obtained. The
test suite encodes the misinterpretation as an assertion (`assert dm_rejects > 0.25`).

**SEVERITY.** CRITICAL. The simulation is the paper's only evidence that its choice of test is the
right one, and it does not support that conclusion.

**REQUIRED REPAIR.** Delete the claim; replace the experiment with a genuine estimated-nested-model
Monte Carlo (Issue 3). Fix `tests/test_stats.py:103` — the test name and docstring both state the
false claim.

---

## Issue 3 — Under a genuine nested null, the paper's test rejects 14–22 % of the time at a nominal 5 %

**EXPERIMENT** (`audit/scripts/mc_null.py`). Resample BTC daily log returns into a synthetic price
path, so future returns are independent of every past-information feature by construction while the
real drift, fat tails, feature collinearity and persistence, sample length, purged walk-forward
geometry, and the actual estimators (fitted intercepts included) are all retained. Forecasts are
genuinely estimated at every origin. 400 replications per cell.

`block=1` is an iid bootstrap of returns: a clean null, so these columns measure **size**.
`block=21` preserves 21-day dependence blocks from the real series; some genuine within-block
predictability may survive resampling, so those columns are a **dependence-sensitivity probe and an
upper bound**, not size.

Rejection rate at nominal 5 % (Monte Carlo s.e. ≈ 1.2–2.5 pp):

| statistic | h=1, iid | h=1, block 21 | h=7, iid | h=7, block 21 |
|---|---|---|---|---|
| CW vs **zero** (the paper's test), ridge | **14.2 %** | 53.8 % | **23.5 %** | 28.2 % |
| CW vs **zero**, elastic net | **22.0 %** | 43.0 % | **26.2 %** | 27.8 % |
| CW vs **recursive mean**, ridge | 6.2 % | 42.8 % | 10.2 % | 11.2 % |
| CW vs **recursive mean**, elastic net | 6.2 % | 28.2 % | 9.8 % | 10.8 % |
| CW vs zero, **drift-only model (no features)** | 27.3 % | 27.0 % | 39.0 % | 34.0 % |
| DM vs zero, ridge (two-sided) | 57.5 % | 15.2 % | 50.0 % | 46.5 % |

Separate run including the actual early-stopped XGBoost configuration (h = 1, iid, 150 reps,
`audit/scripts/mc_null_gbm.py`):

| statistic | rejection at nominal 5 % |
|---|---|
| CW vs zero, GBM | 22.0 % (± 3.4) |
| **CW vs recursive mean, GBM** | **5.3 % (± 1.8)** |
| CW vs recursive mean, ridge (same run) | 6.0 % (± 1.9) |

This answers the open question of whether Clark–West asymptotics survive greedy split selection,
subsampling, and a data-dependent number of trees. Empirically, at h = 1 against the recursive-mean
benchmark under an iid-return null, **they do** — the boosted model is no worse calibrated than
ridge. That is calibration evidence for this estimator, this configuration and this null, not a
theorem, and it does not extend to h = 7 (≈10 %) or to dependence-preserving resampling. But it
converts a hand-wave in the current manuscript into a measurement, and it is the most useful
by-product of this audit.

Three readings, all consequential:

1. **The paper's arithmetic is wrong by a factor of 3–4.** *"Eighteen settings tested at the 5 %
   level produce 0.05 × 18 = 0.9 rejections from noise alone"* assumes the test is correctly sized.
   At the measured size the expectation is 18 × 0.142 … 0.220 = **2.6 to 4.0**. Observed: 5. Against
   a calibrated null there is essentially nothing to explain.
2. **Fixing the benchmark fixes most of the size distortion**, at h = 1: 14–22 % → 6.2 %. The
   distortion is the drift channel of Issue 1, confirmed independently by the drift-only row
   (27 % rejection with zero feature information).
3. **At h = 7 the corrected test is still oversized** (≈10 % at nominal 5 %), which is the
   bandwidth problem of Issue 6. And under realistic dependence the corrected test is not
   trustworthy at all (28–43 %). Any credible rejection count needs dependence-robust inference —
   block bootstrap of the loss differential or a wild bootstrap — which the study does not
   implement.

**SEVERITY.** CRITICAL. This is the experiment that decides whether the paper has a finding.

---

## Issue 4 — The only significant sign-timing results are an overlapping-label artifact

**CLAIM IN THE PAPER.** *"Two are significantly anti-predictive: ETH ridge and ETH elastic net at
h = 7 give S = −3.29 and S = −3.33, both p ≈ 0.001."*

**WHAT THE CODE IMPLEMENTS.** `evaluate/stats.py:163` divides by `n`, and `study.py:77` calls it on
the full OOS frame — 2 174 rows of 7-day labels overlapping 6 times each. The paper's own
Figure 3 builds its coin-flip band from `n/h`. The formal test and the figure therefore use
effective sample sizes differing by a factor of 7.

**MEASURED REPAIR** (`audit/scripts/pt_and_exec.py`), Pesaran–Timmermann on non-overlapping
subsamples, averaged over all h phases:

| setting | n used (paper) | S (paper) | p (paper) | n_eff | S (non-overlapping) | p | worst phase p |
|---|---|---|---|---|---|---|---|
| ETH 7 ridge | 2 174 | −3.29 | **0.001** | 310 | −1.27 | 0.268 | 0.446 |
| ETH 7 elastic_net | 2 174 | −3.33 | **0.001** | 310 | −1.27 | 0.243 | 0.481 |
| SOL 7 ridge | 1 709 | −1.73 | 0.083 | 244 | −0.66 | 0.597 | 1.000 |
| BTC 7 ridge | 2 174 | +1.41 | 0.159 | 310 | +0.54 | 0.592 | 0.751 |

No sign-timing result in the paper, positive or negative, is significant once labels are
non-overlapping. The paper's discussion of these two results ("*Presented as p = 0.001 alone they
read as the strongest findings in the study*") is a rhetorical point built on a statistic that is
not significant.

**SEVERITY.** CRITICAL — a reported p ≈ 0.001 that is really p ≈ 0.25.

---

## Issue 5 — R² and the hypothesis test are computed against different benchmarks, and the R² benchmark is the weaker one

`study.py:28` sets `R2_BENCHMARK = "historical_mean"`; `registry.py:13` sets
`PRIMARY_BENCHMARK = "random_walk"`. So `R²_OS` is measured against the recursive mean while every
p-value is measured against zero.

Measured (`audit/scripts/` R² comparison): the zero forecast has **lower** squared error than the
recursive mean in all six asset–horizon cells — R² of the zero forecast against the mean benchmark
is +0.0005 (BTC 1), +0.0072 (BTC 7), +0.0006 (ETH 1), +0.0057 (ETH 7), +0.0153 (SOL 1), **+0.1023**
(SOL 7). The paper quotes R² against the easier of its two benchmarks and tests against the other.
Against the zero forecast, exactly one of eighteen settings has positive R² (BTC h=1 elastic net,
+0.0026).

The paper's defence of the CW/R² discrepancy — *"Clark–West asks whether the population mean squared
prediction error is lower"* — is true but does not address the fact that the two quantities are
computed against different reference forecasts.

**SEVERITY.** MAJOR. Reported effect sizes and reported significance are not commensurable.

---

## Issue 6 — HAC bandwidth h−1 is zero lags at h = 1

`newey_west_lrv(f, lags=max(0, horizon - 1))` means that at h = 1 no autocovariance is used at all.
Four of the paper's five rejections are at h = 1. The justification given (h-step errors are
MA(h−1)) holds for the forecast error under the null; it does not hold for the adjusted differential
`f_t = 2(y_t − ŷ_b,t)(ŷ_m,t − ŷ_b,t)` once drift is present, because the persistent component
`2μ(ŷ_m − ŷ_b)` is serially correlated.

Sensitivity (`audit/scripts/retest.py`, correct benchmark, three defensible bandwidths):

| bandwidth | raw 5 % rejections | survive BH(5 %) | survive Holm(5 %) |
|---|---|---|---|
| h−1 (paper) | 7/18 | 2 | 0 |
| Newey–West plug-in `4(n/100)^{2/9}` (≈8 at h=1) | 9/18 | 5 | 1 |

Both directions must be reported. The larger bandwidth *increases* significance here, so the
paper's choice is conservative on this axis at h = 1 — but the conclusion is bandwidth-dependent,
which is itself the finding, and the Monte Carlo shows the h = 7 cells remain oversized at either
bandwidth. Selecting a bandwidth by its p-value is not available as a repair.

Related: the headline "5 of 18" hangs on BTC h=7 ridge having p = **0.0500388**, which misses the
threshold by 4 × 10⁻⁵. Any perturbation of sample, bandwidth, or seed flips the count to 6.

**SEVERITY.** MAJOR.

---

## Issue 7 — The backtest trades at the close it needs in order to compute the signal

Features at bar t use `C_t, H_t, L_t, V_t` (§Features). The target is `log(C_{t+h}/C_t)` and
`backtest_strategy` (`backtest/strategy.py:67`) applies the position to exactly that return. Entry
is therefore at `C_t`, the same close that had to be observed to compute the feature.

Same forecasts, entry delayed by one bar (`audit/scripts/pt_and_exec.py`):

| setting | Sharpe (paper) | Sharpe (t+1 entry) | Δ |
|---|---|---|---|
| **ETH 1 gbm** (one of the paper's two "economic winners") | **0.91** | **0.21** | −0.70 |
| BTC 1 ridge | 0.51 | −0.23 | −0.74 |
| ETH 1 elastic_net | −0.11 | −0.61 | −0.50 |
| BTC 1 elastic_net | 0.33 | 0.06 | −0.27 |
| ETH 7 gbm (the other "economic winner") | 0.92 | 0.98 | +0.06 |
| all h = 7 settings | — | — | \|Δ\| ≤ 0.20 |

Every h = 1 result is materially execution-timing dependent; h = 7 is not. The paper's claim that
its economic conclusions are net of realistic costs does not extend to realistic timing.

**SEVERITY.** MAJOR for the h = 1 economic results.

---

## Issue 8 — The multiplicity family excludes most of the search

The family of 18 covers 3 models × 3 assets × 2 horizons for one statistic. Not in the family, and
each reported or used somewhere in the paper: the DM family (18), the PT family (18), the Sharpe
interval family (18), the DSR family, the h phase offsets (18 × h, reported as a range in
Table 2 and used as a robustness argument), the choice of train size 504 / test 63 / embargo 5, the
choice of 12 features, `alpha=1` for ridge and `1e-3` for elastic net, the six GBM hyperparameters,
17 bp costs, the sign rule, the HAC bandwidth, and the benchmark itself. `n_trials = 6` for the
deflated Sharpe (results.csv) counts the six models raced within one asset–horizon — including the
three benchmarks. The paper labels DSR an upper bound, which is honest but does not make the number
usable.

No pre-registration and no held-out confirmation sample exist, so the true trial count is not
recoverable after the fact. The only clean repair is a frozen forward confirmation sample.

**SEVERITY.** MAJOR.

---

## Issue 9 — Sample not frozen; forecasts not persisted

`config.py:59` — `end: str | None = None  # None => today`. The paper acknowledges this in
Limitations. In practice the committed parquet cache pins the sample at 2026-07-18 and the study
reproduces bit-for-bit from it, so the *audit* was reproducible; but the documented pipeline is not,
because a cache miss silently re-dates the study.

`study.py` keeps every OOS forecast in memory and writes only aggregates to `reports/results.csv`.
Re-deriving the forecasts was the first step of this audit (`audit/scripts/gen_forecasts.py`,
72 900 rows). Any independent re-analysis of the inference layer requires that file; it should be a
committed artifact.

Verified accurate: 154 tests collected; `n_boot=500` in `report.py:48` matches the paper's "500
resamples"; DM rejects in exactly 7 of 18 with the statistic favouring the benchmark; CW raw
rejections 5, BH survivors 2 (both p_BH = 0.0458), Holm survivors 0.

**SEVERITY.** MODERATE.

---

## Issue 10 — "Every high-Sharpe result is long-only exposure in disguise" is asserted, not measured

The paper offers visual and verbal support. Measured (`audit/scripts/exposure.py`, regression of
net strategy return on the buy-and-hold return on the same schedule and cost model, Newey–West
intercept t-statistic):

| setting | Sharpe | % long | β to buy-and-hold | annualised α | t(α) |
|---|---|---|---|---|---|
| ETH 1 gbm | 0.91 | 94 % | 0.76 | +0.236 | 1.07 |
| ETH 7 gbm | 0.92 | 91 % | 0.89 | +0.192 | 1.15 |
| BTC 7 gbm | 0.82 | 90 % | 0.82 | +0.097 | 0.66 |
| SOL 7 elastic_net | 0.60 | 56 % | 0.43 | +0.499 | 1.23 |
| historical_mean (all) | 0.76–0.84 | 100 % | 1.00 | ≈0.000 | ≤\|1.6\| |

The claim survives measurement — the two economic winners are 91–94 % long with β ≈ 0.8–0.9 and no
significant alpha — and it is *strengthened* by being measured. This is the one headline claim in
the paper that becomes better rather than worse under audit. It should be a table, not an adjective.

**SEVERITY.** MODERATE (presentation), and it is the seed of a real contribution.

---

## What survives

- The leakage discipline is genuine: purged/embargoed splits, in-fold scaling, the purged
  early-stopping holdout in `trees.py`, and the perturbation tests that enforce them. This is the
  best part of the repository and is above the standard of most applied work in this area.
- 154 tests collected; formulas match their documented references where checked; the Bartlett `1/n`
  normalisation argument in `newey_west_lrv` is correct.
- The exposure claim (Issue 10) holds under measurement.
- The reported counts (7 DM, 5 CW, 2 BH, 0 Holm) all reproduce exactly from the committed cache.

## What must change before any submission

Priority order, following the audit mandate:

1. Replace the zero benchmark with the recursively estimated mean everywhere; restate the estimand.
2. Delete the false simulation claim and the test that encodes it; replace with the estimated
   nested-null Monte Carlo in `audit/scripts/mc_null.py`.
3. Report calibrated size alongside every rejection count. "5 of 18 against 0.9 expected" becomes
   "7 of 18 against a measured 1.5 expected, and the count is bandwidth- and dependence-dependent".
4. Re-run all sign inference on non-overlapping subsamples across all phases.
5. Add dependence-robust inference (block bootstrap of the loss differential) as the primary test;
   the Monte Carlo shows the analytic HAC version is not trustworthy under realistic dependence.
6. Add the one-bar execution delay as the primary specification, not a robustness check.
7. Freeze the sample end date; commit the forecasts.
8. Build the multiplicity ledger honestly, and reserve a forward confirmation sample.
