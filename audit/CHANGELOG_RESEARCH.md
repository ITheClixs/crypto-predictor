# Research changelog

Substantive changes to claims, not to code style. Each entry records what was claimed, why it was
wrong, what replaced it, and which downstream claims moved.

Baseline audited: commit `83881c7`. Data pinned by `data/cache/` at 2026-07-18. All numbers below
reproduce from `audit/scripts/`.

---

## 2026-07-30 — Retraction pass

Scope: remove statements that are false or unsupported, and downgrade the conclusions that depended
on them. The study design, the model roster, the splits, the backtest and `reports/results.csv` are
unchanged; nothing was re-run to produce a more favourable number.

### R1 — The nesting claim (CRITICAL)

**Was:** "Every one of the three collapses to the random walk when its slopes are zero. They are
nested extensions of the benchmark." (`paper.tex` §Forecasters, `README.md` §3.4,
`evaluate/stats.py` module docstring.)

**Why wrong:** `Ridge` and `ElasticNet` are constructed without `fit_intercept=False`, and XGBoost
without an explicit `base_score`. Measured on BTC h=1 fold 0, both linear intercepts equal the
training-window mean of the target to machine precision (ratio 1.000000); the booster's fitted base
score is +1.975e-3 against a training mean of +1.713e-3. The slopes-zero restriction is the
historical-mean forecaster, never the zero forecast.

**Now:** the three sources state the intercept behaviour and identify the zero forecast as the
further restriction that the intercept is also zero, i.e. a joint martingale-difference null.

**Downstream:** every Clark–West p-value in the paper is against that joint null, not against a
feature-predictability null. See R3.

**Pinned by:** `tests/test_models_ml.py::test_zeroing_the_slopes_returns_the_training_mean_not_zero`.

### R2 — The validating simulation (CRITICAL)

**Was:** "We verify the bias by simulation rather than asserting it. Over 200 replications in which
the larger model forecasts pure noise, so that the null holds by construction, the mean
Diebold–Mariano statistic is positive and it rejects in favour of the benchmark far above the
nominal rate, while Clark–West remains centred on zero and holds its size." (`paper.tex` §Clark–West,
`README.md` §3.6.3, `tests/test_stats.py:103`, and the Reproducibility paragraph.)

**Why wrong:** three separate errors in one paragraph.

1. The null is false by construction, not true. With `y ~ N(0, σ_y)` and an independent forecast
   `u ~ N(0, σ_u)` against a zero benchmark, population MSPEs are `σ_y²` and `σ_y² + σ_u²`. Over
   5 000 replications the mean sample loss differential is 1.000e-4, matching `σ_u² = 1e-4`. DM's
   75 % rejection rate is power against a false null, not size distortion.
2. Nothing is estimated. `u` is exogenous noise, so the estimation-noise term Clark–West exists to
   remove is absent and the design cannot measure size under a nested-*estimation* null.
3. Clark–West's centring is algebraic: `f_t = 2(y−b)(m−b)` reduces to `2 y_t u_t` when `b = 0`, and
   `E[2yu] = 0` for any independent `u`. Numerically `−9.7e-7 ± 1.3e-6` over 200 000 draws.

**Now:** the claim is deleted from both manuscripts and replaced by a measured size table (R3). The
test is rewritten as `test_exogenous_noise_forecast_is_genuinely_worse_and_dm_detects_it`, asserting
only what the construction supports, and its docstring records the retracted claim. The assertion
`dm_rejects > 0.25`, which encoded the misreading, is gone.

### R3 — Size of the test, and the "0.9 expected" comparison (CRITICAL)

**Was:** "Eighteen settings tested at the 5 % level produce 0.05 × 18 = 0.9 rejections from noise
alone", supporting "5 of 18 reject … the evidence for predictability is real but marginal".

**Why wrong:** the arithmetic assumes the test is correctly sized. It is not.

**New experiment** (`audit/scripts/mc_null.py`, `mc_null_gbm.py`): resample the daily log-return
series into a synthetic price path, so future returns are independent of every past-information
feature by construction while drift, tails, feature persistence, sample length, the purged
walk-forward geometry and the actual estimators are retained, and forecasts are genuinely estimated
at every origin. 400 replications per cell, 150 for the booster.

Rejection at nominal 5 %:

| statistic | h=1 iid | h=1 block-21 | h=7 iid | h=7 block-21 |
|---|---|---|---|---|
| CW vs zero, ridge / elastic net / GBM | 14.2 / 22.0 / 22.0 % | 53.8 / 43.0 / — | 23.5 / 26.2 / — | 28.2 / 27.8 / — |
| CW vs recursive mean, ridge / elastic net / GBM | 6.2 / 6.2 / 5.3 % | 42.8 / 28.2 / — | 10.2 / 9.8 / — | 11.2 / 10.8 / — |
| CW vs zero, drift-only model | 27.3 % | 27.0 % | 39.0 % | 34.0 % |

iid columns measure size. Block-21 columns preserve 21-day dependence blocks and may carry genuine
within-block predictability, so they bound dependence sensitivity rather than measuring size.

**Now:** the expected count is stated as 2.6–4.0 against the zero benchmark and ≈1.5 against the
recursive mean, and both manuscripts state that the multiplicity adjustments do not control the
error rates they name when applied to p-values from an oversized test.

**Also new, and positive:** Clark–West against the recursive mean is well-sized for early-stopped
XGBoost (5.3 %), answering the open question of whether greedy splits, subsampling and a
data-dependent tree count break the approximation. Calibration evidence for this estimator and this
null, not a theorem.

### R4 — Which settings reject (CRITICAL, consequence of R1)

**Was:** "Four of the five are BTC and four are at the one-day horizon. That concentration is
consistent with a weak short-horizon effect in the most liquid asset."

**Why wrong:** an artifact of the benchmark. Re-testing against the recursive mean at the same
bandwidth (`audit/scripts/retest.py`): raw count 5/18 → 7/18, and membership changes. BTC h=1
survives (p = 0.004). BTC h=7 drops out (0.050 → 0.086, 0.039 → 0.061). SOL, reported as having no
signal at either horizon, becomes the strongest rejecter: SOL gbm h=1 0.207 → 0.031, SOL ridge h=7
0.923 → 0.022, SOL elastic net h=7 0.921 → 0.011.

**Now:** both manuscripts carry the comparison table and state that the geography reverses, that
under calibrated size neither pattern is distinguishable from noise, and that the surviving counts
are additionally bandwidth-dependent (BH survivors 2 at h−1 lags, 5 at the Newey–West plug-in
bandwidth; Holm survivors 0 and 1).

Related: the headline "5 of 18" excluded BTC h=7 ridge at p = 0.0500388, four parts in 10⁵ from the
threshold. Both manuscripts now say so.

### R5 — The sign-timing results (CRITICAL)

**Was:** "Two are significantly *anti*-predictive: ETH ridge and ETH elastic net at h=7 give
S = −3.29 and S = −3.33, both p ≈ 0.001", plus a passage arguing that reporting these as bare
p-values would mislead.

**Why wrong:** `pesaran_timmermann` divides every variance term by `n`, and `study.py` calls it on
all 2 174 rows of a 7-day forecast whose labels overlap six times. The statistic inflates by ≈√h.
Figure 3 in the same document draws its coin-flip band at `n/h`, so the table and the figure
disagreed by √7.

**Now:** recomputed on non-overlapping subsamples averaged over all h phases: ETH ridge −1.27
(p = 0.268), ETH elastic net −1.27 (p = 0.243), SOL ridge −0.66 (0.597), BTC ridge +0.54 (0.592). The
conclusion — no sign-timing skill in either direction — is unchanged; the evidence for its most
striking part was invalid. The rhetorical passage built on p = 0.001 is deleted.

**Pinned by:** `tests/test_stats.py::test_pesaran_timmermann_is_inflated_by_overlapping_labels`, plus
a docstring warning that the caller must pass non-overlapping observations.

### R6 — Execution timing (MAJOR)

**Was:** economic results presented as net of realistic costs, with no timing caveat.

**Why incomplete:** features at bar t use the completed close `C_t`; the backtest enters at that same
`C_t`. Delaying entry one bar with forecasts unchanged (`audit/scripts/pt_and_exec.py`): ETH gbm h=1
0.91 → 0.21, BTC ridge h=1 0.51 → −0.23, ETH elastic net h=1 −0.11 → −0.61, BTC elastic net h=1
0.33 → 0.06. No h=7 setting moves by more than 0.20.

**Now:** both manuscripts carry the delay table and state that the h=1 economic results are
properties of the execution convention.

### R7 — The exposure claim (MODERATE — strengthened, not retracted)

**Was:** "every high-Sharpe result in the study is long-only exposure in disguise", supported by
visual comparison.

**Now:** measured (`audit/scripts/exposure.py`). ETH gbm h=1: 94 % long, β = 0.76 to buy-and-hold,
α t = 1.07. ETH gbm h=7: 91 % long, β = 0.89, α t = 1.15. BTC gbm h=7: 90 %, β = 0.82, t = 0.66.
Historical mean: 100 % long, β = 1.00, α ≈ 0. The claim survives measurement. "In disguise" is
replaced by the table.

### R8 — Benchmark inconsistency between effect size and significance (MAJOR)

**Was:** `R²_OS` quoted against the recursive mean (`study.py:28`), p-values computed against the
zero forecast (`registry.py:13`), with no acknowledgement.

**Also:** the zero forecast has lower squared error than the recursive mean in all six cells, by
`R²` of +0.0005 (BTC h=1) to +0.1023 (SOL h=7). Against the zero forecast, exactly one of 18 settings
has positive `R²`: BTC elastic net h=1, +0.0026.

**Now:** stated in §3.6.1 / §Out-of-sample R² of both documents, with the zero-forecast comparison
given as the stricter number to read alongside a p-value computed against zero. The paper's earlier
defence of the CW/R² gap ("Clark–West asks about population MSPE") is retained but no longer offered
as the whole explanation.

### R9 — Efficiency and power claims (MAJOR)

**Was:** "The result is consistent with weak-form efficiency at daily frequency … net of realistic
costs", and "the full apparatus that would have detected either".

**Why wrong:** an oversized test cannot bound predictability in either direction, and no minimum
detectable effect is computed anywhere in the study.

**Now:** both documents state explicitly that the design is not powered to exclude an economically
relevant effect and that no such claim is made. The abstract's "faint, fragile statistical signal" is
withdrawn.

### R10 — Reproducibility statements (MODERATE)

**Was:** "All results … generated by a single command … with a test suite of 154 tests covering …
the simulation demonstrating the Diebold–Mariano bias."

**Now:** the reference to the retracted simulation is removed, and the paragraph states that the
suite could not have detected the benchmark misspecification: every formula was implemented
correctly and every test passed. `StudyConfig.end` defaults to `None` → today, and
`DEFAULT_CACHE_DIR` is a relative path, so running from a different working directory misses the
cache and silently re-dates the study — reproduced by accident while writing `audit/README.md`
(2 186 forecasts per setting from the repo root, 2 198 from `audit/`). Both documents now say so.

Test count 154 → 157; coverage badge 96 % → 97 %.

---

## Unchanged and verified

`reports/results.csv` was not modified. Reproduced bit-for-bit from the committed cache: DM rejects
in 7 of 18 with the sign favouring the benchmark; CW raw rejections 5 of 18; BH survivors 2, both
p = 0.0458; Holm survivors 0; `n_boot = 500`; R²_OS −0.0079 and +0.0031. No transcription errors
between `reports/results.csv`, `paper/tables/*.tex` and the manuscript text.

## Not done

Not attempted in this pass, and required before any submission:

- Literature review and the novelty gate (`RESEARCH_PIVOT_MEMO.md`, final section). Novelty of the
  benchmark result is **unverified**.
- Bootstrap inference on `f_t`, without which no definitive rejection count exists at h=7 or under
  realistic dependence.
- Switching `PRIMARY_BENCHMARK` in `src/` — both benchmarks are now reported in prose, but the
  pipeline still computes p-values against zero only.
- Pinning `StudyConfig.end`, committing a run manifest with data hashes, persisting forecasts as a
  first-class artifact.
- Minimum detectable effect; cost curve; multiplicity ledger over the full search; frozen forward
  confirmation sample.
