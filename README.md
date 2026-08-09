# Betting Against the Benchmark

**Drift-robust, anytime-valid certificates of out-of-sample predictive ability — with a
reference implementation (`alphacert`) and a six-year cryptocurrency case study**

[![ci](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml/badge.svg)](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![tests](https://img.shields.io/badge/tests-220-informational)
![coverage](https://img.shields.io/badge/coverage-95%25-informational)
![license](https://img.shields.io/badge/license-noncommercial-orange)

---

## Abstract

Tests of out-of-sample return predictability treat the asset's drift as a nuisance and
dispose of it by **estimating** it. That is where they break.

A model with an unpenalised intercept does not reduce to the zero forecast when its slopes
vanish — it reduces to the training-window mean. Racing it against zero therefore tests *no
drift* **and** *no conditional predictability* jointly. In closed form (Proposition 1):

$$\mathbb{E}[\mathrm{CW}_0] \approx \frac{\sqrt{n} S^2}{\sqrt{S^2 + (1+p)/k}} \longrightarrow \sqrt{n} S \quad \text{when } k S^2 \gg 1+p,$$

for per-period Sharpe ratio $S$, estimation window $k$ and $p$ estimated coefficients — the
statistic **converges to the $t$-statistic of the asset's mean return**. Measured: at
Bitcoin's realised drift a model with *no features at all* is declared significantly
predictive in **43%** of samples at a nominal 5%, and **81%** at an annualised Sharpe ratio
of 2.0. Swapping in the recursive mean removes the leading term and leaves the benchmark's
own estimation error behind.

### The instrument

This repository eliminates the drift instead. A **certificate** is a non-negative wealth
process: one bettor stakes on the signal leading the outcome, a second hedges every candidate
drift two-sidedly, and the certificate is the infimum over the nuisance of their average.
Because an infimum is at most the value at the truth, and the value at the truth is a
martingale, Ville's inequality gives

$$P(\exists t : \mathcal{E}_t \ge 1/\alpha) \le \alpha$$

in **finite samples** and **uniformly in time**. No bandwidth, no long-run variance, no
bootstrap, no asymptotics, no refitting — and it is indifferent to what produced the
forecasts, so regularised, early-stopped and black-box learners are covered by the same
theorem that covers OLS.

| | Clark–West | Certificate |
|---|---|---|
| Rejection rate at Sharpe 0.0 / 2.0, no predictability | 5.2% / **81%** (vs zero) | 0.3% / 0.8% |
| Power at IR = 2, size-matched | 0.85 | 0.76 |
| Calibration needed | 2,000 pipeline refits, hours | none — validity is a theorem |
| Valid if you look every day | no | yes, at every stopping time |
| Multiplicity over an 18-cell grid | joint bootstrap | average the e-values |
| Overlapping $h$-step labels | needs a variance for the average | $h$ phase e-values, averaged |
| Units of the statistic | dimensionless | **profit and loss** |

Three consequences a $t$-statistic cannot give:

1. **Monitorable.** Run it forward on a live strategy, look every day, no correction. It also
   yields an anytime-valid **ceiling** on incremental value — the number that lets you
   *retire* a research line rather than merely fail to confirm it.
2. **Denominated in money.** $\log \mathcal{E}_T$ is the P&L of an explicit drift-hedged
   strategy, so rejecting at 5% *is* a twentyfold multiplication of capital that was credited
   with none of the market's return. Statistical and economic significance stop being two
   numbers that might disagree.
3. **Composable.** E-values average under *arbitrary* dependence, so a research grid needs no
   joint bootstrap — and averaging the $h$ phase certificates of an overlapping forecast is
   literally the staggered $h$-vintage portfolio, so the valid combination and the
   implementable strategy are the same object.

### The design law, and its price

Evidence is log wealth; log wealth grows at half the squared information ratio. So the median
time to certify an annualised information ratio IR is

$$T^\ast = \frac{2\ln(1/\alpha)}{\mathrm{IR}^2} \text{ years} \approx \frac{6}{\mathrm{IR}^2} \text{ at the 5 percent level},$$

and no anytime-valid test does better in order. **Six years of daily data — about what any
liquid cryptocurrency offers — cannot certify an information ratio below 1.59.**

We are equally explicit about what the guarantee costs, because that number is missing from
the e-value literature for this comparison.

| | years at IR = 1 |
|---|---|
| Certificate, median crossing, stake pre-committed | 7.4 |
| Certificate, median crossing, stake learned online | 16.1 |
| **Certificate at 80% power** | **11.8** |
| **Fixed-sample one-sided test at 80% power** | **6.2** |

**Matched on level and power, anytime-validity costs about a factor of 1.9 in sample size** —
scale-free, between 1.9 and 2.2 across conventional power levels. An earlier version of this
work put the figure at 3% by comparing the certificate's *median* crossing time with the
z-test's *80th percentile*; that was wrong and is corrected here.

Head to head against Clark–West, in the design where Clark–West is exactly correctly sized
(measured 5.5%), 400 replications:

| IR | CW | CW size-matched | Certificate | Certificate size-matched |
|---|---|---|---|---|
| 1.0 | 0.230 | 0.210 | 0.030 | 0.163 |
| 1.5 | 0.510 | 0.495 | 0.188 | 0.390 |
| 2.0 | 0.853 | 0.850 | 0.480 | 0.758 |
| 3.0 | 1.000 | 1.000 | 0.988 | 0.998 |

Two things separate out. **The construction is nearly as efficient as the incumbent** — at
matched size, 0.758 against 0.850 at IR = 2. **What costs is anytime-validity itself**: the
certificate's measured size at its nominal threshold is 0.5–0.7% against a nominal 5%, because
Ville's inequality is tight only for a process that jumps straight to the threshold.

Nothing here claims the certificate dominates. Where the benchmark is defensible, the sample
is fixed in advance and one look is enough, **use Clark–West**. The certificate is for what
that cannot do: monitoring a live strategy, a contested or estimated benchmark, a dependent
grid, a pipeline outside any asymptotic theory.

### Positive control: it does say yes

A method whose only real-data demonstration finds nothing is unfalsified. Same instrument,
same assets, same window, pointed at realised volatility — which is genuinely predictable:

| | e-value | certified in |
|---|---|---|
| BTC | 3.1×10⁷ | 365 days |
| ETH | 4.2×10⁹ | 416 days |
| SOL | 7.5×10⁸ | 148 days |

### The evidence

Three assets, two horizons, three estimators, purged and embargoed walk-forward validation
over 2020–2026, programmatic leakage tests, transaction costs, feasible one-bar entry delay,
staggered portfolios, exposure regressions.

- **Nothing is certified.** Largest e-value 3.05 against the 20 required; grid-level e-value
  **1.43**; e-BH selects nothing; the directional variant (which replaces an invalid
  Pesaran–Timmermann average) gives 1.07.
- **Implied information ratios run 0.00 to 0.69**, against a certifiable floor of 1.59.
- **Ceiling:** after six years, the incremental annualised information ratio of twelve
  standard technical features on these three assets is **between 0.63 and 2.55, every
  interval containing zero.**
- **Economically, nothing either.** Under the feasible specification no setting of eighteen
  has a Sharpe interval excluding zero, and none has a positive alpha to buy-and-hold
  distinguishable from zero. The instruments agree.

Clark–West rejects in 7 of 18 settings against the recursive mean, and a joint resampling of
the whole experiment puts P(N ≥ 7) at 0.0035 — though no single setting survives family-wise
control (smallest Romano–Wolf p = 0.052). This is not a contradiction: it is a $p$-value and
an e-value disagreeing about what six years can settle, and the design law says which one is
right. The reportable number is the ceiling, not a $p$-value on a statistic whose null was
never the question being asked.

### Three nulls, none of them the truth

A resampling null is only as good as the properties it keeps, so the calibration is repeated
under generators that fail in different ways.

| Generator | s.d. | exc. kurt. | skew | ρ₁(r) | ρ₁(\|r\|) | leverage |
|---|---|---|---|---|---|---|
| Real series (BTC) | 0.0323 | 18.78 | −1.05 | −0.054 | 0.154 | −0.071 |
| Block resample | 0.0323 | 17.77 | −0.99 | −0.046 | 0.150 | −0.066 |
| Sign-flipped | 0.0323 | 17.75 | −0.19 | 0.004 | 0.150 | −0.003 |
| GARCH(1,1), zero mean | 0.0325 | 0.63 | 0.02 | −0.001 | 0.141 | 0.004 |

Sign-flipping keeps volatility clustering exactly and destroys skewness, leverage and sign
dependence — so the block/sign-flip gap is **not** attributable to conditional mean dependence
alone, and we no longer claim it is. The GARCH null shares neither generator's artifacts and
gives the same answer: Clark–West against the recursive mean rejects 5.25% / 6.25% / 5.00%
across the three, the certificate 0.25% / 1.50% / 0.25%.

> The full audit, including reproduction scripts for every number above and a record of which
> earlier claims they overturn, is in [`audit/`](audit/README.md). Retracted claims are listed
> in the paper's appendix and in [`audit/CHANGELOG_RESEARCH.md`](audit/CHANGELOG_RESEARCH.md).

---

## Using `alphacert`

```python
import numpy as np
from alphacert import certify, certify_overlapping, e_bh, merge_average, value_ceiling
from alphacert import certifiable_ratio, detection_horizon

# signal_t must be computable strictly before outcome_t is realised. For a nested forecast
# comparison, use the model's forecast minus the intercept-only forecast from the same window.
signal = model_forecast - intercept_only_forecast
cert = certify(signal, outcome)                    # tanh payoff; conditional symmetry
cert.evalue                                        # e-value; >= 20 rejects at 5%
cert.p_value                                       # anytime-valid p-value
cert.stopping_time(0.05)                           # when the evidence arrived, if it did
cert.log_wealth                                    # the drift-hedged strategy's P&L, in nats

# Overlapping h-step labels: h phase certificates, averaged. Also the staggered portfolio.
evalue, phases = certify_overlapping(signal, outcome, horizon=7)

# A whole research grid, corrected under arbitrary dependence. No bootstrap.
merge_average([c.evalue for c in grid])            # valid test of the global null
e_bh([c.evalue for c in grid], alpha=0.05)         # FDR control, arbitrary dependence

# Decide before you look; it is worth a factor of two in data.
detection_horizon(1.0, kelly_known=True)           # 7.4 years
certifiable_ratio(6.0)                             # 1.59 -- what this sample can speak to
cert = certify(signal, outcome, design_ratio=1.0)  # pre-committed stake

# Retire a research line with a number rather than a shrug.
bound = value_ceiling(signal, outcome, alpha=0.05)
bound.ratio_ceiling(outcome.std())                 # annualised IR the features could still have
```

**Assumptions.** `payoff="identity"` (the default) needs only a martingale-difference null
plus an a-priori envelope on $|y_t|$. `"tanh"` and `"sign"` are bounded and so are not
throttled by that envelope, but they need the outcome to be *conditionally symmetric* about
its drift — which implies **zero unconditional skewness**, and is therefore refutable from the
marginal distribution alone. Daily crypto refutes it (skew −1.05 on BTC, p < 10⁻⁷⁰), which is
why it is not the default. Check before using them.

**The one numerical trap.** The infimum over the drift is evaluated on a grid, and the grid
must resolve the drift to better than $\sigma/\sqrt{n}$ — a coarser grid lifts the numerical
infimum above the true one and the process stops being conservative. The default spacing is
`1e-4`, `recommended_resolution()` computes a safe one, and the test suite records the failure
mode of a deliberately coarse grid so it is documented rather than latent.

---

## 1. Introduction

Retail-facing "crypto price prediction" projects report high accuracy with striking
regularity. The reported accuracy is usually an artifact of one of four errors:

1. **Target leakage.** Predicting a quantity that is a function of contemporaneous or
   future information, so the model is fitting an identity rather than forecasting.
2. **Preprocessing leakage.** Fitting a scaler, an imputer, or a feature selector on the
   whole sample before splitting, which leaks the test distribution into training.
3. **No benchmark.** Reporting $R^2$ or MAPE on a price series, where a naive
   random-walk forecast attains near-perfect scores because prices are close to a
   martingale. A model that appears to explain 99% of price variance may explain none of
   the return variance, which is the only part anyone can trade.
4. **Silent multiple testing.** Trying many model, asset, and horizon combinations and
   reporting the best one without adjusting the significance threshold.

Preprocessing leakage is impossible by design here (standardization is a pipeline step fit
inside each fold). Target leakage is prevented by a test that perturbs future bars and
asserts no feature at or before $t$ changes. Multiple testing is reported over an explicit
family. **Benchmarking is the one item on this list the first version of this study got
wrong.**

Diebold–Mariano is indeed invalid for the comparison this literature makes, and Clark–West
is the standard correction. The question that goes unasked is which null the corrected test
*states*. A regression with a fitted intercept does not reduce to the zero-return forecast
when its slopes are set to zero — it reduces to the training-window mean. Testing against
zero therefore tests a joint hypothesis: no drift **and** no conditional predictability.
Where the drift is large, as it is for these assets over this window, that difference is not
academic: Section 3.6.3 derives the contaminating term, Section 3.6.4 measures the resulting
rejection rate, and Section 4.2 shows that repairing the benchmark changes *which* settings
reject rather than how many.

**What is new here, and what is not.** The diagnosis is not a discovery. Clark & West (2006)
state their null as a *zero-mean* martingale difference, so a drift term entering the
statistic is the test working as specified; the recursive mean has been the standard
out-of-sample benchmark since Goyal & Welch (2008) and Campbell & Thompson (2008); Moosa &
Burns (2016) argue the drift-versus-no-drift benchmark question directly; Pincheira, Hardy &
Muñoz (2021) documented Clark–West's long-horizon size distortion and proposed the
asymptotically normal Wild Clark–West replacement; and Magner & Hardy (2022) already apply
that replacement to 13 cryptocurrencies against *both* a zero and a constant-forecast
benchmark. What this study adds is measurement rather than prescription: a calibrated size
for the whole applied pipeline including an early-stopped booster, for which no prior
calibration evidence was found; an implementation and measurement of Wild Clark–West showing
that it does not help in this regime and cannot in principle repair a benchmark that encodes
the wrong hypothesis; and a worked account of four routine choices that each independently
reversed a headline finding of the first version. See
[`audit/LITERATURE_AND_NOVELTY_MATRIX.csv`](audit/LITERATURE_AND_NOVELTY_MATRIX.csv).

The first version of this README reported "five of 18 settings reject, against 0.9 expected
from a search that size." That comparison assumed the test was correctly sized for the
hypothesis we had in mind. It was not, and the corrected expectation against that benchmark
is about 4. The earlier claim is stated here rather than quietly removed, because the error
is easy to make, invisible in code review, and undetected by a test suite that checks every
formula against hand-computed values — as this one does.

---

## 2. Data

Daily OHLCV bars from Yahoo Finance, adjusted, for the USD pairs of three assets.

| Asset | Bars | First | Last | Labeled rows ($h{=}1$) | Folds |
| --- | ---: | --- | --- | ---: | ---: |
| BTC | 2,756 | 2019-01-01 | 2026-07-18 | 2,696 | 35 |
| ETH | 2,756 | 2019-01-01 | 2026-07-18 | 2,696 | 35 |
| SOL | 2,291 | 2020-04-10 | 2026-07-18 | 2,231 | 28 |

Cryptocurrency markets trade every calendar day, so the bar index is contiguous with no
weekend or holiday gaps; this was verified rather than assumed, and the split logic in
Section 3.3 is proved conservative under gaps regardless.

The evaluated out-of-sample window is **2020-07-23 to 2026-07-17**, shorter than the data
window because the first fold consumes 504 bars of training history. Per-setting
out-of-sample counts are 2,186 forecasts for BTC and ETH and 1,721 for SOL at $h=1$.

Loading is cached to Parquet keyed by request parameters, so a rerun is offline and the
network is touched once per asset.

**Asset selection is not survivorship-free.** BTC, ETH, and SOL were chosen in 2026 with
the knowledge that they survived. Hundreds of 2019-era tokens did not. This biases the
study toward finding profitable long exposure, which strengthens rather than weakens a
negative result on *predictability*, but would invalidate any cross-sectional claim.
Section 6 returns to this.

---

## 3. Method

### 3.1 Features

Let $C_t, H_t, L_t, V_t$ denote close, high, low, and volume at bar $t$. Every feature is
a function of information available at or before $t$. All twelve are scale-free, so a
single model can pool across assets trading at very different price levels.

Cumulative log returns over $k \in \lbrace 1, 5, 10, 21 \rbrace$:

$$r_t^{(k)} = \log \frac{C_t}{C_{t-k}}$$

Realized volatility over $w \in \lbrace 10, 21 \rbrace$, the rolling standard deviation of daily log
returns:

$$\sigma_t^{(w)} = \mathrm{sd}\left(r_{t-w+1}^{(1)}, \dots, r_t^{(1)}\right)$$

Wilder's RSI over 14 bars, recentred to roughly $[-1, 1]$. With
$U_t = \max(C_t - C_{t-1}, 0)$ and $D_t = \max(C_{t-1} - C_t, 0)$ smoothed by an
exponential moving average with $\alpha = 1/14$:

$$\mathrm{RSI}_t = 100 - \frac{100}{1 + \bar{U}_t / \bar{D}_t}, \qquad \text{feature} = \frac{\mathrm{RSI}_t - 50}{50}$$

MACD histogram, normalized by price so it is comparable across assets, where
$\mathrm{EMA}_n$ is the $n$-span exponential moving average:

$$M_t = \mathrm{EMA}_{12}(C)_t - \mathrm{EMA}_{26}(C)_t, \qquad S_t = \mathrm{EMA}_9(M)_t, \qquad \mathrm{macd}_t = \frac{M_t - S_t}{C_t}$$

Moving-average ratio and distance, with $\mathrm{SMA}_n$ the $n$-bar simple moving
average:

$$\log \frac{\mathrm{SMA}_7(C)_t}{\mathrm{SMA}_{21}(C)_t}, \qquad \log \frac{C_t}{\mathrm{SMA}_{50}(C)_t}$$

Mean normalized range over 14 bars, and a 21-bar volume z-score on $\log(1 + V_t)$:

$$\mathrm{range}_t = \frac{1}{14}\sum_{i=0}^{13} \frac{H_{t-i} - L_{t-i}}{C_{t-i}}, \qquad z_t = \frac{\log(1+V_t) - \mu_t^{(21)}}{s_t^{(21)}}$$

The longest warm-up is 50 bars, and rows with any undefined feature are dropped.

**Verification.** `tests/test_features.py::test_no_lookahead` perturbs bars strictly after
$t$ and asserts that no feature value at or before $t$ changes. This is a stronger check
than reading the code, because it would catch an accidental centred window, a negative
shift, or a non-causal smoother.

### 3.2 Target

The label attached to a decision made at $t$ is the return realized over $(t, t+h]$:

$$y_t^{(h)} = \log \frac{C_{t+h}}{C_t}$$

The final $h$ rows have no label. They are retained with features and a missing target,
because that is exactly the state a live forecaster occupies on the most recent bar, and
they are excluded from fitting and scoring.

### 3.3 Purged, embargoed walk-forward cross-validation

Ordinary $k$-fold shuffles time and trains on the future. A plain chronological split is
not sufficient either. Because $y_t^{(h)}$ resolves at $t+h$, a training row close to the
test block carries a label whose realization overlaps the test window. Following
López de Prado (2018, ch. 7), the $h$ rows before each test block are *purged* and a
further *embargo* of $e$ bars is imposed. Writing $\mathcal{T}_k$ and $\mathcal{V}_k$ for
the train and test index sets of fold $k$, every fold satisfies

$$\max(\mathcal{T}_k) + h + e < \min(\mathcal{V}_k)$$

Test blocks tile forward without overlap. The study uses an expanding window with
$|\mathcal{T}| \geq 504$ bars, $|\mathcal{V}| = 63$ bars, and $e = 5$, giving 35 folds for
BTC and ETH and 28 for SOL.

![Figure 1](reports/figures/fig1_design.png)

**Figure 1.** The design, drawn from the splits the study actually runs rather than
illustrated separately. Panel (b) zooms on one boundary, where the purge and the embargo
are individually visible.

Purging is applied by row *position*, while the leak it prevents is a *calendar* one. If
bars were missing, would position-based purging still be enough? Yes, and conservatively
so: gaps only stretch the timeline, so $h$ positions always span at least $h$ calendar
days. `tests/test_splits.py` punches 25% of bars out of a synthetic index and asserts the
calendar guarantee directly.

Two further leakage controls are worth naming because they are commonly missed:

- **Scalers are fit inside the fold.** Standardization is a pipeline step, so it sees
  training rows only. Fitting a scaler on the full sample is the single most common
  preprocessing leak in this genre.
- **The booster purges its own validation slice.** Gradient boosting early-stops on a
  temporal holdout carved from the end of the training window. Without a gap, the last
  $h$ rows of the fitting slice carry labels realized inside that holdout, so early
  stopping is tuned on partly observed outcomes. The same $h$-bar purge is applied one
  level down. The test for this corrupts exactly those rows and asserts the fitted model
  is unchanged, which a leaky implementation would fail.

### 3.4 Forecasters

Three benchmarks and three machine-learning models, all refit from scratch on every fold.

| Name | Definition |
| --- | --- |
| `random_walk` | $\hat{y}_t = 0$. The martingale null. |
| `historical_mean` | $\hat{y}_t = \bar{y}_{\mathcal{T}}$, the training-window mean. Unconditional drift. |
| `ar1` | OLS of $y^{(h)}$ on $r^{(1)}$, the simplest conditional model. |
| `ridge` | $\ell_2$-penalized linear, $\alpha = 1$, on standardized features. |
| `elastic_net` | $\ell_1/\ell_2$ mix, $\alpha = 10^{-3}$, $\rho = 0.5$. |
| `gbm` | XGBoost, depth 3, $\eta = 0.03$, subsample 0.8, $\lambda = 1$, early stopping on a purged temporal holdout. |

Ridge solves

$$\hat{\beta} = \arg\min_{\beta} \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_2^2$$

and elastic net

$$\hat{\beta} = \arg\min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \alpha \rho \lVert \beta \rVert_1 + \frac{\alpha(1-\rho)}{2} \lVert \beta \rVert_2^2$$

The models are deliberately shrunk. The signal-to-noise ratio in return prediction is
small enough that an unregularized learner will fit noise, and a study whose null result
came from an overfit model would be uninformative.

All three fit an intercept: the linear models fit an unpenalized one on standardized
features, and XGBoost estimates its own `base_score` from the training labels. Setting the
slopes to zero therefore returns the **training-window mean**, not zero — measured on BTC at
$h=1$, the ridge and elastic-net intercepts equal $\bar{y}_{\mathcal{T}}$ to machine
precision. The forecaster these models nest is the historical mean; the random walk is the
further restriction that the intercept is also zero. Section 3.6.3 shows what that
distinction does to the test.

### 3.5 Trading rule and costs

Forecasts become unit positions on a non-overlapping schedule: one decision every $h$
bars, held to the next. Sampling every $h$-th bar avoids counting the same $h$-day return
$h$ times.

$$w_t = \mathrm{sign}(\hat{y}_t) \in \lbrace -1, 0, +1 \rbrace$$

Costs are charged on turnover, so reversing a position pays for two sides:

$$\tau_t = |w_t - w_{t-1}|, \qquad \pi_t = w_t \left(e^{y_t^{(h)}} - 1\right) - \tau_t \cdot c$$

with $c = 17$ bps per side (10 bps taker fee, 5 bps slippage, 2 bps half-spread). Equity
compounds as $E_T = \prod_{t \le T}(1 + \pi_t)$, starting from 1.

Because decisions are taken every $h$ bars, the Sharpe ratio annualizes at $365/h$ and
not at 365. Getting this wrong inflates a 7-day strategy's Sharpe by a factor of
$\sqrt{7} \approx 2.6$.

The schedule has $h$ possible start offsets and nothing distinguishes them. Every strategy
is therefore run on all $h$, and the spread is reported (Section 4.6).

An **always-long reference** is computed on the same schedule with the same costs, so that
a forecaster whose only achievement is holding the asset can be recognized as such.

### 3.6 Inference

#### 3.6.1 Out-of-sample $R^2$

Reported in the Campbell–Thompson (2008) form, against a genuine ex-ante benchmark
forecast rather than the realized mean of the evaluation window:

$$R^2_{OS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t (y_t - \hat{y}^{b}_t)^2}$$

where $\hat{y}^{b}$ is the recursively estimated historical mean. This benchmark scores
exactly $0.0000$ against itself, which is a built-in check that the metric is wired
correctly. The alternative denominator $\sum_t (y_t - \bar{y})^2$ uses the realized mean
of the test window, which is not knowable in advance and flatters the benchmark; it is
computed as a cross-check and carried in the CSV as `r2_vs_sample_mean`, but it is not the
number quoted.

#### 3.6.2 Diebold–Mariano, and why it is the wrong test here

For loss differential $d_t = L(e^m_t) - L(e^b_t)$ with squared-error loss, the
Diebold–Mariano (1995) statistic is

$$\mathrm{DM} = \frac{\bar{d}}{\sqrt{\hat{V}/n}}$$

where $\hat{V}$ is a Newey–West long-run variance with $h-1$ Bartlett lags, appropriate
because $h$-step forecast errors follow an MA($h-1$):

$$\hat{V} = \hat{\gamma}_0 + 2\sum_{k=1}^{h-1}\left(1 - \frac{k}{h}\right)\hat{\gamma}_k, \qquad \hat{\gamma}_k = \frac{1}{n}\sum_{t=k+1}^{n}(d_t - \bar{d})(d_{t-k} - \bar{d})$$

Note the $1/n$ normalization on every autocovariance, including the lagged ones. The
Bartlett weights guarantee $\hat{V} \ge 0$ only under that normalization; the more
intuitive $1/(n-k)$ can drive a weighted sum negative in small samples.

The Harvey–Leybourne–Newbold (1997) small-sample correction and a $t_{n-1}$ reference
distribution complete the test:

$$\mathrm{DM}^{\ast} = \mathrm{DM}\sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$

**A negative statistic favours the model.** The test is two-sided, because the null is
equal accuracy and a model can be significantly worse.

The difficulty is that Diebold–Mariano assumes the two forecasts are non-nested. Under a
nested null the larger model must still *estimate* coefficients that are truly zero, and
the resulting estimation noise inflates its sample mean squared prediction error even
though its population MSPE is identical. Diebold–Mariano reads that noise as evidence for
the benchmark and is undersized as a test of predictability.

The bandwidth deserves a warning. The MA($h-1$) argument justifies $h-1$ lags for the
*forecast error*; it does not carry over to the Clark–West adjusted differential once a
drift component enters (below). At $h=1$ it is **zero** lags, so no autocovariance enters
at all, and four of the five settings that reject are at $h=1$. The Newey–West plug-in
bandwidth $\lfloor 4(n/100)^{2/9}\rfloor$ = 8 lags gives *smaller* p-values here, so $h-1$
is the conservative choice at $h=1$ — but the count of surviving settings changes with the
bandwidth, and neither bandwidth is adequate at $h=7$.

#### 3.6.3 Clark–West

Clark and West (2007) subtract the estimation-noise term explicitly. With $\hat{y}^b$ the
nested benchmark and $\hat{y}^m$ the larger model,

$$\hat{f}_t = \left(y_t - \hat{y}^{b}_t\right)^2 - \left[\left(y_t - \hat{y}^{m}_t\right)^2 - \left(\hat{y}^{b}_t - \hat{y}^{m}_t\right)^2\right]$$

and the statistic $\bar{f}/\sqrt{\hat{V}_f/n}$ is compared against a standard normal,
one-sided. **A positive statistic favours the model.** The sign convention is opposite to
Diebold–Mariano, which is a good reason to never report either p-value without its
statistic.

The right-hand side has a closed form that matters more than the left:

$$\hat{f}_t = 2\left(y_t - \hat{y}^{b}_t\right)\left(\hat{y}^{m}_t - \hat{y}^{b}_t\right)$$

So Clark–West is a *t*-test of whether the model's deviation **from the benchmark** covaries
with the outcome's. **The benchmark defines the hypothesis, not just the baseline.** With
$\hat{y}^b = 0$ this becomes $2 y_t \hat{y}^m_t$, whose expectation is

$$\mathbb{E}[\hat{f}_t] = 2\left(\mathbb{E}[y] \cdot \mathbb{E}[\hat{y}^m] + \mathrm{Cov}(y, \hat{y}^m)\right)$$

Only the second term is predictability. The first is drift times the average forecast, and
through the fitted intercept the models are always on average long. BTC's realized drift
here is +0.00102/day, 37% annualized. Measured drift share of the numerator: 9% at BTC
$h=1$, 33% for the booster at $h=1$, 41–61% across BTC $h=7$. The clearest evidence is that
the **historical-mean forecaster, which uses no features at all, scores CW = +1.03 against
the zero benchmark on BTC at $h=1$**.

#### 3.6.4 Calibrating the test instead of assuming it

An earlier version of this README claimed the nesting bias was "demonstrated rather than
asserted" by a simulation drawing $y \sim N(0, 0.03)$ and an independent
$u \sim N(0, 0.01)$ as the larger model's forecast. That claim was wrong:

$$\mathrm{MSPE}_{\text{bench}} = \sigma_y^2, \qquad \mathrm{MSPE}_{\text{model}} = \sigma_y^2 + \sigma_u^2$$

The population MSPEs differ by $\sigma_u^2$, so the null of equal accuracy is **false** by
construction and DM's high rejection rate is *power*, not size distortion. Nor is anything
estimated in that setup, so the estimation-noise term Clark–West removes is absent. And CW's
apparent centring is algebra: $\hat{f}_t = 2 y_t u_t$ and $\mathbb{E}[2yu] = 0$ for *any*
independent $u$, however bad a forecast.

The replacement (`audit/scripts/mc_null_wcw.py`) resamples the daily return series into a
synthetic price path, so future returns are independent of every feature by construction
while drift, tails, feature persistence, sample length, the purged walk-forward geometry and
the actual estimators are retained — and forecasts are genuinely estimated at every origin.
Three nulls are used:

- **indep.** — iid draws. Destroys all dependence, including volatility clustering.
- **block** — 21-day blocks. Keeps volatility clustering, but also keeps whatever genuine
  within-block predictability the real series has, so it is an *upper bound* on size.
- **sign-flip** — 21-day blocks, then randomize the sign of each deviation from the mean:
  $r_t = \mu + s_t(r^\ast_t - \mu)$, $s_t = \pm 1$. $|r_t - \mu|$ is untouched bar for bar,
  so the volatility path survives; $\mathbb{E}[r_t \mid \mathcal{F}_{t-1}] = \mu$ exactly,
  so no conditional mean predictability remains. **This is the null the other two cannot
  isolate** — realistic dependence *and* a true no-predictability hypothesis.

Rejection rate at a nominal 5%, **2,000 replications per cell** (600 for the booster, $h=1$
only). BTC return pool. Monte Carlo standard errors ≤0.6 pp below 10%, ≤1.1 pp elsewhere,
≤1.6 pp in the booster cells.

| statistic | model | $h{=}1$ indep. | $h{=}1$ sign-flip | $h{=}1$ block | $h{=}7$ indep. | $h{=}7$ sign-flip | $h{=}7$ block |
|---|---|---|---|---|---|---|---|
| CW vs zero forecast | ridge | 12.0% | 14.1% | 53.2% | 22.1% | 28.6% | 25.7% |
| CW vs zero forecast | elastic net | 17.5% | 20.9% | 43.9% | 23.4% | 30.4% | 25.1% |
| CW vs zero forecast | GBM | 16.8% | 20.0% | — | — | — | — |
| **CW vs recursive mean** | ridge | **3.7%** | **4.8%** | 44.6% | **5.9%** | **7.2%** | 10.7% |
| **CW vs recursive mean** | elastic net | **3.1%** | **3.5%** | 30.6% | **5.7%** | **6.8%** | 9.5% |
| **CW vs recursive mean** | GBM | **3.7%** | **3.0%** | — | — | — | — |
| CW vs zero, drift-only model | — | 27.5% | 34.4% | 25.2% | 36.8% | 41.9% | 33.3% |
| DM vs zero (two-sided) | ridge | 61.0% | 55.5% | 15.0% | 50.6% | 46.9% | 46.8% |

Wild Clark–West was computed on every replication; its rejection rate differs from CW's by
at most **0.4 pp** in every cell, so it is not tabulated separately.

Four readings:

1. **The standard configuration rejects 3–6× too often**, and the drift-only row — a
   forecaster with no feature information at all, called significantly better than zero in
   27–42% of samples containing nothing — locates the cause.
2. **Substituting the recursive mean repairs it**, and does so for early-stopped gradient
   boosting as readily as for ridge: greedy splits, subsampling and a data-dependent tree
   count do not visibly break the approximation. Calibration evidence for these estimators
   under this null, not a theorem — and no prior measurement of it was found.
3. **The block column was misread in the first version.** It is not size failure under
   dependence: the sign-flipped null keeps the same volatility dynamics and gives 3.0–4.8%
   at $h=1$. Almost all of the block column is genuine within-block predictability that
   block resampling carries into the synthetic path.
4. **Wild Clark–West makes no difference**, including in the $h=7$ cells it was designed for.

**The calibration transfers across all three assets** (2,000 replications each, sign-flipped
null, ranges over ridge and elastic net):

| | BTC $h{=}1$ | BTC $h{=}7$ | ETH $h{=}1$ | ETH $h{=}7$ | SOL $h{=}1$ | SOL $h{=}7$ |
|---|---|---|---|---|---|---|
| CW vs zero forecast | 14.1–20.9% | 28.6–30.4% | 9.8–12.1% | 19.8–20.4% | 9.7–11.9% | 22.9–23.9% |
| **CW vs recursive mean** | **3.5–4.8%** | **6.8–7.2%** | **3.7–4.5%** | **6.9–7.5%** | **3.2–3.8%** | **7.3–7.5%** |

The recursive-mean row is stable across assets whose daily volatilities differ by 2×. The
zero-benchmark row is not, and the drift term says it should not be — it scales with drift
relative to noise, so BTC, with the lowest volatility and a substantial drift, is worst hit.

> **Two protocol lessons, learned the hard way.** The first version of this table came from
> 400 replications against a sample that a working-directory bug had silently re-downloaded
> 12 bars longer than the paper's. Re-running the *identical* script from the repo root moved
> CW vs the recursive mean at $h=7$ from 10.2% to 5.0%, and different seeds at 400
> replications moved cells by up to 3 pp. Pin the sample; use ≥2,000 replications; report
> Monte Carlo error.

#### 3.6.5 Pesaran–Timmermann

A model can be poor at magnitude and useful at direction, which is all a sign strategy
needs. Pesaran and Timmermann (1992) test independence between the sign of the forecast
and the sign of the outcome. With $P$ the hit rate, $P_y$ and $P_x$ the proportions of
positive outcomes and forecasts, and $P_{\ast} = P_y P_x + (1-P_y)(1-P_x)$:

$$S = \frac{P - P_{\ast}}{\sqrt{\widehat{\mathrm{var}}(P) - \widehat{\mathrm{var}}(P_{\ast})}}  \xrightarrow{d} \mathcal{N}(0,1)$$

$$\widehat{\mathrm{var}}(P) = \frac{P_{\ast}(1 - P_{\ast})}{n}$$

$$\widehat{\mathrm{var}}(P_{\ast}) = \frac{(2P_y-1)^2 P_x (1-P_x)}{n} + \frac{(2P_x-1)^2 P_y (1-P_y)}{n} + \frac{4 P_y P_x (1-P_y)(1-P_x)}{n^2}$$

**A positive statistic means sign-timing skill**; a negative one means the forecast is
reliably wrong-way. The statistic is undefined when the forecast never changes sign, which
is precisely the case for the always-long drift models, and it is reported as missing
rather than as a number.

#### 3.6.6 Sharpe ratio, PSR, and DSR

$$\mathrm{SR} = \sqrt{365/h} \cdot \frac{\bar{\pi}}{s_\pi}$$

Sharpe ratios are noisy in short samples and inflated by selection. The Probabilistic
Sharpe Ratio (Bailey and López de Prado, 2014) gives the probability that the true Sharpe
exceeds a benchmark $\mathrm{SR}^{\ast}$, correcting for sample length, skewness $\gamma_3$,
and kurtosis $\gamma_4$:

$$\widehat{\mathrm{PSR}}(\mathrm{SR}^{\ast}) = \Phi\left[ \frac{(\widehat{\mathrm{SR}} - \mathrm{SR}^{\ast})\sqrt{n-1}} {\sqrt{1 - \gamma_3 \widehat{\mathrm{SR}} + \frac{\gamma_4 - 1}{4}\widehat{\mathrm{SR}}^2}} \right]$$

The Deflated Sharpe Ratio sets $\mathrm{SR}^{\ast}$ to the expected maximum across $N$ trials,
so that the bar rises with the size of the search:

$$\mathbb{E}\left[\max_{i \le N} \mathrm{SR}_i\right] \approx \sigma_{\mathrm{SR}}\left[(1-\gamma) \Phi^{-1}\left(1 - \tfrac{1}{N}\right) + \gamma \Phi^{-1}\left(1 - \tfrac{1}{Ne}\right)\right]$$

with $\gamma$ the Euler–Mascheroni constant. Here $N$ is the number of models raced within
one asset-horizon, which is a floor on the true search; the reported DSR is therefore an
upper bound and is labelled as such in the generated report.

#### 3.6.7 Confidence intervals

Strategy returns are serially dependent, so intervals come from a circular block bootstrap
(Politis and Romano, 1994) with block length $\lfloor n^{1/3} \rceil$ and 500 resamples,
reported as percentile intervals.

#### 3.6.8 Multiple testing

Eighteen settings tested at the 5% level produce $0.05 \times 18 = 0.9$ rejections from
noise alone. Raw counts are therefore reported alongside two adjustments of the
Clark–West p-values over the family of 18. Holm–Bonferroni controls the family-wise error
rate,

$$p^{\text{Holm}}_{(i)} = \max_{j \le i} \min\left\lbrace (m - j + 1) p_{(j)}, 1 \right\rbrace$$

and Benjamini–Hochberg controls the false discovery rate,

$$p^{\text{BH}}_{(i)} = \min_{j \ge i} \min\left\lbrace \frac{m}{j} p_{(j)}, 1 \right\rbrace$$

Benchmarks are excluded from the family. They are the null being tested against, not
candidates in the search.

---

## 4. Results

A typeset write-up of this study is in
[**`paper/paper.pdf`**](paper/paper.pdf) (16 pages, NeurIPS preprint format). Its
result tables are emitted from `reports/results.csv` rather than transcribed, so the
manuscript cannot quote a number the study does not produce.

Full tables for every asset, horizon, and model are in
[`reports/results.md`](reports/results.md), regenerated by `make backtest`. The machine
readable version is [`reports/results.csv`](reports/results.csv). Every number quoted in
this document is checked against that file programmatically.

### 4.1 Point accuracy

![Figure 2](reports/figures/fig2_r2.png)

**Figure 2.** Out-of-sample $R^2$ against the drift benchmark. Red is worse than an
ex-ante historical-mean forecast; the benchmark row is exactly zero by construction.

$R^2_{OS}$ is negative in **16 of 18** machine-learning settings, ranging from $-0.0908$
(ETH, $h=7$, ridge) to $+0.0031$ (BTC, $h=1$, elastic net). Rank information coefficients
lie between $-0.080$ and $+0.053$. Representative slice:

| model | RMSE | $R^2_{OS}$ | Dir. acc | Rank IC | DM (p) | CW (p) | CW p Holm / BH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_walk | 0.0301 | +0.0005 | — | — | — | — | — |
| historical_mean | 0.0301 | 0.0000 | 0.506 | 0.002 | +0.18 (0.860) | +1.03 (0.152) | — |
| ar1 | 0.0301 | −0.0013 | 0.513 | 0.033 | +0.34 (0.737) | +1.24 (0.107) | — |
| ridge | 0.0302 | −0.0079 | 0.506 | 0.049 | +0.76 (0.447) | **+2.62 (0.004)** | 0.079 / **0.046** |
| elastic_net | 0.0301 | +0.0031 | 0.505 | 0.046 | −0.46 (0.644) | **+2.57 (0.005)** | 0.087 / **0.046** |
| gbm | 0.0301 | +0.0007 | 0.501 | 0.019 | −0.04 (0.970) | +2.02 (0.022) | 0.350 / 0.131 |

*BTC, $h=1$. Each test is shown as statistic (p-value). DM negative favours the model, CW
positive favours the model.*

That ridge posts $\mathrm{CW} = +2.62$, $p = 0.004$ while its $R^2_{OS}$ is $-0.0079$ is
not a contradiction. Clark–West asks whether the population MSPE is lower. A model can
satisfy that while its own parameter-estimation noise leaves the realized forecast worse
than a constant. It is evidence of a weak signal, not of a usable forecast, and it is
exactly the case the Clark–West paper warns about.

### 4.2 Predictive accuracy against the random walk

![Figure 4](reports/figures/fig4_dm_vs_cw.png)

**Figure 4.** Every setting placed by its Diebold–Mariano statistic against its
Clark–West statistic. The red band is where DM calls the model significantly worse; the
green band is where Clark–West rejects no-predictability.

| Test | Rejects for the model | Rejects for the benchmark |
| --- | ---: | ---: |
| Diebold–Mariano ($p<0.05$, two-sided) | **0** of 18 | 7 of 18 |
| Clark–West ($p<0.05$, one-sided) | **5** of 18 | — |

Under Diebold–Mariano the conclusion would be that machine learning is actively harmful for
this problem: no setting beats the random walk and seven lose to it significantly. Under
Clark–West against the same zero benchmark, five settings reject at an uncorrected 5%. Both
numbers come from the same forecasts, and **neither is the answer**, because the second test
rejects 14–30% of the time when nothing is predictable (Section 3.6.4).

The five, ordered by p-value, are BTC ridge ($p = 0.004$), BTC elastic net ($0.005$), BTC
gbm ($0.022$), ETH ridge ($0.031$), all at $h=1$, and BTC elastic net at $h=7$ ($0.039$).
The sixth, BTC ridge at $h=7$, misses at $p = 0.05004$ — a count that turns on four parts in
$10^5$ should not be reported as five without saying so.

**The benchmark changes which settings reject, not how many.** Substituting the recursively
estimated mean — the actual slopes-zero restriction of these estimators, and the benchmark
already used for $R^2_{OS}$ — moves the raw count from 5 of 18 to 7 of 18 at the same
bandwidth, and relocates it:

| Setting | CW vs zero | $p$ | CW vs recursive mean | $p$ |
| --- | ---: | ---: | ---: | ---: |
| BTC ridge, $h=1$ | 2.62 | 0.004 | 2.66 | **0.004** |
| BTC elastic net, $h=1$ | 2.57 | 0.005 | 2.67 | **0.004** |
| BTC gbm, $h=1$ | 2.02 | **0.022** | 1.75 | **0.040** |
| BTC ridge, $h=7$ | 1.64 | 0.050 | 1.36 | 0.086 |
| BTC elastic net, $h=7$ | 1.76 | **0.039** | 1.54 | 0.061 |
| ETH ridge, $h=1$ | 1.86 | **0.031** | 1.76 | **0.039** |
| ETH ridge, $h=7$ | −0.79 | 0.785 | −1.62 | 0.948 |
| SOL gbm, $h=1$ | 0.82 | 0.207 | 1.87 | **0.031** |
| SOL ridge, $h=7$ | −1.43 | 0.923 | 2.01 | **0.022** |
| SOL elastic net, $h=7$ | −1.41 | 0.921 | 2.28 | **0.011** |

SOL, which shows nothing at either horizon against the zero benchmark, becomes the strongest
rejecter. The earlier reading — four of five rejections in BTC, "consistent with a weak
short-horizon effect in the most liquid asset" — was an artifact of the benchmark. Under the
corrected benchmark the geography reverses; under calibrated size neither pattern is
distinguishable from noise.

### 4.3 Multiple testing

![Figure 5](reports/figures/fig5_multiple_testing.png)

**Figure 5.** The 18 Clark–West p-values ranked, against the Holm–Bonferroni and
Benjamini–Hochberg thresholds. A bar must fall below a line to be rejected by that
procedure.

| Threshold | Surviving |
| --- | ---: |
| Uncorrected, $\alpha = 0.05$ | 5 |
| Benjamini–Hochberg, FDR 5% | **2** |
| Holm–Bonferroni, FWER 5% | **0** |

Controlling the false discovery rate leaves BTC ridge and BTC elastic net at $h=1$, both with
$p^{\text{BH}} = 0.046$, a hair under the line. Controlling the family-wise error rate leaves
nothing: the smallest raw p-value, 0.0044, does not clear the Holm threshold of
$0.05/18 = 0.0028$.

**But 0.9 is the wrong reference, and which reference is right depends on the benchmark.**
Averaging the sign-flipped column of Section 3.6.4 over the measured model–horizon
combinations, the expected count is about **4** against the zero forecast and about **1**
against the recursive mean. So the two benchmarks say opposite things about the same
forecasts:

- **Against zero:** 5 observed, ~4 expected. Uninformative. The multiplicity corrections
  above are applied to p-values from a test whose size is 14–30% rather than 5%, so they do
  not control the error rates they name, and no calibrated FDR bound can be extracted from
  this column.
- **Against the recursive mean:** 7 observed, ~1 expected. A real excess, and the only
  positive result in this study. Two settings survive FDR control (BTC ridge and BTC elastic
  net at $h=1$, $p_{BH} = 0.035$) — but with $\alpha = 10^{-3}$ those are close to the same
  estimator on the same data, so they are one candidate, not two. None survives FWER control
  ($p_{Holm} = 0.069$). At the plug-in bandwidth the counts are 5 and 1 instead, and nothing
  in the data selects the bandwidth, so the surviving set is not identified by this design.

### 4.4 Sign timing

![Figure 3](reports/figures/fig3_diracc.png)

**Figure 3.** Directional accuracy per setting against the band a fair coin would occupy.
The band uses the effective sample size $n/h$, because consecutive $h$-day forecasts are
built from overlapping windows and counting each as independent would shrink the band by
$\sqrt{h}$ and manufacture significance.

Directional accuracy spans 0.471 to 0.528 and every setting lies inside its coin-flip
band. **Zero of 18** settings show significant sign-timing skill under Pesaran–Timmermann.

**The two previously reported as significantly *anti*-predictive are not significant either.**
Every variance term in the Pesaran–Timmermann statistic divides by $n$, so it requires
non-overlapping observations, and the results table feeds it all $n$ rows of an $h$-step
forecast whose labels overlap $h-1$ times. That inflates the statistic by roughly $\sqrt{h}$.
Figure 3 above avoids exactly this error by drawing its band at $n/h$ — so the table and the
figure disagreed by a factor of $\sqrt{7}$. Recomputed on non-overlapping subsamples,
averaged over all $h$ phase offsets:

| Setting | $S$ (all $n$) | $p$ | $S$ (non-overlapping) | $p$ |
| --- | ---: | ---: | ---: | ---: |
| ETH ridge, $h=7$ | −3.29 | 0.001 | −1.27 | 0.268 |
| ETH elastic net, $h=7$ | −3.33 | 0.001 | −1.27 | 0.243 |
| SOL ridge, $h=7$ | −1.73 | 0.083 | −0.66 | 0.597 |
| BTC ridge, $h=7$ | +1.41 | 0.159 | +0.54 | 0.592 |

The conclusion — no sign-timing skill in either direction — is unchanged. The evidence
previously offered for its most striking part was invalid.

### 4.5 Economic performance

![Figure 7](reports/figures/fig7_sharpe.png)

**Figure 7.** Net Sharpe with 95% circular-block-bootstrap intervals. The dashed line is
buy-and-hold on the same schedule and cost model.

Two of 18 settings have a net Sharpe whose interval excludes zero, both gradient boosting
on ETH:

| Setting | Sharpe | 95% CI | PSR | DSR | Max DD | Position changes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ETH, $h=1$, gbm | 0.91 | [0.00, 1.74] | 0.99 | 0.74 | −80.4% | 163 |
| ETH, $h=7$, gbm | 0.92 | [0.02, 1.80] | 0.99 | 0.76 | −84.5% | 43 |

Both intervals begin essentially at zero. Neither setting appears among the two that
survive Benjamini–Hochberg, and neither of those two has an interval excluding zero.

**The statistical winners and the economic winners are disjoint sets.** Two unrelated
groups of survivors drawn from one search is the signature of noise. If a genuine
short-horizon effect existed and were tradeable, the settings that detect it and the
settings that profit from it should overlap.

![Figure 6](reports/figures/fig6_equity.png)

**Figure 6.** Out-of-sample equity net of costs, against the always-long reference.

**Both economic winners are market exposure, and here is the measurement.** Regressing each
strategy's net return on buy-and-hold's, same schedule and cost model, Newey–West standard
error on the intercept:

| Setting | Sharpe | % long | $\beta$ to B&H | $\alpha$ (ann.) | $t(\alpha)$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH gbm, $h=1$ | 0.91 | 94% | 0.76 | +0.236 | 1.07 |
| ETH gbm, $h=7$ | 0.92 | 91% | 0.89 | +0.192 | 1.15 |
| BTC gbm, $h=7$ | 0.82 | 90% | 0.82 | +0.097 | 0.66 |
| SOL elastic net, $h=7$ | 0.60 | 56% | 0.43 | +0.499 | 1.23 |
| `historical_mean` (all) | 0.23–0.84 | 100% | 1.00 | ≈0 | ≤\|1.6\| |

Neither setting whose Sharpe interval excludes zero has an alpha distinguishable from zero
once market exposure is removed.

**And execution is assumed at a price the signal has not been computed from yet.** Every
feature at bar $t$ uses the completed close $C_t$; the backtest enters at that same $C_t$.
Delaying entry one bar, forecasts unchanged:

| Setting | Sharpe (as reported) | Sharpe ($t+1$ entry) | Δ |
| --- | ---: | ---: | ---: |
| ETH gbm, $h=1$ | 0.91 | 0.21 | −0.70 |
| BTC ridge, $h=1$ | 0.51 | −0.23 | −0.74 |
| ETH elastic net, $h=1$ | −0.11 | −0.61 | −0.50 |
| BTC elastic net, $h=1$ | 0.33 | 0.06 | −0.27 |
| ETH gbm, $h=7$ | 0.92 | 0.98 | +0.06 |
| all $h=7$ settings | — | — | \|Δ\| ≤ 0.20 |

Every $h=1$ result is a property of the execution convention as much as of the forecasts.
The one surviving economic result, ETH gbm at $h=7$, is the one shown above to be 91% long
with $\beta = 0.89$.

Six of 18 settings out-Sharpe buy-and-hold on the point estimate, which is the weakest
form of that claim given that every interval contains zero. Meanwhile the
`historical_mean` forecaster attains Sharpe ratios of 0.84, 0.80, and 0.25 across the
three assets, which looks impressive until one notices it makes exactly **one position
change**. It predicts a positive drift, so it is always long. Its net returns are
numerically identical to buy-and-hold, asset by asset and horizon by horizon, and the
report carries buy-and-hold as an explicit row so this cannot be mistaken for model
performance.

Maximum drawdowns run from −55% to −99% across the machine-learning settings. This is a
property of the position sizing rather than of any model: unit leverage flipping direction
on roughly 70% annualized volatility carries a large variance drag, so the geometric
return sits well below the arithmetic one. Buy-and-hold itself draws down 76% to 96% over
the same window.

### 4.6 Robustness to the trading schedule

![Figure 8](reports/figures/fig8_phase.png)

**Figure 8.** Net Sharpe on each of the $h$ possible start offsets of the trading
schedule. The circled point is the offset the results table reports.

Sampling every $h$-th bar leaves $h$ schedules and nothing privileges the one that starts
at the first bar. At $h=7$ the spread across offsets is substantial:

| Setting | Reported Sharpe | Range across 7 offsets | Span |
| --- | ---: | --- | ---: |
| SOL, ridge | 0.45 | [−0.83, 0.57] | 1.40 |
| SOL, elastic_net | 0.60 | [−0.55, 0.60] | 1.15 |
| ETH, gbm | 0.92 | [0.49, 1.05] | 0.56 |
| BTC, gbm | 0.82 | [0.50, 0.82] | 0.32 |

SOL ridge reports a Sharpe of 0.45 from a distribution that runs from −0.83 to 0.57. The
headline number is a property of where the sampling happened to begin. Any study that
reports a single-offset Sharpe at a multi-day horizon without this check is reporting one
draw and calling it an estimate.

---

## 5. Discussion

**The benchmark defines the hypothesis, and the default one defines the wrong hypothesis.**
Diebold–Mariano is the wrong default whenever the benchmark is nested. But swapping in
Clark–West is not sufficient, because the correction inherits whatever null the benchmark
encodes. Against a zero-return benchmark, an estimator with a fitted intercept is tested for
drift and conditional predictability jointly. The measured consequence is a test that rejects
3–4× too often, and a rejecting set whose membership changes completely — BTC to SOL — when
the benchmark is repaired. This generalizes past this dataset: **a paper reporting Clark–West
against a random walk has not stated its null until it says whether the larger model fits an
intercept.**

**Calibrate the test; do not assume it.** That distortion is invisible in code review, survives
a 154-test suite that checks every formula against hand-computed values, and is *not* detected
by simulating a forecast that is independent noise — a design that measures neither size nor
bias, and which this project used. What detects it is a null built by resampling the data
through the actual pipeline with actual estimated forecasts. It costs a few minutes of compute
and should be standard practice.

**Most of the apparent performance was exposure, and it can be measured.** The highest Sharpe
ratios belong to a forecaster that never changes its mind; the two settings whose intervals
exclude zero carry $\beta = 0.76$ and $0.89$ to buy-and-hold with insignificant alpha; and a
one-bar execution delay removes most of the $h=1$ performance. Three cheap controls — a
buy-and-hold row, an exposure regression, a delayed-entry variant — separate signal from
exposure and from timing convention.

**This is not evidence for weak-form efficiency.** The design reaches 80% power only at a
population $R^2$ of 1.34%, so it cannot exclude an economically relevant effect either. What it
establishes is narrower and mostly methodological.

---

## 6. Limitations

- **Asset selection is not survivorship-free.** BTC, ETH, and SOL were chosen knowing they
  survived. This biases toward profitable long exposure and would invalidate a
  cross-sectional claim, though it strengthens a negative result on predictability.
- **Daily spot bars only.** No order book, no funding rates, no intraday structure, no
  cross-exchange information. Short-horizon predictability, if it exists, is more likely
  to live at frequencies this data cannot see.
- **A compact feature set and three model families.** Richer features or sequence models
  might change the picture. The leakage discipline would not need to change with them.
- **Shorts are modelled as fully collateralized**, returning $-r$ with no borrow cost,
  margin requirement, or liquidation. This is reasonable for assessing signal quality and
  optimistic as an execution model.
- **Costs are a flat per-side estimate**, not a queue-level execution model, and do not
  scale with size or volatility. A break-even sweep is reported
  (`audit/scripts/cost_curve.py`): 16/18 settings are positive at 0 bp, 13/18 at 17 bp, 9/18 at
  40 bp, median break-even 31 bp. The settings tolerating the highest costs are the ones that
  barely trade — ETH gbm at $h=1$ breaks even at 148 bp and is 94% long — so cost tolerance
  here tracks turnover, not forecast quality.
- **Intervals are percentile bootstrap**, not BCa. Adequate for a reject / do-not-reject
  reading, mildly biased for a statistic as asymmetric as the Sharpe ratio.
- **The deflated Sharpe understates the true search.** It deflates for the six models raced
  within one asset-horizon — three of which are benchmarks — not for the whole grid, nor for
  the feature set and horizons fixed before the grid ran. It should not be read as a corrected
  number.
- **The size calibration resamples all three assets' returns**, but the booster is calibrated
  at $h=1$ only, and size is not established for other frequencies or estimators outside the
  three used here. The sign-flipped null removes conditional mean
  predictability while preserving the volatility path, which is the dependence that matters
  most for daily returns — but it destroys skewness dynamics and leverage effects along with
  the sign, so it is not a full replica of the real process.
- **The minimum detectable effect is large.** Injecting a known AR(1) signal into the
  sign-flipped null — population $R^2 = \rho^2$ by construction — the design reaches 80% power
  only at $\rho = 0.116$, i.e. a population $R^2$ of **1.34%**, roughly four times the largest
  $R^2_{OS}$ observed anywhere in the study. Nothing here bounds predictability below about one
  percent of return variance. (`audit/scripts/power_table.py`)
- **Hyperparameters were fixed by hand and never tuned.** No conclusion here is about a model
  *family*; they are about `Ridge(alpha=1)`, `ElasticNet(1e-3, 0.5)`, and one booster config.
- **The sample end date was unpinned until this revision.** `StudyConfig.end` defaulted to
  `None`, which resolves to today, and `DEFAULT_CACHE_DIR` is a *relative* path — so running
  from a different working directory missed the cache and silently re-dated the study. That is
  how one Monte Carlo pass ended up measured on a sample twelve bars longer than the paper's.
  `StudyConfig.end` is now pinned to `2026-07-18` in the configuration, not only by the
  presence of the cache.

---

## 7. Conclusion

Across 18 model, asset, and horizon combinations evaluated by purged walk-forward
cross-validation from 2020 to 2026, **the answer depends almost entirely on which null the
test states.** Clark–West against a zero-return benchmark at an $h-1$ bandwidth — the
configuration in standard use — rejects 10–30% of the time on data constructed to contain no
predictability, because a fitted intercept makes the zero forecast the wrong restriction and
the sample drift enters the statistic directly. Its 5 of 18 rejections are roughly what noise
produces. Against the recursively estimated mean, which is the restriction these estimators do
nest, the same statistic is approximately correctly sized — 3–5% at $h=1$ and 7–8% at $h=7$
under a null that preserves volatility clustering, including for early-stopped gradient
boosting — and it rejects in 7 of 18, against about 1 expected.

That excess is the only positive result in the study, and it is weak: the two settings that
survive FDR control are two parametrisations of one linear model on one asset, none survives
FWER control, the surviving set moves with the HAC bandwidth, and the same settings have
negative $R^2_{OS}$, no sign-timing skill, and no alpha net of costs and market exposure. The
only significant sign-timing results in the first version were an overlapping-label artifact;
the two settings with Sharpe intervals excluding zero carry $\beta = 0.76$ and $0.89$ to
buy-and-hold with insignificant alpha, and the one-day one loses three quarters of its Sharpe
to a one-bar execution delay.

I do not claim that no crypto signal exists at daily frequency; this design is not powered to
support that claim. The transferable result is that four routine choices — **which benchmark,
which HAC bandwidth, whether labels overlap, and when execution is assumed** — determined every
apparent finding in the first version of this study, that all four are cheap to check, and that
the benchmark has to be got right first, because it is the only one of the four that changes
the hypothesis rather than the precision.

The full audit, with reproduction scripts and a record of which earlier claims each number
overturns, is in [`audit/`](audit/README.md).

---

## Reproducing

```bash
make setup      # install the package and dev extras into ./venv
make backtest   # download data, run the study, regenerate reports/ and figures
make test       # 164 tests, including the no-lookahead and purge guarantees
make check      # lint, type-check, test (the CI gate)
make app        # a small local viewer on http://127.0.0.1:8000
make paper      # typeset paper/paper.pdf (needs tectonic)
```

The certificate results and the calibration behind them:

```bash
./venv/bin/python audit/scripts/gen_forecasts.py           # 72,900 OOS forecasts
./venv/bin/python audit/scripts/certificate_study.py       # certificates, all 18 settings
./venv/bin/python audit/scripts/certificate_study.py --payoff sign   # directional variant
./venv/bin/python audit/scripts/certificate_calibration.py 400       # drift sweep + horizon law
./venv/bin/python audit/scripts/mc_joint_null.py 2000 --gbm          # joint null for Clark-West
./venv/bin/python audit/scripts/joint_null_report.py                 # P(N >= k), Romano-Wolf
```

The certificate needs no calibration -- `certificate_calibration.py` exists to *demonstrate*
Theorem 1, not to license it. The joint null is the expensive one, and it is expensive
precisely because p-values over a dependent grid have to be simulated.

`make backtest` runs in roughly 15 seconds and rewrites `reports/results.md`,
`reports/results.csv`, and all eight figures. Market data is cached under `data/cache/`,
so re-runs are offline. `reports/results.md` is generated, never hand-edited: its headline
counts, its interpretation section, and its figure captions are all computed from the
results table, which is why they cannot drift away from the numbers they describe.

The CI gate runs ruff, black, mypy with `disallow_untyped_defs`, and the full test suite
with an 80% coverage floor, on Python 3.12 and 3.13.

---

## Repository layout

```
src/alphacert/         the instrument, dependency-light and study-independent
  certificate.py       the drift-robust e-process (Theorem 1)
  payoffs.py           odd transforms: identity / tanh / sign, and what each assumes
  merge.py             e-value averaging, e-BH, phase split for overlapping horizons
  bounds.py            anytime-valid ceiling on incremental value
  design.py            detection-horizon law, certifiable-ratio calculator
src/cryptoforecast/
  config.py            frozen study configuration
  data/                cached Yahoo Finance OHLCV loader
  features.py          leak-free, scale-free features
  targets.py           forward-return targets
  dataset.py           aligned (features, target) builder
  splits.py            purged and embargoed walk-forward
  models/              benchmarks, ridge/elastic-net, gradient boosting
  backtest/            walk-forward engine, cost model, trading rule
  evaluate/            metrics, statistical tests, report renderer
  plots/               the eight figures, one module per topic
  cli.py               cryptoforecast {data,backtest,report}
app/                   a small Flask viewer over the same evaluation
paper/                 NeurIPS-format preprint; tables generated from the results
reports/               generated results.md, results.csv, figures (PNG and PDF)
tests/                 leakage guarantees, statistical formulas, and the certificate's
                       validity, power and documented numerical failure mode
audit/scripts/         every number in the audit, one script per claim
```

---

## References

1. Bailey, D. and López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
   Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio
   Management* 40(5), 94–107.
2. Benjamini, Y. and Hochberg, Y. (1995). Controlling the False Discovery Rate: A
   Practical and Powerful Approach to Multiple Testing. *JRSS-B* 57(1), 289–300.
3. Campbell, J. Y. and Thompson, S. B. (2008). Predicting Excess Stock Returns Out of
   Sample: Can Anything Beat the Historical Average? *Review of Financial Studies* 21(4),
   1509–1531.
4. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*,
   785–794.
5. Clark, T. E. and McCracken, M. W. (2001). Tests of Equal Forecast Accuracy and
   Encompassing for Nested Models. *Journal of Econometrics* 105(1), 85–110.
6. Clark, T. E. and West, K. D. (2006). Using Out-of-Sample Mean Squared Prediction Errors
   to Test the Martingale Difference Hypothesis. *Journal of Econometrics* 135(1–2),
   155–186.
7. Clark, T. E. and West, K. D. (2007). Approximately Normal Tests for Equal Predictive
   Accuracy in Nested Models. *Journal of Econometrics* 138(1), 291–311.
8. Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of
   Business and Economic Statistics* 13(3), 253–263.
9. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work.
   *Journal of Finance* 25(2), 383–417.
10. Goyal, A. and Welch, I. (2008). A Comprehensive Look at the Empirical Performance of
    Equity Premium Prediction. *Review of Financial Studies* 21(4), 1455–1508.
11. Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the Equality of Prediction
    Mean Squared Errors. *International Journal of Forecasting* 13(2), 281–291.
12. Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
    *Scandinavian Journal of Statistics* 6(2), 65–70.
13. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7,
    purged and embargoed cross-validation.
14. Magner, N. and Hardy, N. (2022). Cryptocurrency Forecasting: More Evidence of the
    Meese–Rogoff Puzzle. *Mathematics* 10(13), 2338.
15. Moosa, I. and Burns, K. (2016). The Random Walk as a Forecasting Benchmark: Drift or
    No Drift? *Applied Economics* 48(43), 4131–4142.
16. Newey, W. K. and West, K. D. (1987). A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*
    55(3), 703–708.
17. Pesaran, M. H. and Timmermann, A. (1992). A Simple Nonparametric Test of Predictive
    Performance. *Journal of Business and Economic Statistics* 10(4), 461–465.
18. Pincheira, P., Hardy, N. and Muñoz, F. (2021). "Go Wild for a While!": A New Test for
    Forecast Evaluation in Nested Models. *Mathematics* 9(18), 2254.
19. Pincheira, P., Hardy, N. and Bentancor, A. (2022). A Simple Out-of-Sample Test of
    Predictability against the Random Walk Benchmark. *Mathematics* 10(2), 228.
20. Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the
    American Statistical Association* 89(428), 1303–1313.
21. West, K. D. (1996). Asymptotic Inference about Predictive Ability. *Econometrica*
    64(5), 1067–1084.

---

## License

**Non-commercial.** Code under [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/);
paper, figures and data under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [LICENSE](LICENSE). Research, teaching and evaluation are permitted; commercial use is not,
without a separate written licence. Enquiries: demirguven178@gmail.com

Note what that does *not* cover. Copyright protects this text and this code, not the method.
Anyone may read the paper, reimplement the certificate from the equations, and use it
commercially — no licence can prevent that, and none is claimed to.

Commits up to `4eefb24` were published under MIT and remain so for copies already obtained.

This is a research exercise. It is not investment advice, and its principal empirical finding
is that the models tested do not work.
