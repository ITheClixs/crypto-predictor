# Are Short-Horizon Cryptocurrency Returns Predictable Out of Sample?

**A purged walk-forward study with nested-model tests, transaction costs, and multiple-testing control**

[![ci](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml/badge.svg)](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![tests](https://img.shields.io/badge/tests-157-informational)
![coverage](https://img.shields.io/badge/coverage-97%25-informational)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Abstract

I test whether daily cryptocurrency returns can be forecast out of sample well enough to
survive trading costs. Three assets (BTC, ETH, SOL), two horizons (1 and 7 days), and six
forecasters (random walk, historical mean, AR(1), ridge, elastic net, gradient-boosted
trees) are evaluated by purged and embargoed walk-forward cross-validation over
2020-07-23 to 2026-07-17, giving 18 machine-learning settings against a martingale null.

The headline result depends entirely on which test is used, and on whether that test has
the size it claims. Diebold–Mariano rejects in favour of the random walk in 7 of 18
settings and never in favour of a model; the Clark–West correction for nesting turns that
into 5 of 18 rejecting the no-predictability null at an uncorrected 5%. Neither number is
the answer.

**The test in standard use is not correctly sized here.** Under a nested-estimation null
built by resampling the return series — retaining the drift, the tails, the feature
persistence, the sample length and the estimators, while removing conditional
predictability — Clark–West against a zero-return benchmark rejects **14–22%** of the time
at a nominal 5%. So the relevant comparison for 5 observed rejections is 2.6–4.0 expected,
not 0.9. The cause is identifiable: ridge, elastic net and the booster all fit an
unpenalised intercept, which equals the training-window mean, so the zero forecast is *not*
their slopes-zero restriction. The adjusted differential is
$\hat f_t = 2(y_t - \hat y^b_t)(\hat y^m_t - \hat y^b_t)$, and with $\hat y^b = 0$ its
expectation is $2(\mathbb{E}[y]\mathbb{E}[\hat y^m] + \mathrm{Cov}(y, \hat y^m))$ — the
first term is drift, nonzero whatever the features do.

Against the **recursively estimated mean** instead, measured size is 5–6% at $h=1$
(including for early-stopped XGBoost) and about 10% at $h=7$. Substituting it moves the raw
count from 5 of 18 to 7 of 18 and changes *which* settings reject: BTC survives at $h=1$,
BTC drops out at $h=7$, and SOL — which shows nothing against the zero benchmark — becomes
the strongest rejecter ($p = 0.011$ at $h=7$). Against calibrated size, neither pattern is
distinguishable from noise.

Three supporting results correct in the same direction. The only significant sign-timing
findings ($p \approx 0.001$ for two ETH settings at $h=7$) come from treating overlapping
7-day labels as independent, and are $p \approx 0.24$–$0.27$ on non-overlapping subsamples.
The two settings whose Sharpe interval excludes zero carry $\beta = 0.76$ and $0.89$ to
buy-and-hold with alpha *t*-statistics of 1.07 and 1.15 — that is the "long-only exposure"
claim, measured rather than asserted. And a one-bar execution delay takes the $h=1$ winner
from 0.91 to 0.21, because the backtest enters at the same close its features are computed
from.

**Conclusion.** Over this sample, with this feature set, no predictability claim survives a
correctly benchmarked and size-calibrated test. We do *not* claim the absence of
predictability: no minimum detectable effect is established, so this design is not powered
to exclude an economically relevant one. The transferable finding is that four routine
choices — which benchmark, which HAC bandwidth, whether labels overlap, and when execution
is assumed — determined every apparent result in the first version of this study.

> The full audit, including the reproduction scripts for every number above and a record of
> which earlier claims they overturn, is in [`audit/`](audit/README.md).

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
wrong, and correcting it is the main contribution.**

Diebold–Mariano is indeed invalid for the comparison this literature makes, and Clark–West
is the standard correction. The question that goes unasked is what the corrected test is
*testing*. A regression with a fitted intercept does not reduce to the zero-return forecast
when its slopes are set to zero — it reduces to the training-window mean. Testing against
zero therefore tests a joint hypothesis: no drift **and** no conditional predictability.
Where the drift is large, as it is for these assets over this window, that difference is not
academic: Section 3.6.3 derives the contaminating term, Section 3.6.4 measures the resulting
size distortion at 14–22% against a nominal 5%, and Section 4.2 shows that repairing the
benchmark changes *which* settings reject rather than how many.

The first version of this README reported "five of 18 settings reject, against 0.9 expected
from a search that size." That comparison assumed the test was correctly sized. It is not,
and the corrected expectation is 2.6–4.0. The earlier claim is stated here rather than
quietly removed, because the error is easy to make, invisible in code review, and undetected
by a test suite that checks every formula against hand-computed values — as this one does.

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

The replacement (`audit/scripts/mc_null.py`) resamples the daily return series into a
synthetic price path, so future returns are independent of every feature by construction
while drift, tails, feature persistence, sample length, the purged walk-forward geometry and
the actual estimators are retained — and forecasts are genuinely estimated at every origin.
Rejection rate at a nominal 5%, 400 replications per cell (150 for the booster):

| statistic | model | $h{=}1$ iid | $h{=}1$ block-21 | $h{=}7$ iid | $h{=}7$ block-21 |
|---|---|---|---|---|---|
| CW vs zero forecast | ridge | 14.2% | 53.8% | 23.5% | 28.2% |
| CW vs zero forecast | elastic net | 22.0% | 43.0% | 26.2% | 27.8% |
| CW vs zero forecast | GBM | 22.0% | — | — | — |
| **CW vs recursive mean** | ridge | **6.2%** | 42.8% | 10.2% | 11.2% |
| **CW vs recursive mean** | elastic net | **6.2%** | 28.2% | 9.8% | 10.8% |
| **CW vs recursive mean** | GBM | **5.3%** | — | — | — |
| CW vs zero, drift-only model | — | 27.3% | 27.0% | 39.0% | 34.0% |
| DM vs zero (two-sided) | ridge | 57.5% | 15.2% | 50.0% | 46.5% |

The iid columns resample returns independently and so measure **size**. The block columns
resample 21-day blocks, preserving volatility clustering but possibly carrying genuine
within-block dependence, so they bound the test's *sensitivity to dependence* rather than
measuring size.

Reading: the standard configuration is oversized by 3–4×, and the drift-only row — 27%
rejection with zero feature information — locates the cause. Repairing the benchmark fixes
most of it at $h=1$, and does so for early-stopped gradient boosting as readily as for
ridge, which is a useful negative result about a live worry: greedy splits, subsampling and
a data-dependent tree count do not visibly break the approximation. It is calibration
evidence for this estimator and this null, not a theorem. The repair is also incomplete —
$h=7$ stays at ~10%, and under dependence-preserving resampling nothing we tested is usable,
so a definitive count needs bootstrap inference on $\hat{f}_t$, which this study does not
have.

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
rejects 14–22% of the time when nothing is predictable (Section 3.6.4).

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

**But 0.9 is the wrong reference.** Five raw rejections would be surprising against a
correctly sized test. At the measured size of the test that produced them (Section 3.6.4) the
expectation is $18 \times 0.142 \dots 0.220 = 2.6$ to $4.0$; against the recursive-mean
benchmark, using per-horizon sizes, it is $9 \times 0.062 + 9 \times 0.10 \approx 1.5$. Both
corrections above are also applied to p-values from a test whose size is 14–22% rather than
5%, so they do not control the error rates they name.

The honest summary is that we cannot place a calibrated bound on the false-discovery rate over
this family with the analytic test available here, and that the raw counts are within what the
null produces. With the recursive-mean benchmark the surviving counts are additionally
bandwidth-dependent: 2 survive FDR control at $h-1$ lags, 5 at the plug-in bandwidth, with 0
and 1 surviving FWER control. A conclusion that depends on the bandwidth is not a conclusion.

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

**This is not evidence for weak-form efficiency.** An oversized test cannot bound
predictability in either direction, and no minimum detectable effect is computed here, so this
design is not powered to exclude an economically relevant effect. What it establishes is
narrower and mostly methodological.

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
  scale with size or volatility.
- **Intervals are percentile bootstrap**, not BCa. Adequate for a reject / do-not-reject
  reading, mildly biased for a statistic as asymmetric as the Sharpe ratio.
- **The deflated Sharpe understates the true search.** It deflates for the six models raced
  within one asset-horizon — three of which are benchmarks — not for the whole grid, nor for
  the feature set and horizons fixed before the grid ran. It should not be read as a corrected
  number.
- **The size calibration resamples one asset's returns.** Size is not established for other
  assets, other frequencies, or nulls preserving volatility clustering — where the corrected
  test rejects 28–43% and nothing tested here is usable. A definitive rejection count needs
  bootstrap inference on $\hat{f}_t$, which this study does not have.
- **No minimum detectable effect is computed**, so no statement here bounds predictability from
  above.
- **Hyperparameters were fixed by hand and never tuned.** No conclusion here is about a model
  *family*; they are about `Ridge(alpha=1)`, `ElasticNet(1e-3, 0.5)`, and one booster config.
- **The sample end date is unpinned.** `StudyConfig.end` defaults to `None`, which resolves to
  today, and `DEFAULT_CACHE_DIR` is a *relative* path — so running from a different working
  directory misses the cache and silently re-dates the study. The committed `data/cache/` pins
  this study to 2026-07-18, and the evaluated window is stamped into `reports/results.md`, but
  the pipeline as configured is not reproducible without that cache.

---

## 7. Conclusion

Across 18 model, asset, and horizon combinations evaluated by purged walk-forward
cross-validation from 2020 to 2026, **no predictability claim survives a correctly benchmarked
and size-calibrated test.** The test in standard use — Clark–West against a zero-return
benchmark at an $h-1$ bandwidth — rejects 14–22% of the time on data constructed to contain no
predictability, because a fitted intercept makes the zero forecast the wrong restriction and
the sample drift enters the statistic directly. Repairing the benchmark brings measured size to
5–6% at $h=1$, including for early-stopped gradient boosting, and *relocates* the rejections
from BTC to SOL rather than removing them; against calibrated size, neither pattern is
distinguishable from noise. The only significant sign-timing results were an overlapping-label
artifact. The two settings with Sharpe intervals excluding zero carry $\beta = 0.76$ and $0.89$
to buy-and-hold with insignificant alpha, and the one-day one loses three quarters of its
Sharpe to a one-bar execution delay.

I do not claim that no crypto signal exists at daily frequency; this design is not powered to
support that claim. The transferable result is that four routine choices — **which benchmark,
which HAC bandwidth, whether labels overlap, and when execution is assumed** — determined every
apparent finding in the first version of this study, and that all four are cheap to check.

The full audit, with reproduction scripts and a record of which earlier claims each number
overturns, is in [`audit/`](audit/README.md).

---

## Reproducing

```bash
make setup      # install the package and dev extras into ./venv
make backtest   # download data, run the study, regenerate reports/ and figures
make test       # 157 tests, including the no-lookahead and purge guarantees
make check      # lint, type-check, test (the CI gate)
make app        # a small local viewer on http://127.0.0.1:8000
make paper      # typeset paper/paper.pdf (needs tectonic)
```

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
tests/                 157 tests, including the leakage guarantees
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
5. Clark, T. E. and West, K. D. (2007). Approximately Normal Tests for Equal Predictive
   Accuracy in Nested Models. *Journal of Econometrics* 138(1), 291–311.
6. Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of
   Business and Economic Statistics* 13(3), 253–263.
7. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work.
   *Journal of Finance* 25(2), 383–417.
8. Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the Equality of Prediction
   Mean Squared Errors. *International Journal of Forecasting* 13(2), 281–291.
9. Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
   *Scandinavian Journal of Statistics* 6(2), 65–70.
10. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7,
    purged and embargoed cross-validation.
11. Newey, W. K. and West, K. D. (1987). A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*
    55(3), 703–708.
12. Pesaran, M. H. and Timmermann, A. (1992). A Simple Nonparametric Test of Predictive
    Performance. *Journal of Business and Economic Statistics* 10(4), 461–465.
13. Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the
    American Statistical Association* 89(428), 1303–1313.

---

## License

MIT, see [LICENSE](LICENSE). This is a research exercise. It is not investment advice, and
the result is that the models do not work.
