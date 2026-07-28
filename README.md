# Are Short-Horizon Cryptocurrency Returns Predictable Out of Sample?

**A purged walk-forward study with nested-model tests, transaction costs, and multiple-testing control**

[![ci](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml/badge.svg)](https://github.com/ITheClixs/crypto-return-predictability/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![tests](https://img.shields.io/badge/tests-136-informational)
![coverage](https://img.shields.io/badge/coverage-96%25-informational)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Abstract

I test whether daily cryptocurrency returns can be forecast out of sample well enough to
survive trading costs. Three assets (BTC, ETH, SOL), two horizons (1 and 7 days), and six
forecasters (random walk, historical mean, AR(1), ridge, elastic net, gradient-boosted
trees) are evaluated by purged and embargoed walk-forward cross-validation over
2020-07-23 to 2026-07-17, giving 18 machine-learning settings against a martingale null.

The headline result depends entirely on which test is used, and that is the point of the
paper. Diebold–Mariano rejects in favour of the random walk in 7 of 18 settings and never
in favour of a model. But the random walk is *nested* inside every model considered here,
and under nesting the Diebold–Mariano statistic is biased toward the smaller model.
Applying the Clark–West correction, 5 of 18 settings reject the no-predictability null at
an uncorrected 5% level, against 0.9 expected from a search that size. After controlling
the false discovery rate, 2 survive; after controlling the family-wise error rate, none do.

The two survivors are not usable forecasts. Their out-of-sample $R^2$ against a
recursively estimated drift benchmark is $-0.0079$ and $+0.0031$, and their net Sharpe
ratios of 0.51 and 0.33 carry bootstrap intervals of $[-0.18, 1.23]$ and $[-0.35, 1.02]$.
No setting shows significant sign-timing skill, which is the property a directional
strategy actually trades. The two settings whose Sharpe interval excludes zero are
different settings from the two that survive statistical correction: the statistical and
the economic winners do not coincide, which is what one expects from noise rather than
from an edge.

Every high-Sharpe result in the study is long-only exposure in disguise. The
historical-mean forecaster is always long, and its profit and loss is numerically
identical to buy-and-hold on the same schedule and cost model.

**Conclusion.** Over this sample, with this feature set, there is a faint and fragile
statistical signal in BTC at the one-day horizon and no tradeable one anywhere.

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

This study is constructed so that each of these is either impossible by design or
measured and reported. The contribution is not a model. It is a falsifiable evaluation
protocol, applied honestly, that returns an uncomfortable answer and reports it.

A secondary contribution is methodological. The most common test for comparing forecast
accuracy, Diebold–Mariano, is invalid for the comparison this literature actually makes:
a conditional model against a random walk that the model nests. Section 3.6.2 sets out
why, and Section 4.2 shows that in this data the correction changes the conclusion from
"nothing predicts, several are significantly worse" to "five settings reject, two survive
false-discovery control, none is tradeable." Both statements are defensible. Only the
second is correct.

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

Note that setting all slopes to zero recovers the random walk from any of the three
machine-learning models. They are *nested* extensions of the benchmark, which Section
3.6.2 shows is not a detail.

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

The difficulty is that Diebold–Mariano assumes the two forecasts are non-nested. Here they
are not: the random walk is the point in the model's parameter space where every slope is
zero. Under the null of no predictability the larger model must still *estimate* those
zero coefficients, and the resulting estimation noise inflates its sample mean squared
prediction error even though its population MSPE is identical. Diebold–Mariano reads that
noise as evidence for the benchmark and is undersized as a test of predictability.

#### 3.6.3 Clark–West

Clark and West (2007) subtract the estimation-noise term explicitly. With $\hat{y}^b$ the
nested benchmark and $\hat{y}^m$ the larger model,

$$\hat{f}_t = \left(y_t - \hat{y}^{b}_t\right)^2 - \left[\left(y_t - \hat{y}^{m}_t\right)^2 - \left(\hat{y}^{b}_t - \hat{y}^{m}_t\right)^2\right]$$

and the statistic $\bar{f}/\sqrt{\hat{V}_f/n}$ is compared against a standard normal,
one-sided. **A positive statistic favours the model.** The sign convention is opposite to
Diebold–Mariano, which is a good reason to never report either p-value without its
statistic.

The bias this corrects is demonstrated rather than asserted:
`tests/test_stats.py::test_dm_is_biased_against_a_useless_nested_model_and_clark_west_is_not`
simulates 200 replications in which the larger model forecasts pure noise, so the null is
true by construction. The mean Diebold–Mariano statistic is positive and it rejects in
favour of the benchmark far above the nominal rate, while Clark–West stays centred on zero
and holds its size.

#### 3.6.4 Pesaran–Timmermann

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

#### 3.6.5 Sharpe ratio, PSR, and DSR

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

#### 3.6.6 Confidence intervals

Strategy returns are serially dependent, so intervals come from a circular block bootstrap
(Politis and Romano, 1994) with block length $\lfloor n^{1/3} \rceil$ and 500 resamples,
reported as percentile intervals.

#### 3.6.7 Multiple testing

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

The disagreement is the finding. Under Diebold–Mariano the conclusion would be that
machine learning is actively harmful for this problem: no setting beats the random walk
and seven lose to it significantly. Under the test that is valid for nested models, five
settings reject the null of no predictability. Both numbers come from the same forecasts.

The five, ordered by p-value, are BTC ridge ($p = 0.004$), BTC elastic net ($0.005$), BTC
gbm ($0.022$), ETH ridge ($0.031$), all at $h=1$, and BTC elastic net at $h=7$ ($0.039$).
Four of the five are BTC and four are at the one-day horizon. That concentration is
consistent with a weak short-horizon effect in the most liquid asset, and equally
consistent with chance across a search of this size. Section 4.3 settles which.

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

Five raw rejections against 0.9 expected by chance is more than noise would comfortably
produce, but it is not many. Controlling the false discovery rate leaves BTC ridge and BTC
elastic net at $h=1$, both with $p^{\text{BH}} = 0.046$, which is a hair under the line.
Controlling the family-wise error rate leaves nothing: the smallest raw p-value, 0.0044,
does not clear the Holm threshold of $0.05/18 = 0.0028$.

The honest summary is that the evidence for predictability is real but marginal, and would
not survive a stricter reader.

### 4.4 Sign timing

![Figure 3](reports/figures/fig3_diracc.png)

**Figure 3.** Directional accuracy per setting against the band a fair coin would occupy.
The band uses the effective sample size $n/h$, because consecutive $h$-day forecasts are
built from overlapping windows and counting each as independent would shrink the band by
$\sqrt{h}$ and manufacture significance.

Directional accuracy spans 0.471 to 0.528 and every setting lies inside its coin-flip
band. **Zero of 18** settings show significant sign-timing skill under
Pesaran–Timmermann. Two are significantly *anti*-predictive: ETH ridge and ETH elastic net
at $h=7$ post $S = -3.29$ and $S = -3.33$, both $p \approx 0.001$, with hit rates of 0.476
and 0.479.

Those two are worth dwelling on, because they are the reason this report never prints a
bare p-value. Reported as "$p = 0.001$" alone, they look like the strongest findings in
the study. They are among the worst.

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

Three things are worth taking from this.

**The choice of test changed the answer.** Diebold–Mariano is the default in applied
forecast comparison and it is the wrong default when the benchmark is nested, which it is
in almost every return-predictability study that compares a model to a random walk. Here
the difference is not cosmetic: "0 of 18, and 7 significantly worse" versus "5 of 18
reject." A referee shown only the first would conclude that the models are harmful. A
referee shown only the second, without Section 4.3, would conclude that something was
found. Neither is the result.

**Statistical significance and economic value came apart cleanly.** The settings that
reject no-predictability have negative or near-zero out-of-sample $R^2$ and Sharpe
intervals straddling zero. The settings that make money do not reject. This is what a
correctly specified null looks like when it is true, and it is a more informative outcome
than either group appearing alone.

**Most of the apparent performance was exposure.** The highest Sharpe ratios in the study
belong to a forecaster that never changes its mind. Any evaluation that had omitted a
buy-and-hold reference would have credited the drift model with skill it does not have.

The result is consistent with weak-form efficiency at daily frequency for large-cap
cryptocurrencies over this period, net of realistic costs. It does not say that no crypto
signal exists. It says that this feature set, at these horizons, on these assets, does not
produce one that survives its own error bars.

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
- **The deflated Sharpe understates the true search.** It deflates for the models raced
  within one asset-horizon, not for the whole grid, nor for the feature set and horizons
  fixed before the grid ran.
- **The sample end date is unpinned**, so re-running extends the window and the numbers
  drift. The evaluated window is stamped into `reports/results.md` on every run so any
  table can be tied to the data that produced it.

---

## 7. Conclusion

Across 18 model, asset, and horizon combinations evaluated by purged walk-forward
cross-validation from 2020 to 2026, there is a faint statistical signal in BTC at the
one-day horizon that survives false-discovery control and does not survive family-wise
control. It does not convert into a forecast that beats an ex-ante drift estimate, into
sign-timing skill, or into a Sharpe ratio distinguishable from zero after costs. The
strategies with the highest Sharpe ratios in the study are long-only exposure.

The negative result is the deliverable. It is reported with the tests that would have
detected a positive one, and with the corrections that would have removed a spurious one.

---

## Reproducing

```bash
make setup      # install the package and dev extras into ./venv
make backtest   # download data, run the study, regenerate reports/ and figures
make test       # 136 tests, including the no-lookahead and purge guarantees
make check      # lint, type-check, test (the CI gate)
make app        # a small local viewer on http://127.0.0.1:8000
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
reports/               generated results.md, results.csv, figures
tests/                 136 tests, including the leakage guarantees
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
