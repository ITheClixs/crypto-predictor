# Research changelog

Substantive changes to claims, not to code style. Each entry records what was claimed, why it was
wrong, what replaced it, and which downstream claims moved.

Baseline audited: commit `83881c7`. Data pinned by `data/cache/` at 2026-07-18. All numbers below
reproduce from `audit/scripts/`.

---

## 2026-08-06 — An error in our own corollary, and the price of the guarantee

An adversarial re-read of the manuscript, in the voice of a referee who would run the
experiments the paper omitted, turned up one wrong theorem-adjacent claim, one falsified
assumption, and one missing experiment. All three are fixed. The paper is materially more
honest and, we think, materially better for it.

### R25 — "Anytime-validity costs three per cent" was wrong (CRITICAL)

**Was:** Corollary "Anytime-validity is nearly free" compared `2 ln(1/alpha) / IR^2 = 5.99`
years for the certificate against `(z_a + z_b)^2 / IR^2 = 6.18` years for a fixed-sample test
at 80% power, and concluded that continuous monitoring costs 3% more data.

**Why wrong:** those are not the same quantity. The first is the *median* crossing time of the
wealth process; the second is an *eightieth percentile*. The coincidence of 5.99 and 6.18 is
seductive and meaningless.

**Now:** matched on level and power. Log wealth at the oracle stake is approximately
`N(x, 2x)` with `x = T IR_p^2 / 2`, so power `beta` needs
`x(beta) = ([z_b sqrt(2) + sqrt(2 z_b^2 + 4 ln(1/alpha))] / 2)^2`. The ratio to the
fixed-sample requirement is scale-free and equals **1.90 at 80% power**, 2.21 at median power,
and stays between 1.9 and 2.0 up to 95%. Anytime-validity costs roughly a doubling of the
sample, not three per cent. `power_matched_horizon` and `anytime_validity_cost` compute it and
a test pins both numbers.

### R26 — Assumption S is refuted by the paper's own data; identity is now the default

Conditional symmetry about a *constant* drift implies unconditional symmetry, hence zero
skewness — so the assumption is refutable from the marginal distribution alone. Measured
sample skewness: BTC −1.046 (p = 2.7e-78), ETH −0.956 (p = 1.2e-68), SOL −0.227 (p = 1.2e-05).
The `tanh` payoff was the default and rests on an assumption these data reject outright.

The identity payoff, which needs only a martingale difference plus an a-priori envelope, is
now the default throughout. On this study it is also the stronger result (largest e-value 3.05
and grid-level 1.43, against 1.81 and 1.18 for `tanh`), so the weaker assumption cost nothing.
Theorem "evidence in units of capital" is scoped to the identity payoff as well, because
`tanh((y-mu)/s)` is not a tradeable claim and the profit-and-loss reading is figurative for it.

### R27 — The head-to-head the paper never ran

**The referee's first question:** at matched size, is the certificate more or less powerful
than a correctly sized Clark-West? The paper did not say. It is less, and by a lot at the
nominal threshold.

Measured on this study's geometry, 400 replications, in the design where Clark-West against
the recursive mean is exactly correctly sized (measured 5.5%):

| IR | CW | CW size-matched | certificate | certificate size-matched |
|---|---|---|---|---|
| 1.0 | 0.230 | 0.210 | 0.030 | 0.163 |
| 1.5 | 0.510 | 0.495 | 0.188 | 0.390 |
| 2.0 | 0.853 | 0.850 | 0.480 | 0.757 |
| 3.0 | 1.000 | 1.000 | 0.988 | 0.998 |

Two losses separate cleanly. The *construction* is nearly as efficient as the incumbent — 0.757
against 0.850 at IR = 2, size-matched. What costs is anytime-validity: measured size at the
nominal threshold is 0.5–0.7% against a nominal 5%, which is Ville's inequality being loose for
a smooth wealth path, and is the same factor of two R25 derives independently.

**Consequently a claim in the Discussion is retracted.** The manuscript said the p-value and
the e-value were "disagreeing about what six years can settle, and Corollary [optimality] says
the e-value is right." That is wrong: the optimality corollary bounds *anytime-valid*
procedures, and Clark-West is not one, so its greater power is fully consistent with it. Part
of the certificate's silence is its own conservatism. What survives is narrower and stated as
such: Clark-West's count is jointly surprising (P(N>=7) = 0.0035) and individually
unidentified (smallest Romano-Wolf p = 0.052), and the certificate contributes a *bounded*
answer rather than a stronger one.

### R28 — Two design fixes that recovered most of the power, and two that did not

- **Weighted capital split (kept).** An equal average of the signal and centring bettors
  discards `ln 2 = 0.69` nats at the true drift, where the centring bettor breaks even.
  Weighting the signal 0.9 costs 0.105 nats and lifts power at IR = 2 from 0.217 to 0.310, at
  identical size. Now the default.
- **Pre-committed stake (kept).** Lifts power at IR = 2 from 0.357 to 0.480 at the nominal
  threshold. Already recommended by the regret corollary; now measured.
- **Stakes above Kelly (rejected).** Power at a fixed threshold looks like a goal-reaching
  problem, which would favour over-betting. It does not: doubling the stake takes power at
  IR = 2 from 0.217 to 0.060, quadrupling it to 0.025.
- **Stake mixture (rejected on cost).** A mixture over a stake grid attains the optimal
  `(1/2) ln T` regret but buys almost nothing over the plug-in here (0.257 against 0.233) at
  forty times the computation, because the plug-in is already near that bound.

### R29 — Positive control

The null result on returns was uninterpretable without evidence that the instrument can detect
anything on real data. Pointed at realised volatility — genuinely predictable, same assets,
same window, same identity payoff — it certifies all three within 148 to 416 days, with
e-values of 3.1e7, 4.2e9 and 7.5e8. The silence about returns is therefore about returns and
about the sample length, not about the instrument being inert.

## 2026-08-05 — Submission pass: one retraction, three additions

Preparing the manuscript for arXiv turned up one substantive error and closed the remaining
reviewer items. The error is listed first because it changes reported numbers.

### R21 — The "feasible execution" backtest was still the infeasible one (CRITICAL)

**Was:** the 2026-08-04 pass introduced `staggered_strategy(..., execution_lag=1)` and the
manuscript described it as the primary, *feasible* specification: "the weight in force during
bar t depends only on signals through t-1".

**Why wrong:** that sentence is true and irrelevant. A weight aligned to bar `t` is held from
close `t-1`, so a signal computed at close `t-1` and given `execution_lag=1` is traded *at the
very close that produced it*. That is the same-close convention the pass claimed to have
removed. The proof is arithmetic: at h=1 the "delayed" Sharpe ratios reproduced the
single-phase same-close ones to three decimals (ETH GBM 0.911 vs 0.911, BTC ridge 0.510 vs
0.509), and the genuinely delayed numbers -- the ones the reviewer had quoted, 0.91 to 0.21
and 0.51 to -0.23 -- appeared only at a shift of two.

**Now:** the argument is `entry_delay`, counted in bars between the signal becoming known and
the start of the holding period, so `0` is the same-close convention and `1` is feasible. Two
regression tests pin the semantics: one asserts that `entry_delay=0` reproduces the
single-phase backtest's holdings exactly at h=1, the other that a signal known at close `t`
cannot influence the weight held over `(t, t+1]`.

**Downstream, and it is not small.** Under the corrected primary specification **no setting of
eighteen has a net Sharpe interval excluding zero** (previously two, then one). The largest is
ETH GBM at h=7, 0.788 with interval [-0.007, 1.594]. No setting has a positive alpha to
buy-and-hold distinguishable from zero: the largest t-statistic on a positive alpha is 0.91,
and three settings are significantly *negative*. The break-even cost sweep, the exposure
regressions and the cost curve were all recomputed on the corrected specification.

The economic and statistical instruments now agree completely: nothing survives either.

### R22 — A third null generator, and a retracted attribution

**Was:** "almost all of the block column is genuine within-block predictability, not
distortion in the test."

**Why wrong:** not identified. Independent sign randomisation removes conditional mean
dependence, but `audit/scripts/garch_null.py` measures what else it removes: skewness falls
from -1.05 to -0.19, the leverage correlation from -0.071 to -0.003, and the autocorrelation
of return signs from -0.045 to +0.001. Any of those could move a finite-sample statistic.

**Now:** the claim is that the gap is *consistent with* dependence the block bootstrap retains
and sign randomisation removes, of which conditional mean dependence is one candidate among
several. What settles the question is a third null that shares neither's artifacts: a
GARCH(1,1) fitted to the same returns with the conditional mean held at the sample drift,
which reproduces volatility clustering (0.141 against 0.154) but almost none of the tail
thickness (excess kurtosis 0.63 against 18.8). All three agree -- Clark-West against the
recursive mean rejects 5.25%, 6.25% and 5.00%; the certificate 0.25%, 1.50% and 0.25%.

### R23 — Bootstrap resamples, and what an interval is conditional on

At 500 resamples the lower bound of a 95% Sharpe interval moves by up to 0.25 across five
seeds and one setting's reject-or-not verdict flips between them. At 10,000 the worst movement
is 0.042 and no verdict changes. The manuscript now reports 10,000-resample intervals, states
the seed stability, and states explicitly that these are conditional intervals for the
realised return sequence -- they price no part of model estimation, refitting, early stopping,
or the selection of a setting from the grid.

### R24 — Reviewer items closed without a claim change

Data provenance (composite venue, UTC cutoff, volume aggregation, revision policy, what
"adjusted" does and does not mean for spot crypto, and why a composite high or low is not
executable); the gradient booster's specification in full, including the tree cap, the
stopping patience, the validation slice and whether the best or final iteration is used; the
size table relabelled so that rows testing the stated null are marked size and rows testing a
different one are marked rejection under benchmark misspecification; the power curve scoped as
scenario-specific; the fold-level expanding mean distinguished from the fully recursive one.

## 2026-08-04 — The instrument replaces the diagnosis

Scope: the 2026-08-03 pass fixed what the paper *claimed*; this one changes what the paper
*is*. The benchmark problem is now diagnosed in closed form and then solved, rather than
diagnosed and reported. A new package, `alphacert`, implements the solution.

### R16 — Proposition 1: the distortion in closed form (NEW RESULT)

**Was:** the drift contamination of the zero-benchmark Clark–West statistic was shown to be
nonzero (Equation 12) and its magnitude measured by simulation, cell by cell.

**Now:** it has a closed form. For i.i.d. returns with per-period Sharpe `S`, an estimation
window of `k`, and a larger model estimating `p` coefficients that are truly zero,

    E[CW_0]  ~=  sqrt(n) S^2 / sqrt(S^2 + (1+p)/k)

which tends to `sqrt(n) S` when `k S^2 >> 1+p` — the statistic *converges to the t-statistic
of the asset's mean return*. Validated by simulation in `audit/scripts/certificate_calibration.py`:
predicted 1.341 against measured 1.348 at an annualised Sharpe of 0.8 with `p = 0`. Two
consequences the simulation alone did not make visible. First, a model containing **no
features at all** is declared significantly predictive in 43% of samples at BTC's realised
drift, and 97.5% at a Sharpe of 1.6. Second, the distortion *shrinks* as the model estimates
more parameters, because their estimation noise dilutes the drift signal — which is why the
study's regularised twelve-feature models measured 14–21% rather than 43%.

### R17 — The certificate (NEW METHOD)

**Was:** the paper diagnosed the benchmark and recommended using the recursive mean. That
leaves the residual term `-E[(mu_hat_t - mu) g_t]`, the covariance between the benchmark's
own estimation error and the model's signal, which is what the 7–8% measured size at h=7
consists of.

**Now:** the drift is eliminated rather than estimated. For each candidate drift `mu`, two
non-negative martingales are built — one betting that the signal leads the outcome, one
betting two-sidedly that `mu` is the wrong centre — and the certificate is the infimum over
`mu` of their average. An infimum is at most the value at the truth, the value at the truth
is a martingale, and Ville's inequality closes it:

    P( exists t : E_t >= 1/alpha )  <=  alpha

in finite samples, uniformly in time, for arbitrary black-box pipelines, with no bandwidth,
no long-run variance, no bootstrap, no asymptotics and no refitting. Measured rejection rate
on data with no predictability: 0.3–1.0% across annualised drift Sharpe ratios from 0 to 2.0,
against 5.2% to 81% for Clark–West vs zero.

Three properties follow that the previous instrument did not have.

- **Anytime-valid.** The process may be monitored daily on a live strategy with no correction
  for looking, and it yields a time-uniform *ceiling* on incremental value.
- **Denominated in P&L.** `log E_T` is the cumulative log return of an explicit drift-hedged
  strategy. Rejecting at 5% *is* a twentyfold multiplication of capital credited with none of
  the market's return. The "disjoint statistical and economic winners" problem cannot arise.
- **Composable.** E-values average under arbitrary dependence, so the 18-setting grid needs no
  joint bootstrap, and the h phase certificates of an overlapping forecast are combined by
  averaging — which is exactly the staggered h-vintage portfolio the economic section adopted
  independently.

### R18 — The detection-horizon law (NEW RESULT)

Evidence is log wealth and log wealth grows at `IR_p^2 / 2`, so

    T* = 2 ln(1/alpha) / IR^2  years   ~=  6 / IR^2  at 5%.

Optimal to first order by the sequential lower bound, so no better test recovers the missing
years. Continuous monitoring costs 3% more data than a fixed-sample test that may be looked at
once (5.99 vs 6.18 years at IR = 1). The drift hedge costs exactly `ln 2 = 0.69` nats — 23%
more data — because capital is split with the centring bettor. Learning the stake online
instead of declaring a target effect size costs a further factor of two to three. Simulated
median detection times match: 1.93 years observed against 1.84 predicted at IR = 2, 0.89
against 0.82 at IR = 3.

**Consequence for the study, and for the literature.** Six years of daily data cannot certify
an information ratio below **1.59** (1.11 with the stake pre-committed). The study's implied
ratios run 0.00 to 0.50. Reported Sharpe ratios in the applied cryptocurrency literature sit
in the 0.5–1.5 range and are rarely separated from market exposure; at those magnitudes the
claims are not merely unproven but unprovable at the sample sizes on which they rest.

### R19 — What the certificate says about this study

Nothing is certified. Largest e-value 1.81 against the 20 required; grid-level e-value 1.18;
e-BH selects nothing; the directional variant — the anytime-valid replacement for the averaged
Pesaran–Timmermann statistic, which was itself invalid — gives 1.07. The 95% anytime-valid
ceilings put the features' incremental annualised information ratio between 0.63 and 2.55,
every interval containing zero.

This does **not** retract the 7-of-18 Clark–West count against the recursive mean, whose joint
significance is reported separately. It reframes it: a p-value and an e-value are disagreeing
about what six years can settle, and R18 says which is right. The reportable number is the
ceiling.

### R20 — Reviewer items closed in code

- **Joint null.** The whole 18-test experiment is now simulated jointly
  (`audit/scripts/mc_joint_null.py`), giving P(N >= k), max-T and Romano–Wolf step-down
  adjusted p-values, rather than comparing an observed count to a sum of marginal
  probabilities.
- **Execution and schedule.** The primary economic specification is the staggered portfolio
  over all h daily vintages with one-bar delayed execution (`staggered_strategy`); same-close
  single-phase results are retained only as a labelled optimistic bound.
- **Sign timing.** The phase-averaged Pesaran–Timmermann statistic referred an average of
  dependent standardised statistics to a standard normal without a variance for the average.
  It is superseded by the sign certificate.


## 2026-08-03 — Literature gate applied to the manuscript; size study rebuilt

Scope: the 2026-07-30 pass fixed the *claims*; this one fixes the *framing* and the
*measurements the framing rests on*. Three of the four things the paper presented as its
contribution turned out to be published already, the size numbers turned out to be
under-replicated and computed against an unpinned sample, and the "no usable inference under
dependence" conclusion turned out to be an artifact of the wrong dependence null.

### R11 — The novelty framing (CRITICAL)

**Was:** "Benchmarking is the one item on the list this study initially got wrong, and
correcting it is the paper's main contribution." (`paper.tex` §Introduction, `README.md` §1.)

**Why wrong:** the literature gate of 2026-07-30 had already found the opposite, and the
manuscript had not been updated to reflect it. Two papers that the gate could not read were
retrieved on 2026-08-03 via a text proxy (MDPI returns HTTP 403 to direct requests) and
settle it:

- **Clark & West (2006)** state the null of the MSPE-adjusted test as a *zero-mean* martingale
  difference with the zero forecast as benchmark. A drift term entering the statistic is the
  test behaving as specified. The error is interpretive, not econometric.
- **Magner & Hardy (2022)**, *Mathematics* 10(13):2338, §2: "we compare the predictive ability
  of our models against a simple naive random walk process (we consider the first difference
  of the random walk, which is equivalent to a zero forecast (DRW) or constant forecast
  (RW))." They report **both** benchmarks, on 13 cryptocurrencies over 2018–2022, using **Wild
  Clark–West**. Reporting both benchmarks is established practice in this exact literature.
- **Pincheira, Hardy & Muñoz (2021)** already documented CW's long-horizon size distortion and
  published the fix.
- **Goyal & Welch (2008)** / **Campbell & Thompson (2008)** established the recursive mean as
  the standard benchmark ~20 years ago; **Moosa & Burns (2016)** argue the drift-vs-no-drift
  question directly.

**Now:** both documents carry an explicit "what is new here, and what is not" paragraph naming
all five sources, and a Related Work paragraph on nested predictive-accuracy testing. The
claimed contribution is narrowed to three things: a size calibration of the whole applied
pipeline including an adaptive learner (no prior measurement found), a measurement of WCW in
this regime (negative), and a worked account of four choices that each reversed a headline
finding. The title changed from *On the Out-of-Sample Predictability of Short-Horizon
Cryptocurrency Returns* to *Benchmark Choice and Measured Test Size in Nested Tests of
Cryptocurrency Return Predictability*.

**Verification levels** are recorded per row in `LITERATURE_AND_NOVELTY_MATRIX.csv`; the two
rows previously marked "403, not read" are now marked full-text-read with the quoted
benchmark specification.

### R12 — Wild Clark–West implemented and measured (NEW RESULT, negative)

**Was:** "a bootstrap on $\hat f_t$ rather than an analytic long-run variance is required for
a definitive count. We do not have one." The paper asked for something that had existed since
2021.

**Now:** WCW is implemented (`audit/scripts/wcw.py`, `mc_null_wcw.py`) exactly as specified —
$\hat f^W_t = \hat e^b_t(\hat e^b_t - \theta_t \hat e^m_t)$, $\theta_t \sim$ iid
$N(1, \phi^2)$, $\phi = c\,\mathrm{sd}(\hat e^m)$ for $c \in \{0.01, 0.02, 0.04\}$, smoothed
over $K = 2$ draws — and measured both on the study's forecasts and inside the Monte Carlo.

**Result: it changes nothing here, for two separate reasons.**

1. *Numerically.* Across all 36 setting–benchmark pairs the largest $|WCW - CW|$ is 0.0034 at
   $c = 0.04$ and 0.0019 at $c = 0.02$; no 5% decision changes under any $c$. In the Monte
   Carlo its rejection rate differs from CW's by ≤0.4 pp in every cell. The perturbation is
   scaled to the forecast *error* while the core statistic is scaled by the forecast
   *difference*; in this signal-to-noise regime the injected term is 0.5–5.1% of the core
   statistic's standard deviation.
2. *Structurally.* $\mathbb{E}[\theta_t] = 1$, so the drift term of the benchmark
   misspecification passes through untouched. WCW repairs a degeneracy in the reference
   distribution; it cannot repair a benchmark that encodes the wrong hypothesis.

The one setting where the $\theta$ draw matters is BTC ridge at $h=7$, whose $p$-value is
0.05004: the 5% decision goes to the model in 43% of draws.

### R13 — The size table, rebuilt (CRITICAL — supersedes R3)

**Was:** the table in R3 below: CW vs zero 14.2 / 22.0 / 22.0%, CW vs recursive mean
6.2 / 6.2 / 5.3% at $h=1$, ~10% at $h=7$, from 400 replications (150 for the booster).

**Why wrong:** two independent protocol failures.

1. **Unpinned sample.** Those runs used a BTC return pool of 2,768 bars ending 2026-07-30 —
   a cache miss from running in the wrong working directory, i.e. Issue 9 firing again. The
   *identical* script run from the repository root against the committed cache (2,755 returns
   ending 2026-07-18) gives materially different cells: CW vs recursive mean at $h=7$ moves
   from 10.2% to 5.0%.
2. **Too few replications.** At 400 replications, re-running with a different seed on the same
   pinned pool moved cells by up to 3 pp (e.g. CW vs recursive mean, ridge, $h=1$: 3.2% vs
   6.0%). The published numbers implied a precision the design did not have.

**Now:** 2,000 replications per linear cell and 600 for the booster, against the pinned cache,
with Monte Carlo standard errors reported in the table. Both retracted runs remain committed
as `mc_null_h{1,7}_b1.csv` so the record is checkable; the seed replicates are
`mc_null_wcw_h{1,7}_b1_s11.csv`.

**Downstream:** every quoted size number, the expected-rejection arithmetic, and the abstract,
discussion and conclusion of both documents.

### R14 — The dependence null was the wrong null (CRITICAL — reverses a stated conclusion)

**Was:** "under dependence-preserving resampling it is not trustworthy at any horizon", from
block-21 resampling giving 28–43% rejection against the recursive mean.

**Why wrong:** block resampling preserves volatility clustering *and* whatever genuine
within-block predictability the real return series contains. Its rejection rate is an upper
bound on size, not a measurement of it — the changelog entry for R3 said so, and the
manuscript then read it as a size failure anyway.

**New experiment** (`mc_null_wcw.py --signflip`): block-resample, then randomise the sign of
each deviation from the sample mean, $r_t = \mu + s_t(r^*_t - \mu)$ with $s_t = \pm 1$
independent. $|r_t - \mu|$ is untouched bar for bar, so the volatility path survives exactly;
$\mathbb{E}[r_t \mid \mathcal{F}_{t-1}] = \mu$ by construction, so no conditional mean
predictability remains. This is the null with realistic dependence *and* a true hypothesis.

Rejection at nominal 5%, 2,000 replications, BTC pool:

| statistic | h=1 indep | h=1 sign-flip | h=1 block | h=7 indep | h=7 sign-flip | h=7 block |
|---|---|---|---|---|---|---|
| CW vs zero, ridge / EN / GBM | 12.0 / 17.5 / 16.8 | 14.1 / 20.9 / 20.0 | 53.2 / 43.9 / — | 22.1 / 23.4 / — | 28.6 / 30.4 / — | 25.7 / 25.1 / — |
| CW vs recursive mean, ridge / EN / GBM | 3.7 / 3.1 / 3.7 | 4.8 / 3.5 / 3.0 | 44.6 / 30.6 / — | 5.9 / 5.7 / — | 7.2 / 6.8 / — | 10.7 / 9.5 / — |
| CW vs zero, drift-only model | 27.5 | 34.4 | 25.2 | 36.8 | 41.9 | 33.3 |

**Now:** the corrected test *is* usable under realistic dependence — 3.0–4.8% at $h=1$ and
6.8–7.2% at $h=7$. Almost all of the block column was genuine within-block predictability.
The calibration transfers across assets: on a SOL return pool (shorter, ~2× the volatility
and ~2× the drift) CW vs the recursive mean is 3.8/3.2% at $h=1$ and 7.5/7.3% at $h=7$.

### R15 — The study's conclusion, revised in the other direction (CRITICAL)

**Was:** "no predictability claim survives a correctly benchmarked and size-calibrated test";
"against calibrated size the observed counts are within what the null produces."

**Why wrong:** that was true when the recursive-mean benchmark was believed to have 6–10%
size. It does not have. With calibrated size of 3–8%, the expected rejection count over the
family of 18 is **about 1**, and 7 are observed. The excess is real and the previous framing
buried it.

**Now:** both documents state that the two benchmarks support opposite readings of identical
forecasts — 5 of 18 against ~4 expected under the zero benchmark (uninformative), 7 of 18
against ~1 expected under the recursive mean (a real excess) — and then bound how much the
excess is worth: 2 settings survive FDR control and they are two parametrisations of one
linear model on one asset at one horizon ($p_{BH} = 0.035$), none survives FWER control
($p_{Holm} = 0.069$), the surviving set moves with the HAC bandwidth (2 vs 5), and the same
settings have negative $R^2_{OS}$, no sign-timing skill and no alpha net of costs, execution
timing and market exposure. The stated conclusion is now "one weak, uneconomic candidate",
not "nothing".

### R16 — The pipeline now produces the paper's headline (CRITICAL — was a contradiction)

**Was:** `study.py` computed Clark–West against `PRIMARY_BENCHMARK = "random_walk"` only, and
`pesaran_timmermann` on the full overlapping series. So the generated appendix table
(`paper/tables/accuracy.tex`) carried a single CW column against the **zero** benchmark and the
**inflated** PT statistic — both of which the manuscript retracts in the body. A referee turning
to Table 3 would have found it contradicting the abstract, and every recursive-mean number in
the paper came from `audit/` scripts rather than from the study.

**Now:**

- `ModelRun` carries `cw_stat_vs_mean` / `cw_p_vs_mean` alongside the zero-benchmark pair;
  `accuracy_table` renders both columns, headed "CW vs. 0" and "CW vs. mean".
- `pesaran_timmermann_non_overlapping(y, yhat, horizon)` slices to `y[k::h]` for each phase and
  averages, matching `audit/scripts/pt_and_exec.py`; `study.py` calls it instead of the raw
  version. Its docstring records that the averaged p-value is a cross-phase summary, not a
  combined test.
- `StudyConfig.end` is pinned to `2026-07-18` instead of `None` → today. This closes Issue 9 at
  the source rather than relying on the cache being present.

Regenerated end to end (`make backtest && make tables && make paper`) against the pinned
sample. Every headline reproduces from the pipeline: R²_OS negative in 16 of 18, DM rejecting
in favour of the benchmark in 7 of 18, CW vs zero 5 of 18, CW vs recursive mean 7 of 18 with
p = 0.0038 / 0.0039 on the two BTC h=1 survivors. Point estimates moved by ≤6e-3 (Sharpe),
≤2e-3 (CW), ≤3e-5 (R²_OS) against the previous `reports/results.csv`; `n_oos` moved by one bar
from the explicit end date. The manuscript's sign-timing table was refreshed to the
regenerated values (ETH elastic net −1.28 / 0.242, ETH ridge −1.28 / 0.265, SOL ridge −0.67 /
0.604, BTC ridge +0.54 / 0.596).

Test suite 157 → 164. New tests pin the phase-averaging behaviour, the presence of both CW
columns in the rendered table, the two CW fields in `results.csv`, and the pinned end date.

### R17 — The four disclosed gaps, closed (MAJOR)

Every item the 2026-08-03 draft listed as "open, disclosed in the paper" was measured rather
than left as a caveat.

**Minimum detectable effect.** "Not powered to exclude an economically relevant effect" is not
a statement until the effect is named. `mc_null_wcw.py --rho R` passes the sign-flipped
deviations through an AR(1) filter, `r_t - mu = R (r_{t-1} - mu) + sqrt(1 - R^2) d_t`, which
holds the unconditional variance fixed so the optimal one-step predictor's population `R^2` is
exactly `R^2`; the one-bar return is already in the feature set. Power at h=1 against the
recursive mean, 500 replications per cell:

| rho | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.11 | 0.12 |
|---|---|---|---|---|---|---|---|
| population R² | .0004 | .0016 | .0036 | .0064 | .0100 | .0121 | .0144 |
| ridge | 5.2% | 11.4% | 24.0% | 42.8% | 66.2% | 75.0% | 83.6% |
| elastic net | 4.8% | 14.0% | 31.4% | 55.2% | 75.8% | 83.8% | 90.4% |

**MDE = population R² of 1.34% at 80% power** (rho = 0.116), about four times the largest
`R²_OS` anywhere in the study. Both directions are now stated: the null result says nothing
about effects below ~1% of return variance, and an effect that small would be eaten by the
cost model anyway.

**Cost curve.** `audit/scripts/cost_curve.py` sweeps the per-side charge over the saved
forecasts (no refits). Settings with positive net Sharpe: 16/18 at 0 bp, 13/18 at 17 bp, 11/18
at 25 bp, 9/18 at 40 bp, 6/18 at 100 bp. Median break-even 31 bp — roughly twice the assumed
charge, so the economic conclusion is not delicate at the margin. The sweep also re-derives the
exposure finding from a new direction: the settings tolerating the highest costs are the ones
that barely trade (ETH GBM h=1 breaks even at 148 bp and is 94% long), so cost tolerance here
tracks turnover rather than forecast quality.

**ETH size.** Calibration now covers all three assets rather than BTC and SOL. CW against the
recursive mean under the sign-flipped null: 3.5–4.8% (BTC), 3.7–4.5% (ETH), 3.2–3.8% (SOL) at
h=1; 6.8–7.2 / 6.9–7.5 / 7.3–7.5% at h=7. The recursive-mean calibration is stable across
assets whose daily volatilities differ by a factor of two. The zero-benchmark distortion is
not, and Equation (drift) says it should not be: 14.1–20.9% on BTC against 9.7–12.1% on ETH
and SOL, because it scales with drift relative to noise.

**Figure.** The calibration was the paper's main contribution and had no figure.
`audit/scripts/plot_size.py` draws Figure 9 from the Monte Carlo cells, so it cannot drift from
Table 2.

### R18 — Writing (MODERATE)

Applied the outstanding items from `AI_SLOP_AUDIT.md`: the Pesaran–Timmermann, PSR and
expected-maximum-Sharpe expressions moved to an appendix that also states WCW (§6 of that
audit); the recurring setup–pivot–aphorism construction removed from six places (§3); the
limitations section rewritten so every item states what would change if it were repaired
(§7); "deliberately constrained gradient booster" and "not a formality" replaced by the
hyperparameters and the measured spread (§4); title changed to name the contribution (§8).

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
