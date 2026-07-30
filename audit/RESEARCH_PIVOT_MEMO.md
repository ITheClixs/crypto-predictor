# Research pivot memo

## The problem with the current framing

The paper's stated contribution is "an evaluation protocol applied honestly". After the audit, the
protocol's leakage layer is genuinely good and its inference layer is broken (FORENSIC_REVIEW Issues
1–7). So the current framing is the worst of both: it claims credit for rigour in the exact layer
that fails, and it spends its empirical budget on three assets whose predictability is not, and was
never going to be, the interesting question.

But the audit produced something the paper does not have: a *measured* fact of general relevance.

> The canonical out-of-sample predictability test — a conditional model against a random-walk
> benchmark, Clark–West corrected, HAC bandwidth h−1 — rejects **14–22 %** of the time at a nominal
> 5 % on data with no predictability, because the zero-forecast benchmark makes drift part of the
> alternative. Replacing the benchmark with the recursively estimated mean brings size to 5.3–6.2 %
> at h = 1 (including for early-stopped XGBoost), but not at h = 7 (≈10 %), and not under
> dependence-preserving resampling (28–43 %).

That is not a crypto fact. It is a fact about a test used in hundreds of return-predictability
papers, and it is the seed of a defensible contribution.

## Candidate directions

### Pivot A — Size calibration and benchmark fragility of nested out-of-sample predictability tests

**Claim.** The rejection set in the standard OOS predictability protocol is not identified by the
data; it is jointly determined by three choices the applied literature makes by default — the nested
benchmark (zero vs recursive mean), the HAC bandwidth, and the execution-timing convention — and at
least one common configuration is 3–4× oversized.

**Evidence already in hand.** All of Issues 1, 3, 6, 7. The rejection set changes membership
entirely (BTC-concentrated → SOL-concentrated) when only the benchmark changes. Calibrated size for
ridge, elastic net and XGBoost. Execution-delay sensitivity: every h = 1 Sharpe moves 0.27–0.74.

**What must be added.**
1. Extend the Monte Carlo to a full size/power surface: predictor persistence, heteroskedasticity,
   heavy tails, overlapping horizons, refit frequency, expanding vs rolling, regularisation strength,
   sample size. 2 000+ replications per cell with reported Monte Carlo error.
2. A dependence-robust alternative that *is* correctly sized — block bootstrap or wild bootstrap of
   the adjusted loss differential — demonstrated to hold size where the analytic HAC version does
   not. This is the deliverable a referee can use.
3. **External validity at near-zero cost:** run the identical protocol on the Goyal–Welch equity
   premium dataset (public, monthly, 1926–present, the canonical benchmark for exactly this test).
   If the benchmark-fragility result reproduces there, the paper is no longer about crypto and the
   "why should anyone care" question answers itself.
4. Specification surface over the defensible configurations, with the rejection count as the
   outcome.

**Scores** (1–5): novelty 3.5 · depth 4 · significance 4 · feasibility **5** · data availability 5 ·
compute 4 · reproducibility 5 · risk of failure **low** · time ~4–6 weeks · venue fit 4.5.

**Why it works.** It reuses the existing codebase almost unchanged. It produces knowledge whether or
not crypto is predictable. It is falsifiable, it is a tool, and the negative crypto result becomes a
worked example rather than the point.

### Pivot B — A leakage-and-false-predictability benchmark suite for financial time-series ML

**Claim.** Ship a standardised battery of positive and negative controls (timestamp contracts,
preprocessing-leakage probes, overlapping-label probes, adaptive-validation probes, benchmark
hierarchy, injected-signal power tests) plus reference datasets, and evaluate published-style
pipelines against it.

**Scores:** novelty 4 · depth 3 · significance 4 · feasibility 3 · risk **medium-high** · time
~3 months. Benchmark papers are a recognised contribution type (NeurIPS Datasets & Benchmarks), but
a benchmark nobody adopts is worth nothing, and adoption is not something a first paper controls. The
repo already has three of the probes (look-ahead perturbation, purged early-stopping corruption,
missing-bar purge) — that is 20 % of the work, not 80 %.

**Verdict:** strong as a *component* of Pivot A (the positive/negative control harness), weak as the
central contribution of this paper.

### Pivot C — Cross-sectional prediction on a point-in-time crypto universe

**Claim.** Replace six isolated time-series tasks with a delisting-aware cross-sectional ranking
problem: ex-ante inclusion rules, dead tokens retained, liquidity and market-cap filters, listing
dates, multi-exchange validation.

**Scores:** novelty 3.5 · significance 4.5 · feasibility **2** · data availability **1.5** · risk
**high** · time 3–6 months. This is the best *machine-learning* framing of the six candidates and the
one that most directly answers the survivorship criticism. It is also the one that cannot be built
from yfinance. Genuine point-in-time crypto universe data with delisted assets means a paid vendor or
months of exchange-API archaeology, and getting it subtly wrong reintroduces exactly the survivorship
bias the pivot exists to remove.

**Verdict:** the right second paper. Not this one.

### Pivots D–G (considered, rejected for this paper)

- **D. Alpha decomposition** (drift / exposure / timing / vol-scaling / selection). Issue 10 already
  does the first cut and it works. Too small to carry a paper; fold into Pivot A as the economic
  section.
- **E. Distributional / conformal forecasting.** Interesting and fashionable, but it changes the
  target without fixing the inference problem, and it would abandon the one measured finding.
- **F. Online decay detection / sequential testing.** Needs a signal to monitor. There isn't one.
- **G. More models (sequence models, deep nets).** Would add trials to an uncalibrated test. Actively
  harmful until Pivot A is done.

## Recommendation

**Primary: Pivot A. Secondary: the control harness from Pivot B, as Pivot A's validation layer.**
The crypto study becomes the worked example in Section 5, honestly reported, with its rejection
count presented as bandwidth- and benchmark-dependent rather than as a finding.

Proposed title direction — the contribution, not the domain:

> *What the random-walk benchmark actually tests: size distortion and benchmark fragility in nested
> out-of-sample return predictability tests*

## Venue

Assessed on scope fit, not prestige.

| venue | fit | why |
|---|---|---|
| **ACM ICAIF** | **primary** | Audience is exactly forecast-evaluation-literate finance ML. Values calibration and reproducibility work. Tolerates negative empirical results with a methodological contribution. |
| NeurIPS Datasets & Benchmarks | secondary | Only if Pivot B is fully built into a released, documented benchmark. Not otherwise. |
| *Journal of Financial Econometrics* / *Journal of Forecasting* | strong alternative | A size-calibration result with an equity-premium replication is squarely in scope, and the review will be more competent than at an ML venue. Slower. |
| NeurIPS / ICML / ICLR main track | **no** | No general ML contribution. Would be desk-rejected or scored low on originality regardless of execution quality. |
| KDD applied track | marginal | Possible, but the contribution is inferential rather than algorithmic or scale-driven. |

---

# GATE RESULT (2026-07-30): Pivot A as framed does not clear the novelty bar

The literature review below was run after the sections above were written. It changes the
recommendation. Evidence: `LITERATURE_AND_NOVELTY_MATRIX.csv`.

**Three of Pivot A's four claimed contributions are already in the literature.**

1. **"The zero benchmark makes the test oversized" — not an econometric finding.**
   Clark & West (2006) state the null explicitly as a **zero mean** martingale difference, with the
   "no change" (zero) forecast as the benchmark. Under a nonzero drift that null is *false*, so the
   test rejecting 14–22 % of the time is the test behaving correctly against a null the applied user
   did not intend. Our result is real and worth stating, but it is an **interpretive error in applied
   practice**, not a defect in the test. The framing "the standard test is not correctly sized" is
   wrong and must be dropped; the correct framing is "applied papers test a null they do not mean".

2. **"Use the recursively estimated mean instead" — standard practice since 2008.**
   Goyal & Welch and Campbell & Thompson established the recursive historical mean as *the* benchmark
   for out-of-sample return predictability. This codebase already uses it for `R²_OS`
   (`study.py:28`). The prescription is not new; the project simply failed to apply its own benchmark
   consistently.

3. **"CW is oversized at long horizons and needs a dependence-robust alternative" — solved in 2021.**
   Pincheira, Hardy & Muñoz, *"Go Wild for a While!"* (Mathematics 9(18):2254) document that CW
   "may present severe size distortions at long horizons" and propose the **Wild Clark–West (WCW)**
   statistic, which is well-sized at long horizons. Our h = 7 finding (≈10 % at nominal 5 %)
   replicates theirs. Our recommendation to build a bootstrap on `f_t` should be replaced with:
   *adopt WCW*.

4. **Benchmark choice changes conclusions — published in 2016.**
   Moosa & Burns, *"The random walk as a forecasting benchmark: drift or no drift?"* (Applied
   Economics 48(43)). Different question — point-forecast accuracy, not test size — so this is
   partial overlap, not a kill.

**And there is a same-asset-class, same-test-family competitor.** Magner & Hardy (2022, Mathematics
10(13):2338) test the random-walk hypothesis on 13 cryptocurrencies over 2018–2022 using WCW/ENC-t
with rolling windows, and report that models *do* significantly outperform the random-walk benchmark.
MDPI returns HTTP 403, so **their benchmark specification was not verified**. This is the single
highest-priority read:

- If they use a **drift-inclusive** benchmark, our central point is already standard in crypto and
  Pivot A is finished.
- If they use a **zero** benchmark, their positive result is a direct target for our critique, and
  that re-opens a narrow but real contribution.

## What actually survives

Only one item found no prior art, and the search was not exhaustive so this is weak evidence of
absence:

- **Size calibration of Clark–West-family tests for adaptive ML learners** — greedy split selection,
  subsampling, early stopping, data-dependent model complexity. Our measurement (5.3 % for
  early-stopped XGBoost against the recursive mean, versus 22.0 % against zero, h = 1) appears to be
  new. It is also *small*: one estimator, one configuration, one null, one asset's return
  distribution.

Plus one thing that is not novel but is useful: the **joint demonstration** that four routine choices
— benchmark, HAC bandwidth, label overlap, execution timing — each independently flipped a headline
result in a real, competently engineered study. That is a cautionary-tale contribution, and its value
depends on the honesty of the worked example rather than on methodological originality.

## Revised recommendation

**Pivot A is dead as a methods paper.** Two live options:

**A′ — Calibration of nested predictive-accuracy tests for machine-learning forecasters.**
Narrow the claim to the one thing with no found prior art. Required work: extend the size/power
surface across estimators (ridge, elastic net, random forest, boosting with and without early
stopping, and a small neural model), horizons, dependence structures, and refit schemes; include WCW
alongside CW as the incumbent; replicate on the Goyal–Welch equity premium data for external
validity. Honest ceiling: a solid ICAIF or *Journal of Forecasting* paper. **Not** a strong accept,
and the contribution is "we measured whether an existing test survives modern learners", which is
useful and unglamorous.

**B′ — The leakage-and-false-predictability benchmark suite** (Pivot B, promoted).
The novelty gate that killed A′'s bigger claims does not apply here, because a benchmark's
contribution is the artifact, not the theorem. The repo already contains three working probes
(look-ahead perturbation, purged early-stopping corruption, missing-bar purge), and this audit
produced four more that generalise (intercept/benchmark mismatch, label-overlap inflation,
execution-timing sensitivity, calibrated-size-under-resampled-null). That is seven, and each is a
concrete failure a real published pipeline can be tested against. Target: NeurIPS Datasets &
Benchmarks. Risk: adoption, which no first paper controls.

**Recommended: B′, with A′'s calibration harness as one component of the suite.**
The audit's most transferable output is not the crypto result and not the CW size number — it is the
set of cheap checks that caught them. That is a benchmark, not a theorem, and it is the one framing
where "we found our own paper's four errors" is a strength rather than an embarrassment.

**Before committing:** read Magner & Hardy (2022) and Pincheira & Hardy (2022, Mathematics 10(2):228)
in full. Both were blocked by HTTP 403 in this pass. Neither is cited anywhere in this project's
`references.bib`, and the second's title —*"A Simple Out-of-Sample Test of Predictability against the
Random Walk Benchmark"* — is close enough to the topic that it could moot A′ entirely.

---

## The gap as it stood before the gate was run

**No literature review was performed in this audit, and none should be fabricated.** Novelty for
Pivot A is *unverified*. The size distortion of MSPE-based nested tests is a studied problem — Clark
& West themselves, Clark & McCracken, and the Goyal–Welch/Campbell–Thompson exchange are all directly
adjacent, and there is a real possibility that the benchmark-choice result is known in the
econometrics literature even if applied ML papers keep making the mistake.

This is the **first** task before any writing, and it is a gating decision, not a formality:

1. Read Clark & West (2006) and (2007), Clark & McCracken (2001, 2005), Goyal & Welch (2008),
   Campbell & Thompson (2008), Goyal, Welch & Zafirov (2024) — primary sources, not summaries.
   Establish precisely what is already known about size under the zero vs mean benchmark.
2. Search ICAIF, KDD applied, and *Journal of Forecasting* 2019–2026 for calibration studies of
   nested OOS tests applied to machine-learning forecasters.
3. Build `LITERATURE_AND_NOVELTY_MATRIX.csv` with one row per closest paper: venue, year, problem,
   dataset, benchmark, inference method, what it establishes, what it leaves open.
4. Decide honestly: if the benchmark result is known, the contribution narrows to *calibration for
   adaptive ML forecasters under realistic dependence* plus the fragility surface — still publishable
   at ICAIF, but a smaller claim that must be stated smaller.

Do not write the introduction before step 4.
