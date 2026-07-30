# Acceptance readiness

## Reproduction status

The study reproduces exactly. Environment: macOS 25.0.0 (darwin), Python 3.13 in `./venv`,
xgboost 3.0.4, scikit-learn 1.7.1. Data: `data/cache/*.parquet`, four content-addressed frames,
BTC/ETH 2019-01-01→2026-07-18 (2 756 bars), SOL 2020-04-10→2026-07-18 (2 291 bars), Yahoo Finance
`auto_adjust=True`. Seed 7 throughout.

Verified identical to `reports/results.csv` and to the manuscript: 154 tests collected; DM rejects in
7 of 18 with the sign favouring the benchmark; CW raw rejections 5 of 18; BH survivors 2, both
p_BH = 0.0458; Holm survivors 0; `n_boot = 500`; R²_OS −0.0079 (BTC ridge) and +0.0031 (BTC elastic
net). No transcription errors were found between `reports/results.csv`, `paper/tables/*.tex`, and the
manuscript text.

The failure is not in the arithmetic or the engineering. It is in what the numbers mean.

## Simulated review panel

Scores on a 1–10 scale, 6 = weak accept.

**Reviewer A — financial econometrician. Score 2, confidence 5/5.**
Strongest case for acceptance: the leakage protocol is better than most published applied work, and
the study is fully reproducible. Strongest case for rejection: the paper's entire inferential
apparatus tests the wrong null. The zero-forecast benchmark is not the slopes-zero restriction of a
regression with a fitted intercept — the authors' own `historical_mean` row shows CW = +1.03 with no
features, which should have stopped the study. The validating simulation has a false null and no
estimation in it. Bandwidth h−1 is zero lags at the horizon where four of five rejections occur.
Required: recursive-mean benchmark, calibrated size, dependence-robust inference, non-overlapping
sign tests. Question: what is the estimand?

**Reviewer B — machine-learning researcher. Score 3, confidence 4/5.**
For: honest negative result, real reproducibility, correct treatment of temporal leakage. Against: no
ML contribution. Three model families with untuned, unjustified hyperparameters; twelve hand-made
technical features; three assets. No nested model selection, so no conclusion about a model *family*
is supported — only about `Ridge(alpha=1)`. "The contribution is an evaluation protocol" is the right
instinct, but the protocol on offer is an assembly of existing procedures and the novel-looking part
of it is the part that is wrong. Missing: ablations, positive controls, minimum detectable effect,
any baseline stronger than AR(1).

**Reviewer C — quantitative researcher. Score 3, confidence 4/5.**
For: costs are modelled, turnover is charged, buy-and-hold carried explicitly, phase sensitivity
reported — all rare and all correct. Against: entry is at the same close the features are computed
from, so the h = 1 economic results are not implementable; delaying one bar moves Sharpes by 0.27–0.74
and flips signs. 17 bp is one scenario presented as realism, with no cost curve and no break-even
cost. Shorts are frictionless. Data is a vendor composite, not an execution venue. The exposure claim
is right but asserted; β to buy-and-hold and an alpha t-statistic would have proved it in one table.

**Reviewer D — reproducibility reviewer. Score 5, confidence 5/5.**
Best-scoring dimension. Single-command pipeline, code-generated tables, content-addressed data cache,
154 tests including genuine leakage-perturbation tests. Two defects: `end=None` means a cache miss
silently re-dates the study, and per-observation forecasts are never persisted, so no third party can
re-analyse the inference without re-running the models. Add a run manifest with data hashes, commit
the forecasts, pin the end date.

**Reviewer E — area chair. Score 2, recommend reject.**
Two of the paper's headline claims are mathematically false, one reported p ≈ 0.001 is really
p ≈ 0.25, and the central "5 of 18 against 0.9 expected" comparison uses an expectation that is wrong
by a factor of 3–4. Any one of these is disqualifying at a venue that reviews competently. The
craftsmanship is real and the authors have obvious ability, which makes this more frustrating rather
than less: the audit that would have caught all of it is the audit the paper claims to have
performed. Separately, even with every flaw repaired, "three assets, two horizons, three off-the-shelf
model families, one vendor, one frequency" is not enough empirical scope for a top venue, and the
contribution as framed is not one.

**Synthesis: strong reject, unanimous on the critical issues.** No reviewer's concerns overlap enough
to be dismissed as taste; A, B, C and E each independently identify a different disqualifying problem.

## Gate status

| gate | status | blocker |
|---|---|---|
| Scientific validity | **FAIL** | Wrong estimand (Issue 1); false simulation claim (2); uncalibrated test (3); invalid sign inference (4) |
| Novelty | **FAIL** | Contribution is an assembly of standard procedures. And unverified — no literature review was done |
| Empirical breadth | **FAIL** | 3 assets, 2 horizons, 1 vendor, 1 frequency, untuned hyperparameters, no ablations, no controls, no power analysis |
| Reproducibility | **PASS with defects** | Unpinned end date; forecasts not persisted |
| Economic realism | **FAIL** | Same-close execution; single cost scenario; frictionless shorts |
| Writing | **FAIL** | See `AI_SLOP_AUDIT.md` — claims of verification substituting for verification |
| Venue fit | **FAIL** | Positioned as ML-methodology; is an applied-finance audit |

Zero of seven gates pass cleanly.

## Honest answers to the ten closing questions

1. **Central contribution of a revised paper?** Not yet fixed. The best available candidate, from the
   audit's own measurements: the rejection set of the standard nested OOS predictability test is
   determined jointly by benchmark, bandwidth and execution convention, and one widely used
   configuration is 3–4× oversized. See `RESEARCH_PIVOT_MEMO.md`.
2. **Why novel?** *Unverified.* No literature review was performed and none was invented. This is the
   first gating task, not a formality — the result may be known in the econometrics literature even
   though applied ML papers keep making the error.
3. **Why important?** If it holds, it applies to every paper that races a conditional model against a
   random walk, which is most of the return-predictability literature.
4. **Evidence?** Measured: intercept identity to machine precision; closed-form drift contamination
   `f_t = 2(y−b)(m−b)`; size 14–22 % vs 5.3–6.2 % across ridge, elastic net and early-stopped XGBoost
   (400/150 reps); rejection-set membership changing from BTC-concentrated to SOL-concentrated;
   sign-test p 0.001 → 0.25; Sharpe deltas 0.27–0.74 under one-bar delay.
5. **Assumptions remaining?** The Monte Carlo null is a bootstrap of one asset's returns; size is not
   established for other assets, other frequencies, other estimators, or genuine
   volatility-clustering nulls (where the corrected test rejects 28–43 %).
6. **What could falsify it?** A dependence-robust test that is correctly sized *and* still yields the
   paper's rejections. A literature finding that the benchmark distinction is already settled. Size
   results that do not replicate on the Goyal–Welch equity data.
7. **Why would ICAIF reviewers care?** They referee papers using this exact test every cycle, and a
   calibrated recipe plus a released harness is directly usable.
8. **Three strongest remaining rejection arguments?** (i) novelty unverified and plausibly thin; (ii)
   empirical scope still narrow unless the equity-premium replication is added; (iii) the paper would
   be reporting a null result about crypto *and* a corrective result about a test — two contributions
   competing for one centre, and the crypto half must be demoted to an example.
9. **Every critical flaw resolved?** No. The four CRITICAL issues are diagnosed and quantified;
   nothing in `src/`, `paper/`, `tests/` or `reports/` has been changed. All four remain live in the
   committed artifacts.
10. **Caliber?** **Not yet publishable** in its current form — strong reject at any venue that reviews
    competently. With Tier-1 repairs only: workshop-level. With Pivot A fully executed, including the
    dependence-robust test, the size/power surface and the equity-premium replication: a plausible
    ICAIF accept. **Not** strong-accept caliber on any current evidence, and I will not claim it is.

## Next actions, in order

0. Tag the current state (`git tag audit-baseline-83881c7`) before touching anything.
1. Literature review and the novelty gate decision (`RESEARCH_PIVOT_MEMO.md`, final section). Do not
   write prose before this.
2. Fix `tests/test_stats.py:103` and remove the corresponding claims from `paper/paper.tex` and
   `README.md`. This is a correctness fix and is independent of the pivot.
3. Pin `StudyConfig.end`; commit `forecasts.csv` and a run manifest with data hashes.
4. Switch `PRIMARY_BENCHMARK` to the recursive mean, or add it as a second benchmark and report both.
5. Promote `audit/scripts/mc_null.py` into `src/` as a first-class experiment; extend to a size/power
   surface with ≥2 000 reps and reported Monte Carlo error.
6. Add block-bootstrap inference on the loss differential and show it holds size where the analytic
   HAC version does not.
7. Non-overlapping sign inference across all phases; one-bar execution delay as the primary
   specification; cost curve; exposure decomposition table.
8. Only then rewrite the manuscript, in the voice identified in `AI_SLOP_AUDIT.md` §9.
