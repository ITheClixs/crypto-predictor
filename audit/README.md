# Audit artifacts

Forensic audit of `paper/paper.tex` at commit `83881c7`. Every number in the reports below was
computed by a script here, not read off the manuscript. **Nothing in `src/`, `paper/`, `tests/` or
`reports/` was modified** — this directory is diagnosis only.

## Read in this order

| file | what it is |
|---|---|
| `FORENSIC_REVIEW.md` | The audit. Ten issues, four CRITICAL, each with the code path, the derivation, the measured consequence, and the repair. |
| `CLAIM_EVIDENCE_LEDGER.csv` | The paper's 15 headline claims, each traced to evidence and marked TRUE / FALSE / UNSUPPORTED. |
| `ACCEPTANCE_READINESS.md` | Reproduction status, five simulated referee reports, gate table, honest caliber assessment, ordered next actions. |
| `RESEARCH_PIVOT_MEMO.md` | Seven candidate directions, scored; **the literature gate result**, which killed the original recommendation; two revised routes. |
| `LITERATURE_AND_NOVELTY_MATRIX.csv` | Closest prior work, with per-row verification level. Two decisive papers were blocked by HTTP 403 and are marked unread. |
| `AI_SLOP_AUDIT.md` | Passages where claimed rigour substitutes for demonstrated rigour, with replacements. |
| `CHANGELOG_RESEARCH.md` | What was retracted from `paper/` and `README.md` on 2026-07-30, why, and what replaced it. |

## Headline

The paper's central quantitative claim — five of eighteen settings reject no-predictability against
0.9 expected from noise — does not survive. The test used to produce those rejections rejects
**10–30 %** of the time on data containing no predictability, so the correct comparison is 5 observed
against ≈4 expected. Two further claims are mathematically false (the nesting claim, and the claim
that the validating simulation holds the null by construction), and the paper's only significant
sign-timing results (p ≈ 0.001) become p ≈ 0.24–0.27 once overlapping labels are handled.

The engineering and the leakage discipline are sound. All headline numbers reproduce bit-for-bit.

**Two things reversed on 2026-08-03, after this audit was first written.** (1) Against the
*recursive-mean* benchmark the test is approximately correctly sized — 3–5 % at h=1, 7–8 % at h=7,
under a sign-flipped null that keeps the volatility clustering — so the expected count is ≈1 and the
7 of 18 observed is a **real excess**, not noise. The audit's original "nothing survives calibrated
size" reading was itself an artifact of an under-replicated size estimate. (2) The 28–43 % figure
under block resampling was almost entirely genuine within-block predictability, not size failure.
See `CHANGELOG_RESEARCH.md`, entries R13–R15.

## Reproducing the audit

**Run every command from the repository root.** `DEFAULT_CACHE_DIR` is the relative path
`data/cache`, so running from anywhere else misses the cache and silently re-downloads to *today* —
this is Issue 9 in practice, and it was reproduced by accident while writing this README: the same
script produced 2 186 forecasts per setting from the repo root and 2 198 from `audit/`.

Requires the committed `data/cache/` (sample pinned at 2026-07-18) and `./venv`.

```bash
PY=./venv/bin/python

$PY audit/scripts/gen_forecasts.py   # 72,900 OOS forecasts -> audit/forecasts.csv (the repo never saves these)
$PY audit/scripts/sim_audit.py       # Issue 2: the paper's CW simulation tests a false null
$PY audit/scripts/retest.py          # Issue 1, 6: CW vs zero and vs recursive mean, three HAC bandwidths
$PY audit/scripts/pt_and_exec.py     # Issue 4, 7: non-overlapping sign tests; one-bar execution delay
$PY audit/scripts/exposure.py        # Issue 10: beta to buy-and-hold and alpha t-statistics
$PY audit/scripts/wcw.py             # Wild Clark-West on the saved forecasts, three phi values
$PY audit/scripts/cost_curve.py      # net Sharpe vs per-side cost; break-even cost per setting

# Issue 3 -- the experiment the paper never ran, now with WCW alongside CW.
# ~70 min for the four linear cells run in parallel; the booster cell is the long pole.
$PY audit/scripts/mc_null_wcw.py 1 1  2000   # h, bootstrap block (1 = iid), replications
$PY audit/scripts/mc_null_wcw.py 1 21 2000
$PY audit/scripts/mc_null_wcw.py 7 1  2000
$PY audit/scripts/mc_null_wcw.py 7 21 2000
$PY audit/scripts/mc_null_wcw.py 1 1  600 --gbm   # early-stopped XGBoost configuration
$PY audit/scripts/size_table.py      # assembles the paper's size table from those cells
$PY audit/scripts/plot_size.py       # Figure 9: measured size under all three nulls

# Power. --rho R makes the optimal predictor's population R^2 exactly R^2, so sweeping it
# gives the minimum detectable effect rather than a vague "not powered to exclude".
for R in 0.02 0.04 0.06 0.08 0.10 0.11 0.12; do
  $PY audit/scripts/mc_null_wcw.py 1 21 500 --signflip --rho $R
done
$PY audit/scripts/power_table.py     # power curve + the 80%-power effect size
```

**Run count and sample pinning both matter, and the first pass got both wrong.** The Monte Carlo
outputs committed on 2026-07-30 came from a BTC return pool of 2 768 bars ending 2026-07-30 rather
than the paper's 2 756 ending 2026-07-18 — a cache miss caused by running from the wrong working
directory (Issue 9). Re-running the *identical* script from the repository root reproduces
different cell values: CW against the recursive mean at h=7 moved from 10.2 % to 5.0 %. Separately,
at 400 replications the cells move by up to three percentage points between random seeds, which is
larger than the binomial standard error suggests at a glance and larger than the precision the
first write-up implied. The paper's table is therefore built from 2 000 replications per linear
cell (600 for the booster) against the pinned cache, with Monte Carlo standard errors reported.

`mc_null.py` and `mc_null_gbm.py` are kept because they are what produced the retracted numbers;
`mc_null_wcw.py` supersedes both and computes everything they did.

Saved outputs: `forecasts.csv`, `retest_cw.csv`, `wcw.csv`, `cost_curve.csv`,
`mc_null_wcw_h{1,7}_b{1,21}.csv`, the `_signflip`, `_SOL`, `_gbm`, `_s11` and `_rho*` variants,
and `reports/figures/fig9_size.{png,pdf}`.

## The one constructive result

Clark–West against the **recursively estimated mean** is approximately correctly sized — 3.7 %
(ridge), 3.1 % (elastic net), 3.7 % (early-stopped XGBoost) at h = 1 under an independent null, and
4.8 / 3.5 / 3.0 % under a sign-flipped null that keeps the real volatility clustering; 5.7–7.2 % at
h = 7. Against the **zero forecast** the same statistic is 12–21 % at h = 1 and 22–30 % at h = 7.
Boosting is not the problem; the benchmark is.

Two follow-ups from this pass sharpen that. The published dependence-robust alternative — Wild
Clark–West (Pincheira, Hardy & Muñoz 2021) — was implemented and **does not help**: it is
numerically indistinguishable from CW here, and structurally it cannot repair a benchmark that
encodes the wrong hypothesis, since E[θ] = 1 leaves the drift term untouched. And the analytic HAC
version does **not** in fact fail under realistic dependence; that reading came from using block
resampling, which carries genuine predictability, as if it were a size null.

The most defensible contribution available from this project is therefore the calibration
measurement itself — extended to a full size/power surface across estimators, horizons and
dependence structures, and replicated on the Goyal–Welch equity data for external validity.
