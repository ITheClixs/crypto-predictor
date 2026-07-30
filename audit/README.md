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
**14–22 %** of the time on data containing no predictability, so the correct comparison is 5 observed
against 2.6–4.0 expected. Two further claims are mathematically false (the nesting claim, and the
claim that the validating simulation holds the null by construction), and the paper's only significant
sign-timing results (p ≈ 0.001) become p ≈ 0.24–0.27 once overlapping labels are handled.

The engineering and the leakage discipline are sound. All headline numbers reproduce bit-for-bit.

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

# Issue 3 -- the experiment the paper never ran. ~12 min for the four linear cells.
$PY audit/scripts/mc_null.py 1 1  400   # h, bootstrap block (1 = iid), replications
$PY audit/scripts/mc_null.py 1 21 400
$PY audit/scripts/mc_null.py 7 1  400
$PY audit/scripts/mc_null.py 7 21 400
$PY audit/scripts/mc_null_gbm.py 1 1 150   # same, with the early-stopped XGBoost configuration
```

The committed Monte Carlo outputs were generated with a BTC return pool of 2 768 bars ending
2026-07-30 rather than the paper's 2 756 ending 2026-07-18 — a consequence of the same
working-directory bug. The pool is used only as a resampling source for synthetic paths, so twelve
extra bars out of 2 768 do not move the size estimates, but the discrepancy is recorded here rather
than hidden.

Saved outputs: `forecasts.csv`, `retest_cw.csv`, `mc_null_h{1,7}_b{1,21}.csv`,
`mc_null_gbm_h1_b1.csv`.

## The one constructive result

Clark–West against the **recursively estimated mean** is approximately correctly sized at h = 1 under
an iid-return null — 6.2 % (ridge), 6.2 % (elastic net), 5.3 % (early-stopped XGBoost) at a nominal
5 %. Against the **zero forecast** the same statistic is 14.2 %, 22.0 % and 22.0 %. Boosting is not
the problem; the benchmark is. That measurement, extended to a full size/power surface and paired with
a dependence-robust alternative for the h = 7 and volatility-clustering cases where the analytic HAC
version still fails, is the most defensible contribution available from this project.
