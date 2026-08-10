# NeurIPS 2026 E-Values Workshop — submission guide

**Workshop**: E-Values: From Statistics to ML, NeurIPS 2026, Paris, 12 or 13 December 2026.

**Deadline**: **29 August 2026, 23:59 AoE.**

**Submit at**: <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/E-values>

## Do this first

**Create your OpenReview profile now.** OpenReview requires profiles to exist at least two
weeks before the deadline, which puts the cutoff around **15 August**. Everything else here can
be done in an afternoon; this one has a hard external clock and will silently block you.

## Two mechanical steps before uploading

1. **Swap the style file.** `paper.tex` currently loads the vendored `neurips_2023.sty`,
   because that is what this repository has. The workshop requires the **NeurIPS 2026**
   template with the **`dblblindworkshop`** option. Download it from the workshop site and
   replace line 6:

   ```latex
   \usepackage[dblblindworkshop]{neurips_2026}
   ```

2. **Confirm anonymity.** The submission is double-blind. Already handled in the source:
   `\author{Anonymous}`, no ETH affiliation, no email, and the software paragraph says
   *"Repository link withheld for double-blind review."* **Do not add the GitHub link back** —
   it deanonymises immediately, since the repository carries your name. Grep before uploading:

   ```bash
   grep -niE "guven|eth|zurich|github|demirguven" paper-workshop/paper.tex
   ```

   That must return nothing.

## Length

Four pages excluding references and appendices. Current build is **4 pages total including
references**, so the body is comfortably inside the limit with roughly half a page spare.

## What the paper claims, and what it does not

Worth being clear before a referee is:

**Claimed**: the *assembly* is new — an e-process for nested forecast comparison that
eliminates the benchmark's drift as a nuisance rather than estimating it, plus a measured price
for the construction and for anytime-validity itself.

**Not claimed**: the components. Ville's inequality, testing by betting, and taking an infimum
over a nuisance for a composite null are all standard and cited as such. The paper says
explicitly that the infimum is not growth-optimal and that the numeraire e-variable
(Larsson–Ramdas–Ruf, AoS 2025) would be tighter, and that computing it for this null has not
been done.

The prior-art check that supports this: no e-value, betting-martingale or confidence-sequence
treatment exists in the forecast-evaluation literature (searched 2026-08-11). The closest
adjacent work is anytime-valid testing for elicitable functionals (arXiv 2204.05680), which is
a general framework rather than a nested comparison with a drift nuisance.

## The strongest thing in it

Table 1. Clark–West against a zero benchmark rejects a **pure-noise** signal 81% of the time at
nominal 5% once the asset drifts at Sharpe 2. That is a concrete, reproducible failure of a
test in wide applied use, and the construction fixes it. Lead with it if you present.

## Likely referee objections

| Objection | Where it's answered |
|---|---|
| "The infimum construction is standard" | Conceded in §2, with the numeraire cited as the tighter alternative |
| "Weaker than Clark–West" | Table 2 shows exactly that, at matched size |
| "Why not use the numeraire e-variable?" | Stated as the obvious next step, not done |
| "The application is a null result" | Yes — the ceiling is the output, not the rejection |
| "Is the drift failure realistic?" | Sharpe 2 is high for an index but ordinary for a single crypto asset or a momentum book, which is where zero-benchmark comparisons are actually run |

## Reproducing the two tables

```bash
PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/certificate_calibration.py  # Table 1
PYTHONPATH=src:audit/scripts venv/bin/python audit/scripts/power_head_to_head.py       # Table 2
```

Committed outputs: `audit/certificate_drift_sweep.csv`, `audit/power_head_to_head.csv`.

## Checklist

- [ ] OpenReview profile created (**by ~15 August**)
- [ ] `neurips_2026.sty` with `dblblindworkshop` swapped in
- [ ] Anonymity grep returns nothing
- [ ] Body ≤ 4 pages under the 2026 style — re-check, since the style change reflows everything
- [ ] Both table sources re-run and matching the paper
- [ ] Abstract pasted into OpenReview matches the PDF
