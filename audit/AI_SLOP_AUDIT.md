# Writing audit — passages that read as performed rigour rather than demonstrated rigour

The manuscript's prose is fluent and the sentence-level craft is above average. The failure mode is
different: a recurring rhetorical move in which a methodological choice is dramatised, then the
drama is offered in place of the evidence. A referee who reads carefully will notice that the most
confident sentences are the least supported ones. Below, the worst offenders, ranked by how much
damage they do to credibility.

Format: quoted text → what is wrong → what to write instead.

---

## 1. Claims of verification standing in for verification

> "We verify the bias by simulation rather than asserting it."

The simulation does not verify the bias (FORENSIC_REVIEW Issue 2 — the null it simulates is false and
no parameter is estimated). The sentence announces the virtue of verifying while performing the
failure it disclaims. This is the single most damaging line in the paper because it invites the
referee to check.

Replacement, if the Monte Carlo in `audit/scripts/mc_null.py` is adopted:

> "Under a nested-estimation null constructed by resampling the return series (drift, tails, feature
> persistence and sample length retained; conditional predictability removed), Clark–West against the
> recursively estimated mean rejects 6.2 % of the time at a nominal 5 % (n = 400, s.e. 1.2 pp), while
> Clark–West against the zero forecast rejects 14.2–22.0 %."

Same treatment for:

> "The absence of look-ahead is enforced by a test rather than by inspection"

— this one is true, and it should be the template. It names the mechanism, so the reader can check
it. Every other "we verify" sentence in the paper should meet the standard this one sets.

## 2. "Impossible by construction"

> "This paper is organised so that each of these is either impossible by construction or measured and
> reported."

Of the four failure modes listed, only preprocessing leakage is impossible by construction (in-fold
scaling). Target leakage is prevented by a test, not by construction. Absent benchmarks is a choice,
and it is the choice that fails (Issue 1). Silent multiple testing is measured over a family that
excludes most of the search (Issue 8). Replace with the specific mechanism for each of the four, one
clause each, no collective claim.

## 3. Rhetorical inversions used as arguments

Recurring pattern, roughly one instance per page:

> "The contribution is not a model. It is an evaluation protocol applied honestly to a question with
> a commercially attractive answer, which returns the unattractive one."
>
> "The two tests disagree, and the disagreement is the finding."
>
> "A study reporting a single-offset Sharpe at a multi-day horizon without this check is reporting
> one draw and calling it an estimate."
>
> "which is reason enough never to report either p-value without its statistic"
>
> "which is impressive until one observes that it makes exactly one position change"

Each of these is a true observation delivered as an epigram. Individually they read as confident.
Cumulatively they read as generated, because the cadence never varies: setup, pivot, aphorism. Keep
at most one such construction in the paper — the buy-and-hold one, which earns it — and convert the
rest to plain declaratives. "The two tests disagree, and the disagreement is the finding" should
become the actual claim: which settings, under which benchmark, with what calibrated size.

## 4. Adjectives standing in for measurements

| phrase | occurrences | required replacement |
|---|---|---|
| "faint, fragile statistical signal" | abstract, conclusion | the calibrated rejection count and its expectation under the null |
| "the full apparatus that would have detected either" | abstract | the minimum detectable effect size, computed |
| "deliberately constrained gradient booster" | related work, forecasters | the hyperparameters and why those values (they were never tuned or justified) |
| "realistic costs" | abstract, discussion | the cost curve; 17 bp is one scenario |
| "conservative" (of the split logic, of the cost model) | 3 places | the comparison that makes it conservative |
| "not a formality" / "not a detail of presentation" | 2 places | delete; state the effect size |

## 5. Categorical statements where the evidence is conditional

> "every high-Sharpe result in the study is long-only exposure in disguise"

This one survives measurement (Issue 10: β = 0.76–0.89, 91–94 % long, α t-statistics 1.07 and 1.15) —
so replace "in disguise" with the numbers and keep the finding. "In disguise" is undefined and reads
as flourish; β to buy-and-hold with a Newey–West t on the intercept reads as evidence.

> "Both statements are defensible from the same forecasts. Only the second is correct."

Neither is correct (Issue 3). Delete.

> "This is what a correctly specified null looks like when it is true"

The null is not correctly specified (Issue 1) and its truth is not established. Delete.

## 6. Formula presence without formula consequence

The Pesaran–Timmermann variance expressions, the PSR expression, and the expected-maximum-Sharpe
expression are all reproduced in full and none of them changes a decision in the paper. Worse, the
PT expression as displayed is the one whose independence assumption is violated (Issue 4), so
displaying it in full advertises the error. Move all three to the appendix; in the main text state
what each is used for and what its assumption is.

## 7. The limitations section lists weaknesses without their consequences

The paragraph is honest — it discloses survivorship, the flat cost model, the percentile bootstrap,
the DSR undercount, and the unpinned sample. But every item is disclosed and then abandoned. "The
deflated Sharpe ratio understates the true search" — by how much, and does any conclusion depend on
it? "the sample end date is unpinned" — so is the study reproducible or not? A limitation that does
not say what would change if it were fixed is a hedge, not a limitation.

## 8. Mechanical repairs

- Semicolon-and-em-dash density is high enough to be a stylistic tic; roughly one per two sentences
  in the Method. Cut by half.
- "which is" as a connective appears 14 times. Most are removable.
- Three consecutive paragraphs in Discussion open with a bolded aphorism. Vary or drop the bolding.
- The abstract counts p-values (7, 5, 2, 0) before defining the family. Lead with the estimand.
- Title names the domain, not the contribution. If the pivot in `RESEARCH_PIVOT_MEMO.md` is adopted
  the title should name the inference problem.

## 9. What to keep

The look-ahead perturbation test, the purged early-stopping holdout, the `1/n` Newey–West
normalisation footnote, the sign-convention table in `evaluate/stats.py`, and the position-based
purging argument under missing bars are all examples of the paper explaining a real decision and its
mechanism. That voice — mechanism first, no epigram — is the one to write the whole paper in.
