# arXiv submission

Everything the submission form asks for, plus the checks that were run before building the
tarball. Rebuild the package with `./paper/build_arxiv.sh`, which writes
`paper/arxiv-submission.tar.gz`.

## Settled

**Affiliation.** ETH Zürich, with a title-page footnote and a Declarations section recording
that the work was independent: not funded, supervised, commissioned or endorsed by ETH or any
other institution, and with no collaborators.

**Licence.** Select **CC BY-NC-SA 4.0** on the arXiv submission form. It matches the
repository: PolyForm Noncommercial 1.0.0 for code, CC BY-NC-SA 4.0 for the manuscript,
figures and derived data.

**Patent.** None sought, decided 2026-08-09. Posting is therefore unblocked. Recorded because
the decision is irreversible in one direction: publication is a public disclosure and an
absolute novelty bar at the EPO and in Switzerland, so it forecloses a European filing
permanently. Nothing in this repository should be read as preserving that option.

**Competing interests.** The author holds no position in BTC, ETH or SOL, confirmed
2026-08-09. The Declarations section states this.

## Read this before you press submit

**A non-commercial licence does not protect the method.** Copyright reaches this text and this
code, not the mathematics. Anyone may reimplement the certificate from Section 6 and sell it.
That is stated plainly in `LICENSE` and in the Declarations rather than left for a reader to
discover.

**One cost of NC you are accepting knowingly.** PolyForm Noncommercial is not an OSI-approved
open-source licence. Many quantitative desks and several package ecosystems cannot use
non-OSI code at all, and the paper's pitch is partly "install this and check it on your own
pipeline". The licence therefore narrows the audience for the artifact that carries the
contribution. That is a legitimate trade to make; it should be a deliberate one.

## Form fields

**Title**

    Betting Against the Benchmark: Drift-Robust, Anytime-Valid Certificates of
    Out-of-Sample Predictive Ability

**Categories**

| Field | Value | Why |
|---|---|---|
| Primary | `q-fin.ST` (Statistical Finance) | The estimand, the empirical study and the intended readership are all financial. |
| Cross-list | `stat.ME` (Methodology) | The certificate is a general sequential test with a composite nuisance; nothing about it is specific to returns. |
| Cross-list | `econ.EM` (Econometrics) | It is a nested predictive-accuracy test, which is where Clark–West and Diebold–Mariano live. |
| Cross-list | `cs.LG` (Machine Learning) | The validity statement covers black-box and adaptively-stopped learners, which is the gap in the applied ML-for-finance literature. |

**Comments field**

    40 pages, 10 figures. Reference implementation, data cache, and the scripts reproducing
    every number: https://github.com/ITheClixs/crypto-return-predictability

**MSC class** `62L10, 62M20, 91G70` — sequential analysis, prediction theory, statistical
methods in finance.

**ACM class** `G.3; I.2.6` — probability and statistics; learning.

**Abstract** (1,915 characters, under arXiv's 1,920 limit; plain text, no LaTeX macros)

> Tests of out-of-sample return predictability dispose of the asset's drift by estimating it, and
> that is where they break. A model with an unpenalised intercept does not reduce to the zero
> forecast when its slopes vanish, so racing it against zero tests no drift and no conditional
> predictability jointly. We show in closed form that the resulting Clark--West statistic
> converges to the t-statistic of the asset's mean return: at Bitcoin's drift a model with no
> features at all is called significantly predictive in 43% of samples. We therefore eliminate
> the drift instead of estimating it. A certificate is a non-negative wealth process -- one
> bettor stakes on the signal, a second hedges every candidate drift -- whose infimum over the
> nuisance is bounded by a martingale under the null. Validity is finite-sample and uniform in
> time, with no bandwidth, long-run variance, bootstrap, asymptotics or refitting, for arbitrary
> black-box pipelines. It can therefore be monitored daily with no correction for looking, and
> e-values average under arbitrary dependence, so a research grid needs no joint bootstrap and an
> overlapping h-step forecast combines into exactly the staggered h-vintage portfolio. The
> construction closes the sample-size question -- the median time to certify an annualised
> information ratio IR is 2ln(1/alpha)/IR^2 years, about 6/IR^2 at 5% -- and we are equally
> explicit about the price, which the e-value literature does not report for this comparison:
> matched on level and power the guarantee costs about twice the sample of a fixed-sample test,
> and against a correctly sized Clark--West we measure power of 0.48 against 0.85 at IR=2. As a
> positive control the same instrument certifies volatility persistence on the same assets within
> 148--416 days. On returns it certifies nothing in six years, and its ceiling bounds the
> incremental information ratio at 2.6 without separating it from zero.

## What the tarball contains, and why

arXiv unpacks the archive into a single directory and runs its own TeX Live, so two things
differ from the working copy.

- **Figures are flattened.** The repository refers to `../reports/figures/`, which does not
  exist inside the tarball. `build_arxiv.sh` copies exactly the PDFs the manuscript
  `\includegraphics`, and rewrites `\graphicspath` to `{./}`.
- **`paper.bbl` is shipped.** arXiv will run BibTeX, but shipping the compiled bibliography
  removes the dependency on it resolving a custom `.bst` the same way. `references.bib` is
  included too, so the entry data travels with the paper.

Build products (`.aux`, `.log`, `.out`, `.blg`, `.xdv`, `.pdf`) are deleted before archiving:
arXiv wants source.

## Pre-submission checks

All of these are run by `./paper/build_arxiv.sh` or listed here so they can be repeated.

- [x] Tarball compiles from a clean extraction with no errors.
- [x] Zero undefined references and zero undefined citations in the final pass.
- [x] Zero missing graphics.
- [x] No absolute paths anywhere in `paper.tex`.
- [x] No `\write18`, shell-escape, or `\input` outside the bundled `tables/`.
- [x] Abstract under the 1,920-character limit as plain text.
- [x] Every number in the manuscript is produced by a committed script under `audit/scripts/`
      or by `cryptoforecast backtest`, against a data cache pinned at 2026-07-18.
- [x] Test suite green (`make check`): lint, format, types, and 220 tests including the
      leakage guarantees, the certificate's validity and power, and the regression test that
      pins the meaning of the backtest's entry delay.

## Reproducing every figure and table

```bash
make setup
make backtest                                          # study, reports/, figures 1-8
./venv/bin/python audit/scripts/gen_forecasts.py       # 72,900 OOS forecasts
./venv/bin/python audit/scripts/certificate_study.py   # Table: certificates
./venv/bin/python audit/scripts/certificate_calibration.py 400
./venv/bin/python audit/scripts/garch_null.py 400      # Table: null generators
./venv/bin/python audit/scripts/execution_contrast.py  # Table: execution conventions
./venv/bin/python audit/scripts/bootstrap_stability.py
./venv/bin/python audit/scripts/power_head_to_head.py 400   # Table: what validity costs
./venv/bin/python audit/scripts/positive_control.py         # the instrument does say yes
./venv/bin/python audit/scripts/plot_certificate.py    # Figure 10
./venv/bin/python audit/scripts/mc_joint_null.py 2000 --gbm   # hours
./venv/bin/python audit/scripts/joint_null_report.py
make paper
```
