# arXiv submission — *A Century Is Not Enough*

Everything below is prepared. **You submit**: arXiv uploads require your own account, and
posting is public and effectively permanent, so it is not something to automate.

## The package

```bash
cd paper-equity && ./build_arxiv.sh
```

Produces `paper-equity/arxiv-submission.tar.gz` (104 KB), containing exactly:

```
paper.tex  references.bib  paper.bbl  neurips_2023.sty
fig11_goyal_welch.pdf  fig12_equity.pdf
tables/economic_translation.tex  tables/full_results.tex
```

The tarball is flat and self-contained: `\graphicspath` is rewritten to `./`, and `paper.bbl`
ships pre-compiled because arXiv's bibtex pass cannot be relied on for a custom style.

**Verified**: extracted into a clean directory and compiled with a fresh TeX run — 15 pages,
0 errors, 0 undefined references, 19 bibliography entries. Re-run that check after any edit:

```bash
tar xzf paper-equity/arxiv-submission.tar.gz -C "$(mktemp -d)" && cd "$_" && tectonic -X compile paper.tex
```

## Form fields

**Title**

```
A Century Is Not Enough: Certification Bounds for Out-of-Sample Return Predictability
```

**Authors**

```
Mehmet Demir Guven
```

**Primary category**: `q-fin.ST` (Statistical Finance)

**Cross-lists**: `stat.ME` (Methodology), `econ.EM` (Econometrics)

**Comments field**

```
15 pages, 2 figures. Code, pinned data, and one script per reported number:
https://github.com/ITheClixs/crypto-return-predictability
```

**License**: choose **CC BY-NC-SA 4.0**, matching the repository and your non-commercial
intent. Note that arXiv's licence grant is irrevocable — you cannot narrow it later.

**Abstract**: paste from `paper.tex`, converting LaTeX to plain text. arXiv accepts inline
`$...$`, so the mathematics can stay as-is; strip `\emph{}` and `\citet{}` wrappers.

## Before you click submit

- [ ] Author name spelled as you want it cited, permanently
- [ ] The ETH footnote reads correctly — it states the work was independent, unfunded,
      unsupervised, uncommissioned and unendorsed. Confirm ETH's affiliation-use policy
      permits naming the department this way for independent work; this is the one item
      here that is a policy question rather than a technical one
- [ ] `demirguven178@gmail.com` is the contact address you want public
- [ ] The GitHub repository is public and its README matches what the paper claims
- [ ] You accept that v1 is permanent — arXiv never deletes, only supersedes

## What this paper is

State it plainly if asked, because the paper does:

- A minimum-detectable-effect calculation under an explicit Gaussian benchmark. A textbook
  power calculation, correctly labelled as one, **not** a distribution-free impossibility bound.
- An anytime-valid certificate applied to 117 predictor-frequency pairs. Nothing is certified
  under any of nine specifications.
- The contribution is the **interval**: the median marginal 95% upper endpoint on the
  incremental annualised information ratio is 0.60, and 112 of 116 exceed 0.25.

It confirms Welch & Goyal (2008) with a bound attached. It is not a landmark result and the
paper does not claim to be one. Expect modest citation counts.

## Known referee objections, and where each is answered

| Objection | Answer in the paper |
|---|---|
| "The bound isn't distribution-free" | Remark 2 concedes this explicitly |
| "This is just a power calculation" | Correct, and labelled as such in the abstract |
| "Your instrument is underpowered" | Section 7 states it, and the toolkit comparison shows Clark-West, ENC-t and the MCS find nothing either |
| "Peak years are break-date estimates" | Remark 4 refuses that reading; Table 4 marks uninterpretable peaks |
| "The median of upper endpoints isn't a bound on the median" | Stated in the abstract, results and conclusion |
