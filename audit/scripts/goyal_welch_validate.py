"""External validation: our pipeline against Goyal, Welch and Zafirov's published table.

A reproduction check against a paper's own headline is weak; a per-predictor check against its
published numbers is not. This compares our out-of-sample R-squared, predictor by predictor,
with Table 3 Panel A of Goyal, Welch and Zafirov (RFS 2024, 37(11), 3490-3557).

Two conventions have to match before the comparison means anything.

*The evaluation window.* GWZ: "Our OOS period always starts 20 years after the IS period, but
never earlier than 1946." Our 240-month burn-in from 1926-01 lands on 1946-01. Same window.

*The Campbell-Thompson restriction.* Their reported column is OOSCT, which truncates the
equity-premium forecast at zero on the grounds that a rational investor would not forecast a
negative premium. We report the unrestricted number by default, because truncation is a
economically-motivated shrinkage that changes the estimand. For the comparison to be
apples-to-apples we apply their restriction here.

The restriction is not cosmetic. It helps most exactly where the unrestricted forecast is
wildest, so a predictor whose regression occasionally produces an absurd forecast can look far
better after truncation than before. Reporting both is the point.

Usage: goyal_welch_validate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from goyal_welch_pilot import BURN, load, walk_forward
from goyal_welch_pilot import PREDICTORS as CLASSIC
from scipy import stats

#: Goyal, Welch and Zafirov (2024), Table 3 Panel A, column OOSCT, monthly variables.
#: Their "tby" is this workbook's "tbl". Transcribed by hand from the published table.
PUBLISHED_OOSCT = {
    "ntis": -0.47,
    "tbl": 0.37,
    "d/p": -0.06,
    "d/y": -0.06,
    "e/p": -0.64,
    "d/e": -0.93,
    "svar": -0.01,
    "lty": 0.25,
    "ltr": -0.82,
    "tms": 0.02,
    "dfy": -0.12,
    "dfr": -0.30,
    "infl": 0.10,
    "b/m": -1.06,
}


def oos_r2(outcome: np.ndarray, model: np.ndarray, bench: np.ndarray) -> float:
    return 100.0 * (1.0 - np.sum((outcome - model) ** 2) / np.sum((outcome - bench) ** 2))


def main() -> None:
    frame = load()
    premium = frame["premium"].to_numpy()
    rows = []
    for name in CLASSIC:
        lagged = frame[name].shift(1).to_numpy()
        outcome, model, bench = walk_forward(lagged, premium, BURN)
        if outcome.size < 240:
            continue
        rows.append(
            {
                "predictor": name,
                "ours_unrestricted": oos_r2(outcome, model, bench),
                # Campbell-Thompson: never forecast a negative equity premium, applied to
                # the model and to the benchmark alike, as GWZ do.
                "ours_ct": oos_r2(outcome, np.maximum(model, 0.0), np.maximum(bench, 0.0)),
                "published_ct": PUBLISHED_OOSCT.get(name, np.nan),
            }
        )

    table = pd.DataFrame(rows).dropna(subset=["published_ct"])
    table["gap"] = table["ours_ct"] - table["published_ct"]
    table["sign_agrees"] = np.sign(table["ours_ct"]) == np.sign(table["published_ct"])
    table.to_csv("audit/goyal_welch_validate.csv", index=False)

    print(
        "Our pipeline against Goyal-Welch-Zafirov (2024) Table 3 Panel A, "
        "out-of-sample R^2 in percent.\nBoth apply the Campbell-Thompson restriction; "
        "our unrestricted number is shown for contrast.\n"
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    corr = stats.pearsonr(table["ours_ct"], table["published_ct"])
    rank = stats.spearmanr(table["ours_ct"], table["published_ct"])
    agree = int(table["sign_agrees"].sum())
    print(
        f"\nn = {len(table)} predictors\n"
        f"  sign agreement      {agree}/{len(table)}\n"
        f"  Pearson  r = {corr.statistic:+.3f}  (p = {corr.pvalue:.4f})\n"
        f"  Spearman r = {rank.statistic:+.3f}  (p = {rank.pvalue:.4f})\n"
        f"  median |gap| = {table['gap'].abs().median():.3f} pp, "
        f"max |gap| = {table['gap'].abs().max():.3f} pp"
    )
    print(
        f"\nWhat the Campbell-Thompson restriction is worth on our own numbers: mean "
        f"out-of-sample R^2 moves from {table['ours_unrestricted'].mean():+.3f}% to "
        f"{table['ours_ct'].mean():+.3f}%, and it helps "
        f"{int((table['ours_ct'] > table['ours_unrestricted']).sum())} of {len(table)}."
    )
    print("\nwrote audit/goyal_welch_validate.csv")


if __name__ == "__main__":
    main()
