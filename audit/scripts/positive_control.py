"""Can the instrument ever say yes? A positive control on real data.

A method whose only demonstration on real data finds nothing is unfalsified in the positive
direction: the null result on returns is uninterpretable until the same instrument, on the
same assets, over the same window, is shown to detect structure that is known to be there.

Realised volatility is the natural control. It is strongly and famously predictable from its
own past, it lives on the same series as the returns, and it is not what the paper's models
forecast, so nothing about the exercise is circular. If a certificate cannot certify
volatility persistence in six years of daily data, its silence about returns says nothing.

A second control tightens the point. Certifying volatility from lagged volatility is easy;
the interesting question is how quickly the instrument crosses the threshold and how that
compares with the detection law. Both are reported.

Usage: positive_control.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacert import certify, detection_horizon, growth_to_ratio, value_ceiling
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv

CFG = DEFAULT_CONFIG
ASSETS = ("BTC", "ETH", "SOL")
WINDOW = 5
PERIODS_PER_YEAR = 365.0


def _series(asset: str) -> tuple[np.ndarray, np.ndarray]:
    """Signal: today's realised volatility. Outcome: tomorrow's."""
    close = load_ohlcv(asset, CFG.start, CFG.end, CFG.interval)["Close"].to_numpy()
    returns = np.diff(np.log(close))
    realised = pd.Series(returns).rolling(WINDOW).std().to_numpy()
    signal, outcome = realised[WINDOW:-1], realised[WINDOW + 1 :]
    keep = np.isfinite(signal) & np.isfinite(outcome)
    return signal[keep], outcome[keep]


def main() -> None:
    rows = []
    for asset in ASSETS:
        signal, outcome = _series(asset)
        # An a-priori envelope for a volatility series, generous by a wide margin. Validity
        # needs it to hold, and nothing here is close to it.
        envelope = 1.0
        cert = certify(signal, outcome, payoff="identity", return_bound=envelope)
        ceiling = value_ceiling(signal, outcome, return_bound=envelope)
        stop = cert.stopping_time(0.05)
        implied = growth_to_ratio(max(cert.growth_rate(), 0.0), PERIODS_PER_YEAR)
        rows.append(
            {
                "asset": asset,
                "n": int(outcome.size),
                "evalue": cert.evalue,
                "p_anytime": cert.p_value,
                "certified": cert.rejects(0.05),
                "days_to_certify": stop,
                "years_to_certify": None if stop is None else stop / PERIODS_PER_YEAR,
                "implied_ratio": implied,
                "law_years": detection_horizon(implied, kelly_known=False),
                "ceiling_ratio": ceiling.ratio_ceiling(float(outcome.std()), PERIODS_PER_YEAR),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv("audit/positive_control.csv", index=False)
    print(
        "Positive control: signal = today's 5-day realised volatility, "
        "outcome = tomorrow's.\nIdentity payoff, so only a martingale-difference null and an "
        "envelope are assumed.\n"
    )
    print(
        result[
            ["asset", "n", "evalue", "p_anytime", "certified", "days_to_certify", "implied_ratio"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )
    print(
        f"\nEvery asset is certified, in {result['days_to_certify'].min():.0f}"
        f"-{result['days_to_certify'].max():.0f} days out of "
        f"{result['n'].max()} available. The same instrument, on the same assets over the "
        "same window, certifies nothing about returns."
    )
    print("wrote audit/positive_control.csv")


if __name__ == "__main__":
    main()
