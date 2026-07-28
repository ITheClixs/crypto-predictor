"""Command-line entry point: ``cryptoforecast {data,backtest,report}``.

``backtest`` is the one that matters — it runs the whole study on real data and
writes reports/results.md, results.csv, and figures. ``data`` just warms the
cache; ``report`` re-renders the markdown from a saved results.csv.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, StudyConfig
from .data.loaders import load_ohlcv
from .evaluate.report import results_table, write_markdown
from .plots import generate_figures
from .study import run_study

REPORTS_DIR = Path("reports")


def _config_from_args(args: argparse.Namespace) -> StudyConfig:
    cfg = DEFAULT_CONFIG
    if getattr(args, "assets", None):
        cfg = replace(cfg, assets=tuple(a.strip().upper() for a in args.assets.split(",")))
    if getattr(args, "start", None):
        cfg = replace(cfg, start=args.start)
    if getattr(args, "horizons", None):
        cfg = replace(cfg, horizons=tuple(int(h) for h in args.horizons.split(",")))
    return cfg


def _cmd_data(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    for asset in cfg.assets:
        df = load_ohlcv(asset, cfg.start, cfg.end, cfg.interval)
        print(f"{asset:5s} {len(df):5d} bars  {df.index.min().date()} → {df.index.max().date()}")


def _cmd_backtest(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    print(f"Running study: assets={cfg.assets} horizons={cfg.horizons} start={cfg.start}")
    study = run_study(cfg)
    table = results_table(study)

    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "results.csv", index=False)
    figures = generate_figures(study, table, REPORTS_DIR / "figures")
    write_markdown(cfg, table, figures, REPORTS_DIR / "results.md")
    print(f"Wrote {REPORTS_DIR}/results.md, results.csv, and {len(figures)} figures.")


def _cmd_report(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    table = pd.read_csv(REPORTS_DIR / "results.csv")
    figures = sorted(
        str(p.relative_to(REPORTS_DIR)) for p in (REPORTS_DIR / "figures").glob("*.png")
    )
    write_markdown(cfg, table, figures, REPORTS_DIR / "results.md")
    print(f"Re-rendered {REPORTS_DIR}/results.md from results.csv.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cryptoforecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, needs_run in (
        ("data", _cmd_data, True),
        ("backtest", _cmd_backtest, True),
        ("report", _cmd_report, False),
    ):
        p = sub.add_parser(name)
        p.add_argument("--assets", help="comma-separated symbols, e.g. BTC,ETH")
        p.add_argument("--start", help="start date YYYY-MM-DD")
        if needs_run:
            p.add_argument("--horizons", help="comma-separated horizons in days, e.g. 1,7")
        p.set_defaults(func=handler)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
