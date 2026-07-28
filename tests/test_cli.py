"""CLI wiring for backtest/report, with the study runner stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

import cryptoforecast.cli as cli
from conftest import make_study


@pytest.mark.unit
def test_backtest_then_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(cli, "run_study", lambda cfg: make_study())

    cli.main(["backtest", "--assets", "BTC", "--horizons", "1"])
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.csv").exists()
    assert sorted((tmp_path / "figures").glob("*.png"))

    # report re-renders markdown from the saved csv without re-running the study.
    (tmp_path / "results.md").unlink()
    cli.main(["report", "--assets", "BTC"])
    assert (tmp_path / "results.md").exists()


@pytest.mark.unit
def test_data_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {c: [1.0, 2.0] for c in ("Open", "High", "Low", "Close", "Volume")},
        index=pd.date_range("2022-01-01", periods=2),
    )
    monkeypatch.setattr(cli, "load_ohlcv", lambda *a, **k: frame)
    cli.main(["data", "--assets", "BTC"])
    assert "BTC" in capsys.readouterr().out
