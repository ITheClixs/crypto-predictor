"""A thin, honest Flask demo over the cryptoforecast research library.

Unlike a typical "price predictor", this demo refuses to print a confident future
price. For the asset/model/horizon you choose it runs the *same* leak-free
walk-forward evaluation as the study and shows what actually happened out of
sample: error vs a random walk, directional accuracy, and a net-of-cost equity
curve. The model's current directional lean is shown only with the caveat that,
per the backtest, it carries no statistically significant edge.
"""

from __future__ import annotations

import base64
import io
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from cryptoforecast.backtest.engine import walk_forward
from cryptoforecast.backtest.strategy import StrategyResult, backtest_strategy
from cryptoforecast.config import DEFAULT_CONFIG
from cryptoforecast.data.loaders import load_ohlcv
from cryptoforecast.dataset import build_supervised
from cryptoforecast.evaluate.metrics import regression_metrics
from cryptoforecast.evaluate.stats import (
    block_bootstrap_ci,
    clark_west,
    diebold_mariano,
    max_drawdown,
    sharpe_ratio,
)
from cryptoforecast.models.registry import PRIMARY_BENCHMARK, default_models

app = Flask(__name__)

ASSETS = ["BTC", "ETH", "SOL"]
HORIZONS = [1, 7]
MODELS = [name for name in default_models() if name != PRIMARY_BENCHMARK]


def _validated_selection(form: dict[str, str]) -> tuple[str, str, int]:
    """Resolve the form into a known asset/model/horizon.

    Everything reaching this endpoint is untrusted, and `asset` ends up in an
    outbound market-data request, so it is matched against the allow-list rather
    than passed through.
    """
    asset = form.get("asset", ASSETS[0])
    model_name = form.get("model", MODELS[0])
    raw_horizon = form.get("horizon", str(HORIZONS[0]))
    if asset not in ASSETS:
        raise ValueError(f"unknown asset; choose one of {', '.join(ASSETS)}")
    if model_name not in MODELS:
        raise ValueError(f"unknown model; choose one of {', '.join(MODELS)}")
    try:
        horizon = int(raw_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizon must be a whole number of days") from exc
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon; choose one of {HORIZONS}")
    return asset, model_name, horizon


def _equity_png(strategy: StrategyResult) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    eq = strategy.equity
    ax.plot(eq.index, eq.to_numpy(), color="#4C72B0", lw=1.4)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("net equity")
    ax.set_title("Out-of-sample strategy equity (after costs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _stat(statistic: float, p_value: float) -> str:
    """Render a test as ``statistic (p)``. The sign is what says who won."""
    if statistic != statistic:  # NaN
        return "—"
    return f"{statistic:+.2f} (p = {p_value:.3f})"


def evaluate(asset: str, model_name: str, horizon: int) -> dict[str, object]:
    """Run the leak-free walk-forward evaluation for one asset/model/horizon."""
    cfg = DEFAULT_CONFIG
    ohlcv = load_ohlcv(asset, cfg.start, cfg.end, cfg.interval)
    ds = build_supervised(ohlcv, horizon)
    models = default_models(horizon)

    oos = walk_forward(ds, models[model_name], cfg.wf, horizon)
    rw = walk_forward(ds, models[PRIMARY_BENCHMARK], cfg.wf, horizon)
    metrics = regression_metrics(oos["y_true"], oos["y_pred"].to_numpy())
    dm = diebold_mariano(oos["y_true"], oos["y_pred"].to_numpy(), rw["y_pred"].to_numpy(), horizon)
    # The random walk is nested in every model offered here, so Clark-West is the
    # valid comparison; DM is shown alongside it because it is the familiar one.
    cw = clark_west(oos["y_true"], oos["y_pred"].to_numpy(), rw["y_pred"].to_numpy(), horizon)

    strategy = backtest_strategy(oos, horizon, cfg.costs)
    ppy = strategy.periods_per_year
    sr = sharpe_ratio(strategy.net, ppy)
    lo, hi = block_bootstrap_ci(
        strategy.net, partial(sharpe_ratio, periods_per_year=ppy), n_boot=400
    )

    # The model's live directional lean: fit on all labeled data, read the last row.
    labeled = ds.labeled
    live_model = models[model_name]().fit(labeled.X, labeled.y)
    latest_pred = float(live_model.predict(ds.X.iloc[[-1]])[0])

    # A win needs the right direction, not merely a small p: DM is negative and
    # Clark-West positive when the model genuinely has the lower loss.
    beats_rw = dm.statistic < 0 and dm.p_value < 0.05
    beats_rw_cw = cw.statistic > 0 and cw.p_value < 0.05
    return {
        "asset": asset,
        "model": model_name,
        "horizon": horizon,
        "current_price": f"${ds.close.iloc[-1]:,.2f}",
        "n_oos": len(oos),
        "rmse": f"{metrics['rmse']:.4f}",
        "rmse_rw": f"{regression_metrics(rw['y_true'], rw['y_pred'].to_numpy())['rmse']:.4f}",
        "dir_acc": f"{metrics['dir_acc']:.1%}",
        "dm": _stat(dm.statistic, dm.p_value),
        "cw": _stat(cw.statistic, cw.p_value),
        "beats_rw": beats_rw,
        "beats_rw_cw": beats_rw_cw,
        "sharpe": f"{sr:.2f}",
        "sharpe_ci": f"[{lo:.2f}, {hi:.2f}]",
        "max_dd": f"{max_drawdown(strategy.equity):.1%}",
        "lean_up": latest_pred > 0,
        "equity_png": _equity_png(strategy),
    }


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    result: dict[str, object] | None = None
    error: str | None = None
    asset, model_name, horizon = ASSETS[0], MODELS[0], HORIZONS[0]

    if request.method == "POST":
        try:
            asset, model_name, horizon = _validated_selection(request.form)
        except ValueError as exc:
            return render_template(
                "index.html",
                assets=ASSETS,
                models=MODELS,
                horizons=HORIZONS,
                selected={"asset": asset, "model": model_name, "horizon": horizon},
                result=None,
                error=str(exc),
            )
        try:
            result = evaluate(asset, model_name, horizon)
        except Exception:  # surface a failure to the user, never a 500 or a stack trace
            app.logger.exception("evaluation failed for %s/%s/%sd", asset, model_name, horizon)
            error = "Could not evaluate that combination; see the server log for details."

    return render_template(
        "index.html",
        assets=ASSETS,
        models=MODELS,
        horizons=HORIZONS,
        selected={"asset": asset, "model": model_name, "horizon": horizon},
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
