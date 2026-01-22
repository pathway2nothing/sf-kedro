from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile

import mlflow
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


FigureLike = Union[go.Figure, List[go.Figure]]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_pl_df(x: Any) -> Optional[pl.DataFrame]:
    if x is None:
        return None
    if isinstance(x, pl.DataFrame):
        return x
    # інколи можуть прилетіти pandas
    try:
        import pandas as pd  # type: ignore
        if isinstance(x, pd.DataFrame):
            return pl.from_pandas(x)
    except Exception:
        pass
    return None


@dataclass
class StrategyMetricsConfig:
    # візуал
    chart_height: int = 900
    chart_width: int = 1400

    # pair breakdown
    top_n_pairs: int = 12

    # колонки (підлаштуй під те, що реально пише runner.get_results())
    equity_time_col: str = "timestamp"
    equity_col: str = "equity"
    equity_return_col: str = "equity_return"  # якщо є

    trades_pair_col: str = "pair"
    trades_entry_col: str = "entry_time"
    trades_exit_col: str = "exit_time"
    trades_profit_col: str = "profit"         # абсолют або у частці — неважливо, підпишемо


def compute_strategy_metrics(
    backtest_results: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, FigureLike]]:
    """
    Аналог compute_signal_metrics, але для backtest_results.

    Повертає:
      - results: dict з агрегованими метриками
      - plots: dict(name -> plotly Figure або list[Figure])
    """
    cfg = StrategyMetricsConfig(**(params or {}))

    metrics_df = _as_pl_df(backtest_results.get("metrics_df"))
    trades_df = _as_pl_df(backtest_results.get("trades_df"))

    # 1) агреговані скаляри (беремо те, що runner вже поклав)
    results: Dict[str, Any] = {
        "quant": {
            "final_return": _safe_float(backtest_results.get("final_return", 0.0)),
            "max_drawdown": _safe_float(backtest_results.get("max_drawdown", 0.0)),
            "sharpe_ratio": _safe_float(backtest_results.get("sharpe_ratio", 0.0)),
            "win_rate": _safe_float(backtest_results.get("win_rate", 0.0)),
            "total_trades": int(backtest_results.get("total_trades", 0) or 0),
            "final_equity": _safe_float(backtest_results.get("final_equity", 0.0)),
        },
        "meta": {
            "has_metrics_df": metrics_df is not None,
            "has_trades_df": trades_df is not None,
        },
    }

    plots: Dict[str, FigureLike] = {}

    # 2) equity curve
    if metrics_df is not None and metrics_df.height > 0:
        try:
            plots["StrategyEquityCurve"] = _plot_equity_curve(metrics_df, results, cfg)
        except Exception as e:
            logger.exception(f"Failed to plot equity curve: {e}")
            plots["StrategyEquityCurve"] = None

    # 3) pair breakdown
    if trades_df is not None and trades_df.height > 0:
        try:
            plots["StrategyPairBreakdown"] = _plot_pair_breakdown(trades_df, results, cfg)
        except Exception as e:
            logger.exception(f"Failed to plot pair breakdown: {e}")
            plots["StrategyPairBreakdown"] = None

    # 4) MLflow logging (як у signal_metrics)
    if mlflow.active_run():
        logger.info("Logging strategy metrics to MLflow")
        _log_strategy_metrics_to_mlflow(results=results, plots=plots)
    else:
        logger.debug("No active MLflow run, skipping MLflow logging")

    return results, plots


def save_strategy_plots(
    plots: Dict[str, FigureLike],
    output_dir: str,
    width: int = 1400,
    height: int = 900,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for name, figs in plots.items():
        if figs is None:
            logger.warning(f"Skipping {name}: no figures")
            continue

        metric_dir = out / name
        metric_dir.mkdir(exist_ok=True)

        fig_list = figs if isinstance(figs, list) else [figs]
        for i, fig in enumerate(fig_list):
            if fig is None:
                continue
            png_path = metric_dir / f"{i}.png"
            html_path = metric_dir / f"{i}.html"
            fig.write_image(str(png_path), width=width, height=height, scale=2)
            fig.write_html(str(html_path))
            total += 1

    logger.info(f"Saved {total} strategy plots to {out}")


def _plot_equity_curve(
    metrics_df: pl.DataFrame,
    results: Dict[str, Any],
    cfg: StrategyMetricsConfig,
) -> go.Figure:
    df = metrics_df

    # мінімальна нормалізація колонок
    if cfg.equity_time_col not in df.columns:
        # пробуємо типові варіанти
        for alt in ["time", "date", "datetime", "index"]:
            if alt in df.columns:
                df = df.rename({alt: cfg.equity_time_col})
                break

    if cfg.equity_col not in df.columns:
        for alt in ["balance", "equity_usd", "nav"]:
            if alt in df.columns:
                df = df.rename({alt: cfg.equity_col})
                break

    time = df.get_column(cfg.equity_time_col).to_list()
    equity = df.get_column(cfg.equity_col).to_list()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Equity Curve", "Drawdown (approx)"],
    )

    fig.add_trace(
        go.Scatter(x=time, y=equity, mode="lines", name="Equity"),
        row=1,
        col=1,
    )

    # простий drawdown від running max
    eq_s = pl.Series(equity)
    running_max = eq_s.cum_max()
    dd = (eq_s / running_max) - 1.0
    fig.add_trace(
        go.Scatter(x=time, y=dd.to_list(), mode="lines", name="Drawdown"),
        row=2,
        col=1,
    )

    q = results["quant"]
    metrics_text = (
        f"Return: {q['final_return']:.4f}<br>"
        f"Max DD: {q['max_drawdown']:.4f}<br>"
        f"Sharpe: {q['sharpe_ratio']:.3f}<br>"
        f"Win rate: {q['win_rate']:.2%}<br>"
        f"Trades: {q['total_trades']}"
    )

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        bordercolor="black",
        borderwidth=1,
        borderpad=6,
        bgcolor="white",
        opacity=0.85,
        align="left",
    )

    fig.update_layout(
        title="Strategy Performance",
        height=cfg.chart_height,
        width=cfg.chart_width,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

    return fig


def _plot_pair_breakdown(
    trades_df: pl.DataFrame,
    results: Dict[str, Any],
    cfg: StrategyMetricsConfig,
) -> go.Figure:
    df = trades_df

    # назви колонок — підлаштуй під реальні у себе, але це “дефолтний” набір
    for must in [cfg.trades_pair_col, cfg.trades_profit_col]:
        if must not in df.columns:
            raise ValueError(f"Missing required trades column: {must}")

    agg = (
        df.group_by(cfg.trades_pair_col)
        .agg(
            pl.count().alias("trades"),
            pl.sum(cfg.trades_profit_col).alias("pnl_sum"),
            pl.mean(cfg.trades_profit_col).alias("pnl_mean"),
            (pl.col(cfg.trades_profit_col) > 0).mean().alias("win_rate"),
        )
        .sort("pnl_sum", descending=True)
    )

    top = agg.head(cfg.top_n_pairs)
    pairs = top.get_column(cfg.trades_pair_col).to_list()
    pnl = top.get_column("pnl_sum").to_list()
    wr = top.get_column("win_rate").to_list()
    ntr = top.get_column("trades").to_list()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        subplot_titles=[f"Top {len(pairs)} Pairs by PnL (sum)", "Win rate by pair"],
    )

    fig.add_trace(
        go.Bar(
            x=pairs,
            y=pnl,
            name="PnL sum",
            customdata=list(zip(ntr)),
            hovertemplate="Pair: %{x}<br>PnL sum: %{y}<br>Trades: %{customdata[0]}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=pairs,
            y=[v * 100 for v in wr],
            name="Win rate (%)",
            customdata=list(zip(ntr)),
            hovertemplate="Pair: %{x}<br>Win rate: %{y:.2f}%<br>Trades: %{customdata[0]}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Pair Breakdown",
        height=cfg.chart_height,
        width=cfg.chart_width,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="PnL (sum)", row=1, col=1)
    fig.update_yaxes(title_text="Win rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Pair", row=2, col=1)

    return fig


def _log_strategy_metrics_to_mlflow(
    results: Dict[str, Any],
    plots: Dict[str, FigureLike],
) -> None:
    # 1) metrics
    quant = results.get("quant", {})
    if isinstance(quant, dict):
        mlflow.log_metrics({f"strategy.{k}": v for k, v in quant.items() if isinstance(v, (int, float))})

    # 2) artifacts (html)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, figs in plots.items():
            if figs is None:
                continue
            fig_list = figs if isinstance(figs, list) else [figs]
            metric_dir = tmp / name
            metric_dir.mkdir(exist_ok=True)

            for i, fig in enumerate(fig_list):
                if fig is None:
                    continue
                html_path = metric_dir / f"{i}.html"
                fig.write_html(str(html_path))

            mlflow.log_artifacts(str(metric_dir), artifact_path=f"strategy_metrics/{name}")
