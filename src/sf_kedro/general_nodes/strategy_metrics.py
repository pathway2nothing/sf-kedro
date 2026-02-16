from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import plotly.graph_objects as go
from loguru import logger

import signalflow as sf
from sf_kedro.custom_modules.strategy_metrics import *
from sf_kedro.utils.telegram import send_plots_to_telegram
from signalflow import RawData, StrategyState


def log_last_state_metrics(backtest_results: dict) -> dict:
    """
    Calculate and log backtest metrics.

    Args:
        backtest_results: Results from run_backtest

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "final_return": backtest_results.get("final_return", 0),
        "max_drawdown": backtest_results.get("max_drawdown", 0),
        "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
        "win_rate": backtest_results.get("win_rate", 0),
        "total_trades": backtest_results.get("total_trades", 0),
        "final_equity": backtest_results.get("final_equity", 0),
    }

    mlflow.log_metrics(
        {
            "backtest.return": metrics["final_return"],
            "backtest.max_drawdown": metrics["max_drawdown"],
            "backtest.sharpe": metrics["sharpe_ratio"],
            "backtest.win_rate": metrics["win_rate"],
            "backtest.total_trades": metrics["total_trades"],
            "backtest.final_equity": metrics["final_equity"],
        }
    )

    metrics_df = backtest_results.get("metrics_df")
    if metrics_df is not None:
        output_path = Path("data/07_model_output/backtest/equity_curve.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.write_csv(output_path)
        mlflow.log_artifact(str(output_path), artifact_path="backtest")

    trades_df = backtest_results.get("trades_df")
    if trades_df is not None:
        output_path = Path("data/07_model_output/backtest/trades.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.write_parquet(output_path)
        mlflow.log_artifact(str(output_path), artifact_path="backtest")

    return metrics


def compute_strategy_metrics(
    backtest_results: dict[str, Any],
    params: dict[str, dict[str, Any]],
    telegram_config: dict[str, Any] | None = None,
    strategy_name: str | None = None,
    raw_data: RawData | None = None,
    state: StrategyState | None = None,
) -> tuple[dict[str, Any], dict[str, go.Figure | list[go.Figure]]]:

    logger.info(f"Computing strategy metrics for {len(params)} metric types")

    metrics: list[Any] = []
    for metric_type, metric_params in (params or {}).items():
        logger.debug(f"Loading strategy metric processor: {metric_type}")
        metric_cls = sf.get_component(
            type=sf.SfComponentType.STRATEGY_METRIC,
            name=metric_type,
        )
        metrics.append(metric_cls(**(metric_params or {})))

    summary = _build_strategy_summary(backtest_results)

    results: dict[str, Any] = {"quant": summary, "per_metric": {}}
    plots: dict[str, go.Figure | list[go.Figure]] = {}

    for metric in metrics:
        metric_name = metric.__class__.__name__
        logger.info(f"Processing strategy metric: {metric_name}")

        try:
            metric_plots = metric.plot(
                results=backtest_results,
                state=state,
                raw_data=raw_data,
            )

            plots[metric_name] = metric_plots

            n_plots = (
                len(metric_plots)
                if isinstance(metric_plots, list)
                else (1 if metric_plots is not None else 0)
            )
            results["per_metric"][metric_name] = {
                "n_plots": n_plots,
            }

            logger.info(f"Metric {metric_name}: generated {n_plots} plots")
        except Exception as e:
            logger.error(f"Failed to process strategy metric {metric_name}: {e}")
            plots[metric_name] = None
            results["per_metric"][metric_name] = None

    if mlflow.active_run():
        logger.info("Logging strategy metrics to MLflow")
        _log_metrics_to_mlflow(results, plots, artifact_root="strategy_metrics")
    else:
        logger.debug("No active MLflow run, skipping MLflow logging")

    if telegram_config and telegram_config.get("enabled", False):
        try:
            full_header = (
                f"Strategy: {strategy_name}" if strategy_name else "Strategy Metrics"
            )
            send_plots_to_telegram(
                plots=plots,
                bot_token=telegram_config.get("bot_token"),
                chat_id=telegram_config.get("chat_id"),
                header_message=full_header,
                image_width=telegram_config.get("image_width", 1400),
                image_height=telegram_config.get("image_height", 900),
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            if telegram_config.get("raise_on_error", False):
                raise

    logger.info(
        f"Successfully computed {len(results.get('per_metric', {}))} strategy metric types"
    )
    return results, plots


def save_strategy_plots(
    plots: dict[str, go.Figure | list[go.Figure]],
    output_dir: str,
) -> None:
    """
    Save strategy plots to disk as PNG and HTML files.
    Mirror save_signal_plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for metric_name, figures in plots.items():
        if figures is None:
            logger.warning(f"Skipping {metric_name}: no figures")
            continue

        metric_dir = output_path / metric_name
        metric_dir.mkdir(exist_ok=True)

        figs_list = figures if isinstance(figures, list) else [figures]
        logger.info(f"Saving {len(figs_list)} plots for {metric_name}")

        for i, fig in enumerate(figs_list):
            if fig is None:
                continue
            png_path = metric_dir / f"{i}.png"
            fig.write_image(str(png_path), width=1400, height=900, scale=2)

            html_path = metric_dir / f"{i}.html"
            fig.write_html(str(html_path))

            total_saved += 1

        logger.info(f"Saved {len(figs_list)} plots to {metric_dir}")

    logger.info(f"Total saved {total_saved} plots to {output_path}")


def _build_strategy_summary(backtest_results: dict[str, Any]) -> dict[str, Any]:
    """
    Try to extract standard scalars from backtest_results.
    Keep it tolerant to missing keys.
    """

    def _sf(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _si(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    return {
        "final_return": _sf(backtest_results.get("final_return", 0.0)),
        "max_drawdown": _sf(backtest_results.get("max_drawdown", 0.0)),
        "sharpe_ratio": _sf(backtest_results.get("sharpe_ratio", 0.0)),
        "win_rate": _sf(backtest_results.get("win_rate", 0.0)),
        "total_trades": _si(backtest_results.get("total_trades", 0)),
        "final_equity": _sf(backtest_results.get("final_equity", 0.0)),
        "initial_capital": _sf(backtest_results.get("initial_capital", 0.0)),
    }


def _log_metrics_to_mlflow(
    results: dict[str, Any],
    plots: dict[str, go.Figure | list[go.Figure] | None],
    artifact_root: str,
) -> None:
    """
    Same style as signal _log_metrics_to_mlflow:
    - logs numeric entries recursively
    - logs plots as html artifacts
    """
    _log_dict_metrics(prefix="strategy", obj=results)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for metric_name, figures in plots.items():
            if figures is None:
                continue

            metric_dir = tmpdir_path / metric_name
            metric_dir.mkdir(exist_ok=True)

            figs_list = figures if isinstance(figures, list) else [figures]
            for i, fig in enumerate(figs_list):
                if fig is None:
                    continue
                html_path = metric_dir / f"{i}.html"
                fig.write_html(str(html_path))

            mlflow.log_artifacts(
                str(metric_dir), artifact_path=f"{artifact_root}/{metric_name}"
            )


def _log_dict_metrics(prefix: str, obj: Any) -> None:
    """
    Recursively logs numeric values as mlflow metrics with dot-separated keys.
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            _log_dict_metrics(f"{prefix}.{k}", v)
        return

    if isinstance(obj, (int, float)):
        mlflow.log_metric(prefix, float(obj))
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _log_dict_metrics(f"{prefix}[{i}]", v)
        return
