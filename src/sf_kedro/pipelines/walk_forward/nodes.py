"""Walk-Forward validation nodes for SignalFlow pipelines.

Walk-Forward validation (also known as rolling window validation) is a
time-series cross-validation technique that:

1. Divides data into sequential windows
2. Trains on window N, validates on window N+1
3. Rolls forward and repeats
4. Aggregates results across all windows

This provides more realistic out-of-sample performance estimates than
simple train/test splits.

Example window configuration:
    |------ Train 1 ------|-- Test 1 --|
                |------ Train 2 ------|-- Test 2 --|
                          |------ Train 3 ------|-- Test 3 --|
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import mlflow
import numpy as np
import polars as pl
from signalflow.core import RawData, Signals
from signalflow.validator import SklearnSignalValidator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def train_period(self) -> tuple[datetime, datetime]:
        return (self.train_start, self.train_end)

    @property
    def test_period(self) -> tuple[datetime, datetime]:
        return (self.test_start, self.test_end)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""

    windows: list[WalkForwardWindow]
    window_results: list[dict[str, Any]]
    aggregated_metrics: dict[str, float]
    trades_df: pl.DataFrame
    equity_curve: pl.DataFrame

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "Walk-Forward Validation Results",
            "=" * 60,
            f"Total Windows: {len(self.windows)}",
            f"Total Trades: {self.trades_df.height}",
            "",
            "Aggregated Metrics:",
        ]

        for metric, value in self.aggregated_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f}")
            else:
                lines.append(f"  {metric}: {value}")

        lines.extend(
            [
                "",
                "Per-Window Results:",
            ]
        )

        for i, result in enumerate(self.window_results):
            lines.append(
                f"  Window {i + 1}: Sharpe={result.get('sharpe', 0):.3f}, "
                f"Return={result.get('total_return', 0):.2%}, "
                f"Trades={result.get('n_trades', 0)}"
            )

        return "\n".join(lines)


def create_walk_forward_windows(
    raw_data: RawData,
    config: dict[str, Any],
) -> list[WalkForwardWindow]:
    """Create walk-forward validation windows.

    Args:
        raw_data: Raw market data to determine time range
        config: Window configuration including:
            - train_size: Training window size in days
            - test_size: Test window size in days
            - step_size: Step between windows in days (default: test_size)
            - n_windows: Number of windows (optional, calculated if not provided)
            - min_train_bars: Minimum bars required for training

    Returns:
        List of WalkForwardWindow objects
    """
    view = raw_data.view()
    df = view.to_polars("spot")

    # Get time range
    min_ts = df.select("timestamp").min().item()
    max_ts = df.select("timestamp").max().item()

    train_size = timedelta(days=config.get("train_size", 60))
    test_size = timedelta(days=config.get("test_size", 14))
    step_size = timedelta(days=config.get("step_size", config.get("test_size", 14)))
    n_windows = config.get("n_windows")

    windows = []
    window_id = 0

    # Start first window
    train_start = min_ts

    while True:
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        # Check if we have enough data for this window
        if test_end > max_ts:
            break

        if n_windows is not None and window_id >= n_windows:
            break

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        window_id += 1
        train_start = train_start + step_size

    logger.info(f"Created {len(windows)} walk-forward windows")

    # Log window info to MLflow
    mlflow.log_params(
        {
            "wf.n_windows": len(windows),
            "wf.train_size_days": config.get("train_size", 60),
            "wf.test_size_days": config.get("test_size", 14),
            "wf.step_size_days": config.get("step_size", config.get("test_size", 14)),
        }
    )

    return windows


def run_walk_forward_validation(
    raw_data: RawData,
    signals: Signals,
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    windows: list[WalkForwardWindow],
    config: dict[str, Any],
) -> WalkForwardResult:
    """Run walk-forward validation across all windows.

    Args:
        raw_data: Raw market data
        signals: Trading signals
        features_df: Feature DataFrame with timestamp column
        labels_df: Labels DataFrame with timestamp column
        windows: List of WalkForwardWindow objects
        config: Validation configuration including:
            - validator: Validator configuration
            - strategy: Strategy configuration
            - retrain: Whether to retrain validator each window

    Returns:
        WalkForwardResult with aggregated metrics
    """
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.strategy.component.entry.signal import SignalEntryRule
    from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
    from signalflow.strategy.runner import BacktestRunner

    validator_config = config.get("validator", {})
    strategy_config = config.get("strategy", {})
    retrain = config.get("retrain", True)
    initial_capital = strategy_config.get("initial_capital", 10_000.0)

    window_results = []
    all_trades = []
    all_equity_points = []
    current_capital = initial_capital

    signals_df = signals.value

    for window in windows:
        logger.info(
            f"Processing window {window.window_id + 1}/{len(windows)}: "
            f"Train {window.train_start} to {window.train_end}, "
            f"Test {window.test_start} to {window.test_end}"
        )

        # Filter data for this window
        train_features = features_df.filter(
            (pl.col("timestamp") >= window.train_start) & (pl.col("timestamp") < window.train_end)
        )
        train_labels = labels_df.filter(
            (pl.col("timestamp") >= window.train_start) & (pl.col("timestamp") < window.train_end)
        )

        test_features = features_df.filter(
            (pl.col("timestamp") >= window.test_start) & (pl.col("timestamp") < window.test_end)
        )
        test_signals = signals_df.filter(
            (pl.col("timestamp") >= window.test_start) & (pl.col("timestamp") < window.test_end)
        )

        # Skip window if insufficient data
        if train_features.height < 100 or test_signals.height == 0:
            logger.warning(f"Skipping window {window.window_id}: insufficient data")
            continue

        # Train validator on this window (if retraining)
        if retrain:
            validator = _train_validator(
                train_features,
                train_labels,
                validator_config,
            )

            # Validate test signals
            validated_signals = _validate_signals(
                Signals(test_signals),
                test_features,
                validator,
            )
        else:
            validated_signals = Signals(test_signals)

        # Run backtest on test period
        entry_config = strategy_config.get("entry", {})
        exit_config = strategy_config.get("exit", {})

        entry_rule = SignalEntryRule(
            base_position_size=current_capital * entry_config.get("position_size_pct", 0.1),
            max_positions_per_pair=entry_config.get("max_positions_per_pair", 1),
            max_total_positions=entry_config.get("max_total_positions", 5),
        )

        exit_rule = TakeProfitStopLossExit(
            take_profit_pct=exit_config.get("take_profit_pct", 0.02),
            stop_loss_pct=exit_config.get("stop_loss_pct", 0.01),
        )

        broker = BacktestBroker(executor=VirtualSpotExecutor(fee_rate=strategy_config.get("fee_rate", 0.001)))

        runner = BacktestRunner(
            strategy_id=f"wf_window_{window.window_id}",
            broker=broker,
            entry_rules=[entry_rule],
            exit_rules=[exit_rule],
            initial_capital=current_capital,
        )

        # Filter raw_data for test period
        view = raw_data.view()
        view.to_polars("spot").filter(
            (pl.col("timestamp") >= window.test_start) & (pl.col("timestamp") < window.test_end)
        )

        try:
            state = runner.run(raw_data=raw_data, signals=validated_signals)
            trades_df = runner.trades_df

            # Calculate window metrics
            window_pnl = state.capital - current_capital
            window_return = window_pnl / current_capital if current_capital > 0 else 0

            # Calculate Sharpe for this window
            if trades_df.height > 0:
                returns = trades_df.select("pnl").to_series() / current_capital
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                win_rate = trades_df.filter(pl.col("pnl") > 0).height / trades_df.height
            else:
                sharpe = 0
                win_rate = 0

            window_result = {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_trades": trades_df.height,
                "total_return": window_return,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "final_capital": state.capital,
            }

            # Track trades and equity
            if trades_df.height > 0:
                trades_df = trades_df.with_columns(pl.lit(window.window_id).alias("window_id"))
                all_trades.append(trades_df)

            # Update capital for next window (compound returns)
            current_capital = state.capital

            all_equity_points.append(
                {
                    "window_id": window.window_id,
                    "timestamp": window.test_end,
                    "capital": current_capital,
                }
            )

        except Exception as e:
            logger.warning(f"Window {window.window_id} failed: {e}")
            window_result = {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_trades": 0,
                "total_return": 0,
                "sharpe": 0,
                "win_rate": 0,
                "final_capital": current_capital,
                "error": str(e),
            }

        window_results.append(window_result)

        # Log per-window metrics to MLflow
        mlflow.log_metrics(
            {
                f"wf.window_{window.window_id}.return": window_result["total_return"],
                f"wf.window_{window.window_id}.sharpe": window_result["sharpe"],
                f"wf.window_{window.window_id}.n_trades": window_result["n_trades"],
            }
        )

    # Aggregate results
    all_trades_df = pl.concat(all_trades) if all_trades else pl.DataFrame()
    equity_curve = pl.DataFrame(all_equity_points)

    aggregated_metrics = _aggregate_metrics(window_results, initial_capital, current_capital)

    # Log aggregated metrics to MLflow
    mlflow.log_metrics({f"wf.agg.{k}": v for k, v in aggregated_metrics.items() if isinstance(v, (int, float))})

    return WalkForwardResult(
        windows=windows,
        window_results=window_results,
        aggregated_metrics=aggregated_metrics,
        trades_df=all_trades_df,
        equity_curve=equity_curve,
    )


def _train_validator(
    features_df: pl.DataFrame,
    labels_df: pl.DataFrame,
    config: dict[str, Any],
) -> SklearnSignalValidator:
    """Train validator for a single window."""
    # Merge features with labels
    train_df = features_df.join(
        labels_df.select(["pair", "timestamp", "label"]),
        on=["pair", "timestamp"],
        how="inner",
    ).drop_nulls()

    if train_df.height == 0:
        raise ValueError("No training data after joining features and labels")

    feature_cols = [col for col in train_df.columns if col not in ["timestamp", "pair", "label"]]

    X_train = train_df.select(feature_cols)
    y_train = train_df.select("label")

    validator = SklearnSignalValidator(
        model_type=config.get("model_type", "lightgbm"),
        model_params=config.get("model_params", {}),
    )
    validator.fit(X_train, y_train)

    return validator


def _validate_signals(
    signals: Signals,
    features_df: pl.DataFrame,
    validator: SklearnSignalValidator,
) -> Signals:
    """Validate signals using trained validator."""
    # This is a simplified version - actual implementation would use
    # validator.validate_signals() method
    return signals


def _aggregate_metrics(
    window_results: list[dict[str, Any]],
    initial_capital: float,
    final_capital: float,
) -> dict[str, float]:
    """Aggregate metrics across all windows."""
    if not window_results:
        return {"total_return": 0, "sharpe": 0, "n_windows": 0, "n_trades": 0}

    # Total return (compounded)
    total_return = (final_capital - initial_capital) / initial_capital

    # Average per-window metrics
    sharpes = [r["sharpe"] for r in window_results if r.get("sharpe")]
    returns = [r["total_return"] for r in window_results if r.get("total_return") is not None]
    win_rates = [r["win_rate"] for r in window_results if r.get("win_rate")]
    n_trades = sum(r.get("n_trades", 0) for r in window_results)

    # Calculate aggregate sharpe from returns
    if returns:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        aggregate_sharpe = mean_return / std_return if std_return > 0 else 0
    else:
        aggregate_sharpe = 0

    return {
        "total_return": total_return,
        "annualized_return": total_return * (365 / (len(window_results) * 14)),  # Approximate
        "sharpe_avg": np.mean(sharpes) if sharpes else 0,
        "sharpe_std": np.std(sharpes) if sharpes else 0,
        "sharpe_aggregate": aggregate_sharpe,
        "win_rate_avg": np.mean(win_rates) if win_rates else 0,
        "n_windows": len(window_results),
        "n_trades": n_trades,
        "final_capital": final_capital,
    }


def save_walk_forward_results(
    result: WalkForwardResult,
    output_dir: str,
) -> str:
    """Save walk-forward validation results.

    Args:
        result: WalkForwardResult object
        output_dir: Directory to save results

    Returns:
        Path to saved results
    """
    import json
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = output_path / "walk_forward_summary.txt"
    with open(summary_path, "w") as f:
        f.write(result.summary())

    # Save window results as JSON
    window_results_path = output_path / "window_results.json"
    serializable_results = []
    for r in result.window_results:
        serialized = {}
        for k, v in r.items():
            if isinstance(v, datetime):
                serialized[k] = v.isoformat()
            else:
                serialized[k] = v
        serializable_results.append(serialized)

    with open(window_results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    # Save aggregated metrics
    metrics_path = output_path / "aggregated_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.aggregated_metrics, f, indent=2, default=str)

    # Save trades DataFrame
    if result.trades_df.height > 0:
        trades_path = output_path / "all_trades.parquet"
        result.trades_df.write_parquet(str(trades_path))

    # Save equity curve
    if result.equity_curve.height > 0:
        equity_path = output_path / "equity_curve.parquet"
        result.equity_curve.write_parquet(str(equity_path))

    # Log artifacts to MLflow
    mlflow.log_artifact(str(summary_path))
    mlflow.log_artifact(str(window_results_path))
    mlflow.log_artifact(str(metrics_path))

    if result.trades_df.height > 0:
        mlflow.log_artifact(str(output_path / "all_trades.parquet"))

    logger.info(f"Saved walk-forward results to {output_path}")

    return str(output_path)


def compare_walk_forward_strategies(
    results: dict[str, WalkForwardResult],
) -> pl.DataFrame:
    """Compare multiple strategies from walk-forward validation.

    Args:
        results: Dict mapping strategy name to WalkForwardResult

    Returns:
        Comparison DataFrame
    """
    comparison_data = []

    for strategy_name, result in results.items():
        metrics = result.aggregated_metrics.copy()
        metrics["strategy"] = strategy_name
        comparison_data.append(metrics)

    comparison_df = pl.DataFrame(comparison_data)

    # Sort by aggregate sharpe
    comparison_df = comparison_df.sort("sharpe_aggregate", descending=True)

    return comparison_df
