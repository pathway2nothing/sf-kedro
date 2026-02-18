"""Validate pipeline nodes - walk-forward validation.

Usage:
    kedro run --pipeline=validate --params='flow_id=grid_sma'
    kedro run --pipeline=validate --params='flow_id:grid_sma,n_folds=5'
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import signalflow as sf
import yaml
from loguru import logger

from sf_kedro.utils.flow_config import load_flow_config


def _parse_date(date_config: dict) -> datetime:
    """Parse date from config dict."""
    return datetime(
        year=date_config.get("year", 2025),
        month=date_config.get("month", 1),
        day=date_config.get("day", 1),
    )


def load_validation_data(
    flow_id: str,
    n_folds: int = 5,
) -> tuple[dict[str, Any], sf.RawData]:
    """Load data for walk-forward validation.

    Args:
        flow_id: Flow identifier
        n_folds: Number of validation folds

    Returns:
        Tuple of (config, raw_data)
    """
    config = load_flow_config(flow_id)
    config["n_folds"] = n_folds

    logger.info(f"Loading validation data: {flow_id}, n_folds={n_folds}")

    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    period_config = data_config.get("period", {})

    start_date = _parse_date(period_config.get("start", {}))
    end_date = _parse_date(period_config.get("end", {}))
    pairs = data_config.get("pairs", ["BTCUSDT"])
    db_path = store_config.get("db_path", "data/01_raw/market.duckdb")

    raw_data = sf.load(source=db_path, pairs=pairs, start=start_date, end=end_date)

    logger.success(f"Loaded {raw_data.data['spot'].height} rows for validation")

    return config, raw_data


def run_walk_forward(
    config: dict[str, Any],
    raw_data: sf.RawData,
) -> dict[str, Any]:
    """Run walk-forward validation.

    Walk-forward validation splits data into n_folds + 1 periods.
    For each fold i, we train on periods 0..i and test on period i+1.

    Args:
        config: Flow configuration
        raw_data: Market data

    Returns:
        Validation results per fold
    """

    n_folds = config.get("n_folds", 5)
    flow_id = config["flow_id"]

    logger.info(f"Running walk-forward validation with {n_folds} folds...")

    spot_data = raw_data.data.get("spot")
    if spot_data is None:
        raise ValueError("No spot data available")

    # Split data by time
    timestamps = spot_data["timestamp"].unique().sort()
    total_periods = len(timestamps)
    fold_size = total_periods // (n_folds + 1)

    if fold_size < 100:
        logger.warning(f"Small fold size: {fold_size} periods per fold")

    fold_results = []

    for fold in range(n_folds):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")

        # Define train/test split
        train_end_idx = (fold + 1) * fold_size
        test_end_idx = (fold + 2) * fold_size

        if test_end_idx > total_periods:
            test_end_idx = total_periods

        train_end_ts = timestamps[train_end_idx - 1]
        test_start_ts = timestamps[train_end_idx]
        test_end_ts = timestamps[min(test_end_idx - 1, total_periods - 1)]

        # Get test data
        test_data = spot_data.filter((pl.col("timestamp") >= test_start_ts) & (pl.col("timestamp") <= test_end_ts))

        if test_data.height == 0:
            logger.warning(f"Fold {fold + 1}: No test data")
            continue

        # Create RawData for test period
        pairs = config.get("data", {}).get("pairs", ["BTCUSDT"])
        test_raw_data = sf.RawData(
            datetime_start=test_start_ts,
            datetime_end=test_end_ts,
            pairs=pairs,
            data={"spot": test_data},
        )

        # Run detection on test data
        from sf_kedro.utils.detection import run_detection

        signals = run_detection(config, test_raw_data)

        if signals.value.height == 0:
            logger.warning(f"Fold {fold + 1}: No signals detected")
            fold_results.append(
                {
                    "fold": fold,
                    "train_end": str(train_end_ts),
                    "test_start": str(test_start_ts),
                    "test_end": str(test_end_ts),
                    "n_signals": 0,
                    "final_return": 0.0,
                    "total_trades": 0,
                }
            )
            continue

        # Run backtest on test period
        try:
            fold_result = _run_fold_backtest(
                config=config,
                raw_data=test_raw_data,
                signals=signals,
                fold=fold,
                flow_id=flow_id,
            )
            fold_result["train_end"] = str(train_end_ts)
            fold_result["test_start"] = str(test_start_ts)
            fold_result["test_end"] = str(test_end_ts)
            fold_result["n_signals"] = signals.value.height
            fold_results.append(fold_result)

            ret = fold_result.get("final_return", 0) * 100
            trades = fold_result.get("total_trades", 0)
            logger.info(f"Fold {fold + 1}: Return={ret:.2f}%, Trades={trades}")

        except Exception as e:
            logger.error(f"Fold {fold + 1} failed: {e}")
            fold_results.append(
                {
                    "fold": fold,
                    "train_end": str(train_end_ts),
                    "test_start": str(test_start_ts),
                    "test_end": str(test_end_ts),
                    "error": str(e),
                }
            )

    # Calculate aggregate metrics
    valid_folds = [f for f in fold_results if "final_return" in f]
    avg_return = sum(f.get("final_return", 0) for f in valid_folds) / len(valid_folds) if valid_folds else 0.0
    total_trades = sum(f.get("total_trades", 0) for f in valid_folds)

    results = {
        "flow_id": flow_id,
        "n_folds": n_folds,
        "valid_folds": len(valid_folds),
        "folds": fold_results,
        "avg_return": avg_return,
        "total_trades": total_trades,
    }

    logger.success(
        f"Walk-forward validation complete: {len(valid_folds)}/{n_folds} valid folds, "
        f"avg return={avg_return * 100:.2f}%"
    )

    return results


def _run_fold_backtest(
    config: dict[str, Any],
    raw_data: sf.RawData,
    signals: sf.Signals,
    fold: int,
    flow_id: str,
) -> dict[str, Any]:
    """Run backtest for a single fold."""
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.strategy.runner import BacktestRunner

    strategy_config = config.get("strategy", {})

    # Create temporary store
    db_path = f"data/07_model_output/validate_{flow_id}_fold{fold}.duckdb"
    strategy_store = DuckDbStrategyStore(db_path)
    strategy_store.init()

    executor = VirtualSpotExecutor(
        fee_rate=strategy_config.get("fee_rate", 0.001),
        slippage_pct=strategy_config.get("slippage_pct", 0.001),
    )

    broker = BacktestBroker(executor=executor, store=strategy_store)

    # Build entry rules
    entry_rules = []
    for rule_config in strategy_config.get("entry_rules", []):
        rule_config = rule_config.copy()
        rule_type = rule_config.pop("type")
        if "entry_filters" in rule_config:
            filters = []
            for filter_config in rule_config.pop("entry_filters"):
                filter_config = filter_config.copy()
                filter_type = filter_config.pop("type")
                filters.append(
                    sf.get_component(type=sf.SfComponentType.STRATEGY_ENTRY_RULE, name=filter_type)(**filter_config)
                )
            rule_config["entry_filters"] = filters
        entry_rules.append(sf.get_component(type=sf.SfComponentType.STRATEGY_ENTRY_RULE, name=rule_type)(**rule_config))

    # Build exit rules
    exit_rules = []
    for rule_config in strategy_config.get("exit_rules", []):
        rule_config = rule_config.copy()
        rule_type = rule_config.pop("type")
        exit_rules.append(sf.get_component(type=sf.SfComponentType.STRATEGY_EXIT_RULE, name=rule_type)(**rule_config))

    initial_capital = strategy_config.get("initial_capital", 10000.0)

    runner = BacktestRunner(
        strategy_id=f"{flow_id}_validate_fold{fold}",
        broker=broker,
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        metrics=[],
        initial_capital=initial_capital,
        data_key="spot",
    )

    runner.run(raw_data, signals)
    results = runner.get_results()

    # Clean up temp db
    Path(db_path).unlink(missing_ok=True)

    return {
        "fold": fold,
        "final_return": results.get("final_return", 0.0),
        "total_trades": results.get("total_trades", 0),
        "final_equity": results.get("final_equity", initial_capital),
    }


def save_validation_report(
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Save validation report.

    Args:
        config: Flow configuration
        results: Validation results
    """
    flow_id = config["flow_id"]

    output_dir = Path(f"data/08_reporting/{flow_id}/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "walk_forward_results.yml"

    with open(report_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    # Log detailed results
    logger.info("=" * 50)
    logger.success(f"Walk-Forward Validation: {config.get('flow_name', config['flow_id'])}")
    logger.info("-" * 50)

    n_folds = results.get("n_folds", 0)
    valid_folds = results.get("valid_folds", 0)
    avg_return = results.get("avg_return", 0)
    total_trades = results.get("total_trades", 0)

    logger.info(f"  Valid folds:     {valid_folds}/{n_folds}")
    avg_ret_pct = avg_return * 100
    emoji = "📈" if avg_ret_pct > 0 else "📉" if avg_ret_pct < 0 else "➡️"
    logger.info(f"  Avg Return:      {emoji} {avg_ret_pct:+.2f}%")
    logger.info(f"  Total trades:    {total_trades}")

    # Log per-fold results
    folds = results.get("folds", [])
    if folds:
        logger.info("  Per-fold results:")
        for fold_data in folds:
            fold_num = fold_data.get("fold", 0) + 1
            fold_ret = fold_data.get("final_return", 0) * 100
            fold_trades = fold_data.get("total_trades", 0)
            fold_emoji = "📈" if fold_ret > 0 else "📉" if fold_ret < 0 else "➡️"
            logger.info(f"    Fold {fold_num}: {fold_emoji} {fold_ret:+.2f}% ({fold_trades} trades)")

    logger.info(f"  Report saved:    {report_path}")
    logger.info("=" * 50)

    # Send to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config.get("enabled", False):
        _send_telegram_notification(telegram_config, config, results)


def _send_telegram_notification(telegram_config: dict, config: dict, results: dict) -> None:
    """Send Telegram notification about validation results."""
    try:
        from sf_kedro.utils.telegram import send_message_to_telegram

        message = f"""
✅ <b>SignalFlow Walk-Forward Validation Complete</b>

🔍 Flow: {config.get("flow_name", config["flow_id"])}
📊 Folds: {results.get("valid_folds", 0)}/{results.get("n_folds", 0)} valid
📈 Avg Return: {results.get("avg_return", 0) * 100:.2f}%
📊 Total Trades: {results.get("total_trades", 0)}
"""
        send_message_to_telegram(
            message=message,
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )
        logger.info("Sent Telegram notification")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}")
