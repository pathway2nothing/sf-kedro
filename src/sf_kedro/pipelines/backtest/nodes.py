"""Universal backtest pipeline nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import signalflow as sf
from loguru import logger

from sf_kedro.utils.flow_config import load_flow_config


def load_flow_data(
    flow_id: str,
    common_config: dict[str, Any],
) -> tuple[dict[str, Any], sf.RawData]:
    """Load flow configuration and market data.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma')
        common_config: Common configuration with defaults

    Returns:
        Tuple of (merged config, raw data)
    """
    # Load and merge flow config
    config = load_flow_config(flow_id)

    # Merge with common_config from parameters
    if common_config:
        from sf_kedro.utils.flow_config import deep_merge

        config = deep_merge(common_config.get("defaults", {}), config)
        if "telegram" in common_config:
            config["telegram"] = common_config["telegram"]

    logger.info(f"Running flow: {config.get('flow_name', flow_id)}")
    logger.info(f"Description: {config.get('description', 'N/A')}")

    # Load market data
    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    period_config = data_config.get("period", {})

    # Build period
    from datetime import datetime

    start = period_config.get("start", {})
    end = period_config.get("end", {})

    start_date = datetime(
        year=start.get("year", 2025),
        month=start.get("month", 1),
        day=start.get("day", 1),
    )
    end_date = datetime(
        year=end.get("year", 2025),
        month=end.get("month", 12),
        day=end.get("day", 31),
    )

    pairs = data_config.get("pairs", ["BTCUSDT"])

    # Load from DuckDB
    db_path = store_config.get("db_path", "data/01_raw/market.duckdb")

    raw_data = sf.load(
        source=db_path,
        pairs=pairs,
        start=start_date,
        end=end_date,
    )

    logger.success(f"Loaded {len(pairs)} pairs, {raw_data.data['spot'].height} rows")

    return config, raw_data


def run_flow_detection(
    config: dict[str, Any],
    raw_data: sf.RawData,
) -> sf.Signals:
    """Run signal detection based on flow config.

    Args:
        config: Flow configuration
        raw_data: Market data

    Returns:
        Detected signals
    """
    from signalflow.feature import FeaturePipeline

    detector_config = config.get("detector", {}).copy()
    detector_type = detector_config.pop("type", "example/sma_cross")

    detector = sf.get_component(
        type=sf.SfComponentType.DETECTOR,
        name=detector_type,
    )(**detector_config)

    # Extract spot data and compute features
    data_key = config.get("strategy", {}).get("data_key", "spot")
    df = raw_data.data[data_key]

    # Compute detector features if available
    if hasattr(detector, "features") and detector.features:
        pipeline = FeaturePipeline(features=detector.features)
        df = pipeline.compute(df)

    signals = detector.detect(df)

    logger.success(f"Detected {signals.value.height} signals")
    return signals


def run_flow_backtest(
    config: dict[str, Any],
    raw_data: sf.RawData,
    signals: sf.Signals,
) -> tuple[dict[str, Any], sf.StrategyState]:
    """Run backtest with flow configuration.

    Args:
        config: Flow configuration
        raw_data: Market data
        signals: Detected signals

    Returns:
        Tuple of (results dict, final state)
    """
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.strategy.runner import BacktestRunner

    strategy_config = config.get("strategy", {})
    output_config = config.get("output", {})

    # Create components
    db_path = output_config.get("db", f"data/07_model_output/strategy_{config['flow_id']}.duckdb")
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

        # Handle nested entry_filters
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

    # Build metrics
    metrics = []
    metrics_config = strategy_config.get("metrics", config.get("defaults", {}).get("metrics", []))
    for metric_config in metrics_config:
        metric_config = metric_config.copy()
        metric_type = metric_config.pop("type")
        # Add initial_capital if not specified
        if "initial_capital" not in metric_config and metric_type in ["total_return", "balance_allocation"]:
            metric_config["initial_capital"] = strategy_config.get("initial_capital", 10000.0)
        metrics.append(sf.get_component(type=sf.SfComponentType.STRATEGY_METRIC, name=metric_type)(**metric_config))

    initial_capital = strategy_config.get("initial_capital", 10000.0)

    runner = BacktestRunner(
        strategy_id=strategy_config.get("strategy_id", config["flow_id"]),
        broker=broker,
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        metrics=metrics,
        initial_capital=initial_capital,
        data_key=strategy_config.get("data_key", "spot"),
    )

    final_state = runner.run(raw_data, signals)
    results = runner.get_results()

    # Log results
    logger.info("=" * 50)
    logger.success(f"Backtest Complete: {config.get('flow_name', config['flow_id'])}")
    logger.info("-" * 50)
    logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
    if "final_equity" in results:
        logger.info(f"  Final Equity:    ${results['final_equity']:,.2f}")
    if "final_return" in results:
        ret_pct = results["final_return"] * 100
        emoji = "📈" if ret_pct > 0 else "📉" if ret_pct < 0 else "➡️"
        logger.info(f"  Total Return:    {emoji} {ret_pct:+.2f}%")
    logger.info(f"  Trades Executed: {results.get('total_trades', 0)}")
    if "win_rate" in results:
        logger.info(f"  Win Rate:        {results['win_rate'] * 100:.1f}%")
    if "max_drawdown" in results:
        logger.info(f"  Max Drawdown:    {results['max_drawdown'] * 100:.2f}%")
    if "sharpe_ratio" in results:
        logger.info(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    logger.info("=" * 50)

    return results, final_state


def compute_metrics(
    config: dict[str, Any],
    results: dict[str, Any],
    state: sf.StrategyState,
    raw_data: sf.RawData,
) -> dict[str, Any]:
    """Compute and format metrics for reporting.

    Args:
        config: Flow configuration
        results: Backtest results
        state: Final strategy state
        raw_data: Market data

    Returns:
        Metrics dictionary with plots
    """
    metrics = {
        "flow_id": config["flow_id"],
        "flow_name": config.get("flow_name", config["flow_id"]),
        "results": results,
        "config": config,
    }

    # Add summary metrics
    if "final_equity" in results:
        metrics["final_equity"] = results["final_equity"]
    if "final_return" in results:
        metrics["final_return"] = results["final_return"]
    if "total_trades" in results:
        metrics["total_trades"] = results["total_trades"]

    return metrics


def save_flow_plots(
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save plots and send to Telegram if configured.

    Args:
        config: Flow configuration
        metrics: Computed metrics
    """
    output_config = config.get("output", {})
    strategy_output = output_config.get("strategy", f"data/08_reporting/{config['flow_id']}/strategy")

    # Create output directory
    output_path = Path(strategy_output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Plots saved to: {output_path}")

    # Send to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config.get("enabled", False):
        try:
            from sf_kedro.utils.telegram import send_message_to_telegram

            message = f"""
📊 <b>SignalFlow Backtest Complete</b>

🔍 Flow: {metrics.get("flow_name", config["flow_id"])}
💰 Final Equity: ${metrics.get("final_equity", 0):.2f}
📈 Return: {metrics.get("final_return", 0) * 100:.2f}%
📊 Trades: {metrics.get("total_trades", 0)}
"""
            send_message_to_telegram(
                message=message,
                bot_token=telegram_config.get("bot_token"),
                chat_id=telegram_config.get("chat_id"),
            )
            logger.info("Sent Telegram notification")
        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")
