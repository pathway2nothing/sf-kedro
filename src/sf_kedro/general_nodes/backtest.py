"""Backtesting utilities."""

import signalflow as sf
from loguru import logger
from signalflow.analytic.strategy import *  # noqa: F403


def run_backtest(
    raw_data: sf.RawData,
    validated_signals: sf.Signals,
    strategy_config: dict,
) -> dict:
    """
    Run backtest with validated signals.

    Args:
        raw_data: Raw market data
        validated_signals: Validated signals
        strategy_config: Strategy configuration

    Returns:
        Backtest results dict
    """
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.strategy.runner import BacktestRunner

    # Create components
    strategy_store = DuckDbStrategyStore(strategy_config.get("db_path", "strategy.duckdb"))
    strategy_store.init()

    executor = VirtualSpotExecutor(
        fee_rate=strategy_config.get("fee_rate", 0.001),
        slippage_pct=strategy_config.get("slippage_pct", 0.001),
    )

    broker = BacktestBroker(executor=executor, store=strategy_store)

    entry_rules = []
    for rule_config in strategy_config.get("entry_rules", []):
        rule_config = rule_config.copy()  # Don't mutate original
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

    exit_rules = []
    for rule_config in strategy_config.get("exit_rules", []):
        rule_type = rule_config.pop("type")
        exit_rules.append(sf.get_component(type=sf.SfComponentType.STRATEGY_EXIT_RULE, name=rule_type)(**rule_config))

    metrics = []
    for metric_config in strategy_config.get("metrics", []):
        metric_type = metric_config.pop("type")
        metrics.append(sf.get_component(type=sf.SfComponentType.STRATEGY_METRIC, name=metric_type)(**metric_config))

    initial_capital = strategy_config.get("initial_capital", 100000)

    runner = BacktestRunner(
        strategy_id=strategy_config.get("strategy_id", "default_strategy"),
        broker=broker,
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        metrics=metrics,
        initial_capital=initial_capital,
        data_key=strategy_config.get("data_key", "spot"),
    )

    final_state = runner.run(raw_data, validated_signals)
    results = runner.get_results()

    if "final_equity" in results:
        logger.success(f"\nFinal Equity: ${results['final_equity']:.2f}")
    if "final_return" in results:
        logger.info(f"Total Return: {results['final_return'] * 100:.2f}%")
    logger.info(f"Trades Executed: {results['total_trades']}")
    if results["trades_df"].height > 0:
        logger.info("Recent Trades:")
        logger.info(results["trades_df"].tail(10))

    return results, final_state
