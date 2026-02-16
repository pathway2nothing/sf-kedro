"""Backtesting utilities."""

from typing import Dict
from loguru import logger

import signalflow as sf
from signalflow.analytic.strategy import *


def run_backtest(
    raw_data: sf.RawData,
    validated_signals: sf.Signals,
    strategy_config: Dict,
) -> Dict:
    """
    Run backtest with validated signals.

    Args:
        raw_data: Raw market data
        validated_signals: Validated signals
        strategy_config: Strategy configuration

    Returns:
        Backtest results dict
    """
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.strategy.runner import OptimizedBacktestRunner

    # Create components
    strategy_store = DuckDbStrategyStore(
        strategy_config.get("db_path", "strategy.duckdb")
    )
    strategy_store.init()

    executor = VirtualSpotExecutor(
        fee_rate=strategy_config.get("fee_rate", 0.001),
        slippage_pct=strategy_config.get("slippage_pct", 0.001),
    )

    broker = BacktestBroker(executor=executor, store=strategy_store)

    entry_rules = []
    for rule_config in strategy_config.get("entry_rules", []):
        rule_type = rule_config.pop("type")
        entry_rules.append(
            sf.get_component(
                type=sf.SfComponentType.STRATEGY_ENTRY_RULE, name=rule_type
            )(**rule_config)
        )

    exit_rules = []
    for rule_config in strategy_config.get("exit_rules", []):
        rule_type = rule_config.pop("type")
        exit_rules.append(
            sf.get_component(
                type=sf.SfComponentType.STRATEGY_EXIT_RULE, name=rule_type
            )(**rule_config)
        )

    metrics = []
    for metric_config in strategy_config.get("metrics", []):
        metric_type = metric_config.pop("type")
        metrics.append(
            sf.get_component(type=sf.SfComponentType.STRATEGY_METRIC, name=metric_type)(
                **metric_config
            )
        )

    initial_capital = strategy_config.get("initial_capital", 100000)

    runner = OptimizedBacktestRunner(
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

    logger.success(f"\nFinal Equity: ${results['final_equity']:.2f}")
    logger.info(f"Total Return: {results['final_return'] * 100:.2f}%")
    logger.info(f"Trades Executed: {results['total_trades']}")
    logger.info("Recent Trades:")
    logger.info(results["trades_df"].tail(10))

    return results, final_state
