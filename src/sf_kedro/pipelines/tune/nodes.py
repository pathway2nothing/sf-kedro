"""Tune pipeline nodes - Optuna optimization.

Usage:
    kedro run --pipeline=tune --params='flow_id:grid_sma,level=detector'
    kedro run --pipeline=tune --params='flow_id:grid_sma,level=strategy,n_trials=100'
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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


def load_tune_data(
    flow_id: str,
    level: str = "detector",
    n_trials: int = 50,
) -> tuple[dict[str, Any], sf.RawData]:
    """Load data for optimization.

    Args:
        flow_id: Flow identifier
        level: Optimization level (detector/strategy)
        n_trials: Number of Optuna trials

    Returns:
        Tuple of (config, raw_data)
    """
    config = load_flow_config(flow_id)
    config["tune_level"] = level
    config["n_trials"] = n_trials

    logger.info(f"Loading data for tuning: {flow_id}, level={level}, n_trials={n_trials}")

    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    period_config = data_config.get("period", {})

    start_date = _parse_date(period_config.get("start", {}))
    end_date = _parse_date(period_config.get("end", {}))
    pairs = data_config.get("pairs", ["BTCUSDT"])
    db_path = store_config.get("db_path", "data/01_raw/market.duckdb")

    raw_data = sf.load(source=db_path, pairs=pairs, start=start_date, end=end_date)

    logger.success(f"Loaded {raw_data.data['spot'].height} rows for optimization")

    return config, raw_data


def run_optimization(
    config: dict[str, Any],
    raw_data: sf.RawData,
) -> dict[str, Any]:
    """Run Optuna optimization.

    Args:
        config: Flow configuration
        raw_data: Market data

    Returns:
        Optimization results with best parameters
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    level = config.get("tune_level", "detector")
    flow_id = config["flow_id"]
    n_trials = config.get("n_trials", 50)

    logger.info(f"Running Optuna optimization for {level}...")

    if level == "detector":
        results = _optimize_detector(config, raw_data, n_trials, flow_id)
    elif level == "strategy":
        results = _optimize_strategy(config, raw_data, n_trials, flow_id)
    else:
        raise ValueError(f"Unknown tune level: {level}")

    logger.success(f"Best params: {results['best_params']}")
    logger.success(f"Best value: {results['best_value']:.4f}")

    return results


def _optimize_detector(
    config: dict[str, Any],
    raw_data: sf.RawData,
    n_trials: int,
    flow_id: str,
) -> dict[str, Any]:
    """Optimize detector parameters."""
    import optuna

    detector_config = config.get("detector", {})
    detector_type = detector_config.get("type", "example/sma_cross")

    # Get detector search space
    search_space = config.get("tune", {}).get("detector", {})
    if not search_space:
        # Default search space for SMA cross detector
        search_space = {
            "fast_period": {"type": "int", "low": 5, "high": 100},
            "slow_period": {"type": "int", "low": 50, "high": 500},
        }

    # Get labeling config for profiling signals
    labeling_config = config.get("labeling", {}).copy()
    labeling_type = labeling_config.pop("type", "fixed_horizon")

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters
        params = {}
        for name, space in search_space.items():
            if space["type"] == "int":
                params[name] = trial.suggest_int(name, space["low"], space["high"])
            elif space["type"] == "float":
                params[name] = trial.suggest_float(name, space["low"], space["high"])
            elif space["type"] == "log_float":
                params[name] = trial.suggest_float(name, space["low"], space["high"], log=True)

        # Ensure slow_period > fast_period for SMA cross
        if "fast_period" in params and "slow_period" in params and params["slow_period"] <= params["fast_period"]:
            return float("-inf")

        try:
            # Create detector with sampled params and compute features
            from sf_kedro.utils.detection import run_detection_with_detector

            detector = sf.get_component(type=sf.SfComponentType.DETECTOR, name=detector_type)(**params)
            signals = run_detection_with_detector(detector, raw_data)

            if signals.value.height == 0:
                return float("-inf")

            # Apply labeling to evaluate signal quality
            labeler = sf.get_component(type=sf.SfComponentType.LABELER, name=labeling_type)(**labeling_config)
            spot_data = raw_data.data.get("spot")
            labeled = labeler.compute(spot_data, signals=signals)

            # Calculate signal quality metric
            label_col = labeling_config.get("out_col", "label")
            if label_col in labeled.columns:
                # Count valid signals (RISE or FALL, not NONE)
                valid_signals = labeled.filter((labeled[label_col] == 1) | (labeled[label_col] == 2)).height
                total_signals = labeled.height
                if total_signals > 0:
                    return valid_signals / total_signals  # Signal validity ratio
            else:
                # Fallback: optimize for signal count
                return float(signals.value.height) / 1000  # Normalize

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return float("-inf")

        return 0.0

    study = optuna.create_study(direction="maximize", study_name=f"{flow_id}_detector")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "level": "detector",
        "detector_type": detector_type,
    }


def _optimize_strategy(
    config: dict[str, Any],
    raw_data: sf.RawData,
    n_trials: int,
    flow_id: str,
) -> dict[str, Any]:
    """Optimize strategy parameters using backtesting."""
    import optuna
    from signalflow.data.strategy_store import DuckDbStrategyStore
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor
    from signalflow.strategy.runner import BacktestRunner

    # First, detect signals with current detector config
    from sf_kedro.utils.detection import run_detection

    signals = run_detection(config, raw_data)

    logger.info(f"Using {signals.value.height} signals for strategy optimization")

    # Get strategy search space
    search_space = config.get("tune", {}).get("strategy", {})
    if not search_space:
        # Default search space
        search_space = {
            "take_profit_pct": {"type": "float", "low": 0.005, "high": 0.05},
            "stop_loss_pct": {"type": "float", "low": 0.005, "high": 0.03},
        }

    strategy_config = config.get("strategy", {})
    initial_capital = strategy_config.get("initial_capital", 10000.0)

    def objective(trial: optuna.Trial) -> float:
        # Sample parameters
        params = {}
        for name, space in search_space.items():
            if space["type"] == "int":
                params[name] = trial.suggest_int(name, space["low"], space["high"])
            elif space["type"] == "float":
                params[name] = trial.suggest_float(name, space["low"], space["high"])

        try:
            # Create temporary store
            db_path = f"data/07_model_output/tune_{flow_id}_{trial.number}.duckdb"
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
                        filter_cls = sf.get_component(type=sf.SfComponentType.STRATEGY_ENTRY_RULE, name=filter_type)
                        filters.append(filter_cls(**filter_config))
                    rule_config["entry_filters"] = filters
                rule_cls = sf.get_component(type=sf.SfComponentType.STRATEGY_ENTRY_RULE, name=rule_type)
                entry_rules.append(rule_cls(**rule_config))

            # Build exit rules with sampled params
            exit_rules = [
                sf.get_component(type=sf.SfComponentType.STRATEGY_EXIT_RULE, name="tp_sl")(
                    take_profit_pct=params.get("take_profit_pct", 0.02),
                    stop_loss_pct=params.get("stop_loss_pct", 0.01),
                )
            ]

            runner = BacktestRunner(
                strategy_id=f"{flow_id}_tune_{trial.number}",
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

            # Return total return as optimization target
            return results.get("final_return", 0.0)

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return float("-inf")

    study = optuna.create_study(direction="maximize", study_name=f"{flow_id}_strategy")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "level": "strategy",
    }


def save_best_params(
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Save best parameters.

    Args:
        config: Flow configuration
        results: Optimization results
    """
    flow_id = config["flow_id"]
    level = results["level"]

    output_dir = Path(f"data/06_models/{flow_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / f"best_params_{level}.yml"

    with open(params_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    # Log detailed results
    logger.info("=" * 50)
    logger.success(f"Optuna Optimization: {config.get('flow_name', config['flow_id'])}")
    logger.info("-" * 50)

    best_params = results.get("best_params", {})
    best_value = results.get("best_value", 0)
    n_trials = results.get("n_trials", 0)

    logger.info(f"  Level:           {level}")
    logger.info(f"  Trials:          {n_trials}")
    logger.info(f"  Best value:      {best_value:.4f}")
    logger.info("  Best params:")
    for k, v in best_params.items():
        if isinstance(v, float):
            logger.info(f"    - {k}: {v:.4f}")
        else:
            logger.info(f"    - {k}: {v}")

    logger.info(f"  Saved to:        {params_path}")
    logger.info("=" * 50)

    # Send to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config.get("enabled", False):
        _send_telegram_notification(telegram_config, config, results)


def _send_telegram_notification(telegram_config: dict, config: dict, results: dict) -> None:
    """Send Telegram notification about optimization results."""
    try:
        from sf_kedro.utils.telegram import send_message_to_telegram

        level = results["level"]
        best_params = results["best_params"]
        best_value = results["best_value"]

        message = f"""
🎯 <b>SignalFlow Optimization Complete</b>

🔍 Flow: {config.get("flow_name", config["flow_id"])}
📊 Level: {level}
🏆 Best value: {best_value:.4f}
📝 Best params:
"""
        for k, v in best_params.items():
            message += f"  • {k}: {v}\n"

        send_message_to_telegram(
            message=message,
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )
        logger.info("Sent Telegram notification")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}")
