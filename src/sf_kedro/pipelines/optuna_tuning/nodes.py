"""Optuna hyperparameter tuning nodes for SignalFlow pipelines.

This module provides nodes for:
- Creating and configuring Optuna studies
- Running hyperparameter optimization for validators, detectors, and strategies
- Logging results to MLflow
- Selecting best parameters
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import mlflow
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

import polars as pl
import signalflow as sf
from signalflow.core import Signals, RawData


logger = logging.getLogger(__name__)


def create_optuna_study(
    study_config: Dict[str, Any],
) -> optuna.Study:
    """Create an Optuna study with specified configuration.

    Args:
        study_config: Study configuration including:
            - study_name: Name of the study
            - direction: 'maximize' or 'minimize'
            - sampler: Sampler configuration ('tpe', 'cmaes', etc.)
            - pruner: Pruner configuration ('median', 'hyperband', etc.)
            - storage: Optional storage URL for distributed optimization
            - load_if_exists: Whether to load existing study

    Returns:
        Configured Optuna study
    """
    study_name = study_config.get("study_name", "signalflow_tuning")
    direction = study_config.get("direction", "maximize")
    storage = study_config.get("storage", None)
    load_if_exists = study_config.get("load_if_exists", True)

    # Configure sampler
    sampler_config = study_config.get("sampler", {"type": "tpe"})
    sampler = _create_sampler(sampler_config)

    # Configure pruner
    pruner_config = study_config.get("pruner", {"type": "median"})
    pruner = _create_pruner(pruner_config)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )

    logger.info(f"Created Optuna study: {study_name} (direction={direction})")

    # Log study info to MLflow
    mlflow.log_params(
        {
            "optuna.study_name": study_name,
            "optuna.direction": direction,
            "optuna.sampler": sampler_config.get("type", "tpe"),
            "optuna.pruner": pruner_config.get("type", "median"),
        }
    )

    return study


def _create_sampler(config: Dict[str, Any]) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler from config."""
    sampler_type = config.get("type", "tpe")

    if sampler_type == "tpe":
        return TPESampler(
            seed=config.get("seed", 42),
            n_startup_trials=config.get("n_startup_trials", 10),
        )
    elif sampler_type == "cmaes":
        return CmaEsSampler(seed=config.get("seed", 42))
    else:
        return TPESampler(seed=42)


def _create_pruner(config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    """Create Optuna pruner from config."""
    pruner_type = config.get("type", "median")

    if pruner_type == "median":
        return MedianPruner(
            n_startup_trials=config.get("n_startup_trials", 5),
            n_warmup_steps=config.get("n_warmup_steps", 0),
        )
    elif pruner_type == "hyperband":
        return HyperbandPruner(
            min_resource=config.get("min_resource", 1),
            max_resource=config.get("max_resource", 100),
        )
    else:
        return MedianPruner()


def tune_validator(
    study: optuna.Study,
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    tuning_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], object]:
    """Tune validator hyperparameters using Optuna.

    Args:
        study: Optuna study
        train_data: Training data dict with 'full' DataFrame
        val_data: Validation data dict with 'full' DataFrame
        tuning_config: Configuration including:
            - n_trials: Number of optimization trials
            - timeout: Timeout in seconds (optional)
            - validator_type: 'sklearn' or 'nn'
            - model_size: 'small', 'medium', 'large' for search space
            - metric: Metric to optimize ('accuracy', 'f1', 'sharpe', etc.)

    Returns:
        Tuple of (best_params, trained_validator)
    """
    from signalflow.validator import SklearnSignalValidator
    from sklearn.metrics import accuracy_score, f1_score

    n_trials = tuning_config.get("n_trials", 50)
    timeout = tuning_config.get("timeout", None)
    validator_type = tuning_config.get("validator_type", "sklearn")
    model_size = tuning_config.get("model_size", "medium")
    metric = tuning_config.get("metric", "accuracy")

    train_df = train_data["full"]
    val_df = val_data["full"]

    feature_cols = [
        col for col in train_df.columns if col not in ["timestamp", "pair", "label"]
    ]

    X_train = train_df.select(feature_cols)
    y_train = train_df.select("label")
    X_val = val_df.select(feature_cols)
    y_val = val_df.select("label")

    def objective(trial: optuna.Trial) -> float:
        # Get hyperparameters from SignalFlow's tune() method
        params = SklearnSignalValidator.tune(trial, model_size=model_size)

        # Create and train validator
        validator = SklearnSignalValidator(
            model_type=params.pop("model_type", "lightgbm"),
            model_params=params,
        )
        validator.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = validator.model.predict(X_val.to_pandas())
        y_true = y_val.to_pandas()["label"]

        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, average="weighted")
        else:
            score = accuracy_score(y_true, y_pred)

        # Log intermediate result to MLflow
        mlflow.log_metric(f"trial_{trial.number}_score", score)

        return score

    # Run optimization
    logger.info(f"Starting validator tuning: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Best {metric}: {best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # Log best results to MLflow
    mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})
    mlflow.log_metric(f"best_{metric}", best_value)

    # Train final validator with best params
    final_validator = SklearnSignalValidator(
        model_type=best_params.pop("model_type", "lightgbm"),
        model_params=best_params,
    )
    final_validator.fit(X_train, y_train)

    return best_params, final_validator


def tune_detector(
    study: optuna.Study,
    raw_data: RawData,
    labels: pl.DataFrame,
    tuning_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Signals]:
    """Tune detector hyperparameters using Optuna.

    Args:
        study: Optuna study
        raw_data: Raw market data
        labels: Ground truth labels for evaluation
        tuning_config: Configuration including:
            - n_trials: Number of optimization trials
            - detector_name: Name of detector in registry
            - model_size: Search space size
            - metric: Metric to optimize

    Returns:
        Tuple of (best_params, signals_with_best_params)
    """
    from signalflow.core import default_registry, SfComponentType

    n_trials = tuning_config.get("n_trials", 30)
    timeout = tuning_config.get("timeout", None)
    detector_name = tuning_config.get("detector_name")
    model_size = tuning_config.get("model_size", "medium")
    metric = tuning_config.get("metric", "precision")

    # Get detector class from registry
    detector_cls = default_registry.get(SfComponentType.DETECTOR, detector_name)

    view = raw_data.view()

    def objective(trial: optuna.Trial) -> float:
        # Get hyperparameters from detector's tune() method
        if hasattr(detector_cls, "tune"):
            params = detector_cls.tune(trial, model_size=model_size)
        else:
            # Fallback to basic parameter suggestions
            params = (
                detector_cls.default_params()
                if hasattr(detector_cls, "default_params")
                else {}
            )

        # Create detector and generate signals
        detector = detector_cls(**params)
        signals = detector.run(view)

        # Evaluate signals against labels
        score = _evaluate_signals(signals, labels, metric)

        # Log to MLflow
        mlflow.log_metric(f"detector_trial_{trial.number}_score", score)

        return score

    # Run optimization
    logger.info(f"Starting detector tuning: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Get best parameters and generate final signals
    best_params = study.best_params
    best_detector = detector_cls(**best_params)
    best_signals = best_detector.run(view)

    # Log best results
    mlflow.log_params({f"detector.best.{k}": v for k, v in best_params.items()})
    mlflow.log_metric("detector.best_score", study.best_value)

    return best_params, best_signals


def _evaluate_signals(
    signals: Signals,
    labels: pl.DataFrame,
    metric: str,
) -> float:
    """Evaluate signal quality against labels."""
    signals_df = signals.value

    # Join signals with labels
    merged = signals_df.join(
        labels.select(["pair", "timestamp", "label"]),
        on=["pair", "timestamp"],
        how="inner",
    )

    # Filter to active signals only
    active = merged.filter(pl.col("signal_type") != "none")

    if active.height == 0:
        return 0.0

    # Calculate metric
    if metric == "precision":
        # Precision: correct signals / total signals
        correct = active.filter(
            (pl.col("signal_type") == "rise") & (pl.col("label") == "rise")
            | (pl.col("signal_type") == "fall") & (pl.col("label") == "fall")
        ).height
        return correct / active.height if active.height > 0 else 0.0

    elif metric == "recall":
        # Recall: captured labels / total positive labels
        total_positive = labels.filter(pl.col("label") != "none").height
        correct = active.filter(
            (pl.col("signal_type") == "rise") & (pl.col("label") == "rise")
            | (pl.col("signal_type") == "fall") & (pl.col("label") == "fall")
        ).height
        return correct / total_positive if total_positive > 0 else 0.0

    elif metric == "f1":
        precision = _evaluate_signals(signals, labels, "precision")
        recall = _evaluate_signals(signals, labels, "recall")
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    else:
        return 0.0


def tune_strategy(
    study: optuna.Study,
    raw_data: RawData,
    signals: Signals,
    tuning_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tune strategy (entry/exit) hyperparameters using Optuna.

    Args:
        study: Optuna study
        raw_data: Raw market data
        signals: Trading signals
        tuning_config: Configuration including:
            - n_trials: Number of optimization trials
            - metric: Metric to optimize ('sharpe', 'total_return', 'max_drawdown')
            - initial_capital: Starting capital

    Returns:
        Tuple of (best_params, backtest_results)
    """
    from signalflow.strategy.runner import BacktestRunner
    from signalflow.strategy.component.entry.signal import SignalEntryRule
    from signalflow.strategy.component.exit.tp_sl import TakeProfitStopLossExit
    from signalflow.strategy.broker import BacktestBroker
    from signalflow.strategy.broker.executor import VirtualSpotExecutor

    n_trials = tuning_config.get("n_trials", 50)
    timeout = tuning_config.get("timeout", None)
    metric = tuning_config.get("metric", "sharpe")
    initial_capital = tuning_config.get("initial_capital", 10_000.0)

    def objective(trial: optuna.Trial) -> float:
        # Suggest strategy parameters
        take_profit = trial.suggest_float("take_profit_pct", 0.005, 0.10)
        stop_loss = trial.suggest_float("stop_loss_pct", 0.005, 0.05)
        position_size = trial.suggest_float("position_size_pct", 0.05, 0.30)
        max_positions = trial.suggest_int("max_positions", 1, 10)
        fee_rate = trial.suggest_float("fee_rate", 0.0001, 0.002)

        # Create strategy components
        entry_rule = SignalEntryRule(
            base_position_size=initial_capital * position_size,
            max_positions_per_pair=1,
            max_total_positions=max_positions,
        )

        exit_rule = TakeProfitStopLossExit(
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
        )

        broker = BacktestBroker(executor=VirtualSpotExecutor(fee_rate=fee_rate))

        # Run backtest
        runner = BacktestRunner(
            strategy_id=f"optuna_trial_{trial.number}",
            broker=broker,
            entry_rules=[entry_rule],
            exit_rules=[exit_rule],
            initial_capital=initial_capital,
        )

        try:
            state = runner.run(raw_data=raw_data, signals=signals)
            trades_df = runner.trades_df

            if trades_df.height == 0:
                return (
                    float("-inf")
                    if study.direction == optuna.study.StudyDirection.MAXIMIZE
                    else float("inf")
                )

            # Calculate metric
            if metric == "sharpe":
                returns = trades_df.select("pnl").to_series() / initial_capital
                mean_return = returns.mean()
                std_return = returns.std()
                score = mean_return / std_return if std_return > 0 else 0.0
            elif metric == "total_return":
                score = (state.capital - initial_capital) / initial_capital
            elif metric == "max_drawdown":
                # Minimize max drawdown (return negative for maximization)
                equity_curve = trades_df.select("cumulative_pnl").to_series()
                peak = equity_curve.cum_max()
                drawdown = (equity_curve - peak) / peak
                score = -drawdown.min()  # Negative because we want to minimize
            elif metric == "win_rate":
                wins = trades_df.filter(pl.col("pnl") > 0).height
                score = wins / trades_df.height
            else:
                score = (state.capital - initial_capital) / initial_capital

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return (
                float("-inf")
                if study.direction == optuna.study.StudyDirection.MAXIMIZE
                else float("inf")
            )

        # Log to MLflow
        mlflow.log_metric(f"strategy_trial_{trial.number}_{metric}", score)

        return score

    # Run optimization
    logger.info(f"Starting strategy tuning: {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Get best parameters
    best_params = study.best_params

    # Log best results
    mlflow.log_params({f"strategy.best.{k}": v for k, v in best_params.items()})
    mlflow.log_metric(f"strategy.best_{metric}", study.best_value)

    # Run final backtest with best params
    final_entry = SignalEntryRule(
        base_position_size=initial_capital * best_params["position_size_pct"],
        max_positions_per_pair=1,
        max_total_positions=best_params["max_positions"],
    )

    final_exit = TakeProfitStopLossExit(
        take_profit_pct=best_params["take_profit_pct"],
        stop_loss_pct=best_params["stop_loss_pct"],
    )

    final_broker = BacktestBroker(
        executor=VirtualSpotExecutor(fee_rate=best_params["fee_rate"])
    )

    final_runner = BacktestRunner(
        strategy_id="optuna_best",
        broker=final_broker,
        entry_rules=[final_entry],
        exit_rules=[final_exit],
        initial_capital=initial_capital,
    )

    final_state = final_runner.run(raw_data=raw_data, signals=signals)

    backtest_results = {
        "trades": final_runner.trades_df,
        "final_capital": final_state.capital,
        "n_trades": len(final_runner.trades),
    }

    return best_params, backtest_results


def save_optuna_study(
    study: optuna.Study,
    output_path: str,
) -> str:
    """Save Optuna study results and visualizations.

    Args:
        study: Completed Optuna study
        output_path: Directory to save results

    Returns:
        Path to saved study
    """
    import json
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save study summary
    summary = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }

    with open(output_dir / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
        )

    with open(output_dir / "trials.json", "w") as f:
        json.dump(trials_data, f, indent=2, default=str)

    # Generate visualizations
    try:
        fig = plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        mlflow.log_artifact(str(output_dir / "optimization_history.html"))
    except Exception as e:
        logger.warning(f"Could not generate optimization history plot: {e}")

    try:
        fig = plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        mlflow.log_artifact(str(output_dir / "param_importances.html"))
    except Exception as e:
        logger.warning(f"Could not generate param importances plot: {e}")

    try:
        fig = plot_slice(study)
        fig.write_html(str(output_dir / "slice_plot.html"))
        mlflow.log_artifact(str(output_dir / "slice_plot.html"))
    except Exception as e:
        logger.warning(f"Could not generate slice plot: {e}")

    # Log artifacts to MLflow
    mlflow.log_artifact(str(output_dir / "study_summary.json"))
    mlflow.log_artifact(str(output_dir / "trials.json"))

    logger.info(f"Saved Optuna study results to {output_dir}")

    return str(output_dir)


def apply_best_params(
    best_params: Dict[str, Any],
    component_type: str,
    component_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply best parameters to component configuration.

    Args:
        best_params: Best parameters from tuning
        component_type: Type of component ('validator', 'detector', 'strategy')
        component_config: Original component configuration

    Returns:
        Updated configuration with best parameters
    """
    updated_config = component_config.copy()

    if component_type == "validator":
        updated_config["model_params"] = best_params
    elif component_type == "detector":
        updated_config.update(best_params)
    elif component_type == "strategy":
        updated_config["entry"] = {
            "position_size_pct": best_params.get("position_size_pct", 0.1),
            "max_positions": best_params.get("max_positions", 5),
        }
        updated_config["exit"] = {
            "take_profit_pct": best_params.get("take_profit_pct", 0.02),
            "stop_loss_pct": best_params.get("stop_loss_pct", 0.01),
        }
        updated_config["fee_rate"] = best_params.get("fee_rate", 0.001)

    # Log updated config
    mlflow.log_params(
        {f"applied.{component_type}.{k}": v for k, v in best_params.items()}
    )

    return updated_config
