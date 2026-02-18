"""Train pipeline nodes - validator training.

Usage:
    kedro run --pipeline=train --params='flow_id=grid_sma'
    kedro run --pipeline=train --params='flow_id:grid_sma,validator_type=xgboost'
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import signalflow as sf
from loguru import logger

from sf_kedro.utils.flow_config import load_flow_config


def _parse_date(date_config: dict) -> datetime:
    """Parse date from config dict."""
    return datetime(
        year=date_config.get("year", 2025),
        month=date_config.get("month", 1),
        day=date_config.get("day", 1),
    )


def load_training_data(
    flow_id: str,
    validator_type: str = "validator/lightgbm",
) -> tuple[dict[str, Any], sf.RawData, sf.Signals, pl.DataFrame]:
    """Load data for validator training.

    Args:
        flow_id: Flow identifier
        validator_type: Validator type (validator/lightgbm, validator/xgboost, etc.)

    Returns:
        Tuple of (config, raw_data, signals, labeled_data)
    """
    config = load_flow_config(flow_id)

    # Normalize validator type
    if not validator_type.startswith("validator/"):
        validator_type = f"validator/{validator_type}"
    config["validator_type"] = validator_type

    logger.info(f"Loading training data: {flow_id}, validator={validator_type}")

    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    period_config = data_config.get("period", {})

    start_date = _parse_date(period_config.get("start", {}))
    end_date = _parse_date(period_config.get("end", {}))
    pairs = data_config.get("pairs", ["BTCUSDT"])
    db_path = store_config.get("db_path", "data/01_raw/market.duckdb")

    raw_data = sf.load(source=db_path, pairs=pairs, start=start_date, end=end_date)
    logger.info(f"Loaded {raw_data.data['spot'].height} rows")

    # Run detection with features
    from sf_kedro.utils.detection import run_detection

    signals = run_detection(config, raw_data)
    logger.info(f"Detected {signals.value.height} signals")

    # Generate labels using labeler
    labeling_config = config.get("labeling", {}).copy()
    labeling_type = labeling_config.pop("type", "fixed_horizon")
    labeler = sf.get_component(type=sf.SfComponentType.LABELER, name=labeling_type)(**labeling_config)

    # Apply labeling to spot data
    spot_data = raw_data.data.get("spot")
    if spot_data is None:
        raise ValueError("No spot data available for labeling")

    labeled_data = labeler.compute(spot_data, signals=signals)

    logger.success(f"Generated labels for {labeled_data.height} rows")

    return config, raw_data, signals, labeled_data


def prepare_features(
    config: dict[str, Any],
    raw_data: sf.RawData,
    signals: sf.Signals,
    labeled_data: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare features for validator training.

    Splits data into train/validation sets and extracts features.

    Args:
        config: Flow configuration
        raw_data: Market data
        signals: Detected signals
        labeled_data: Data with labels

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    logger.info("Preparing features for training...")

    spot_data = raw_data.data.get("spot")
    if spot_data is None:
        raise ValueError("No spot data available")

    # Get label column name
    labeling_config = config.get("labeling", {})
    label_col = labeling_config.get("out_col", "label")

    # Filter to signals only (join signals with labeled data)
    signals_df = signals.value
    feature_cols = ["pair", "timestamp"]

    # Add OHLCV as basic features
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    available_ohlcv = [c for c in ohlcv_cols if c in spot_data.columns]
    feature_cols.extend(available_ohlcv)

    # Join signals with spot data and labels
    joined = signals_df.select(["pair", "timestamp", "signal_type"]).join(
        spot_data.select(["pair", "timestamp", *available_ohlcv]),
        on=["pair", "timestamp"],
        how="left",
    )

    # Add labels
    if label_col in labeled_data.columns:
        joined = joined.join(
            labeled_data.select(["pair", "timestamp", label_col]),
            on=["pair", "timestamp"],
            how="left",
        )
    else:
        logger.warning(f"Label column '{label_col}' not found, using signal_type as label")
        joined = joined.with_columns(pl.col("signal_type").alias(label_col))

    # Drop rows with missing labels
    joined = joined.drop_nulls(subset=[label_col])

    if joined.height == 0:
        raise ValueError("No valid labeled data after joining")

    logger.info(f"Prepared {joined.height} labeled samples")

    # Split by time (80% train, 20% validation)
    split_idx = int(joined.height * 0.8)
    train_data = joined.head(split_idx)
    val_data = joined.tail(joined.height - split_idx)

    # Separate features and labels
    exclude_cols = ["pair", "timestamp", label_col, "signal_type"]
    feature_cols = [c for c in joined.columns if c not in exclude_cols]

    X_train = train_data.select(["pair", "timestamp", *feature_cols])
    y_train = train_data.select([label_col])
    X_val = val_data.select(["pair", "timestamp", *feature_cols])
    y_val = val_data.select([label_col])

    logger.success(f"Train: {X_train.height} samples, Val: {X_val.height} samples")

    return X_train, y_train, X_val, y_val


def train_validator(
    config: dict[str, Any],
    X_train: pl.DataFrame,
    y_train: pl.DataFrame,
    X_val: pl.DataFrame,
    y_val: pl.DataFrame,
) -> Any:
    """Train validator model.

    Args:
        config: Flow configuration
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained validator model
    """
    validator_type = config.get("validator_type", "validator/lightgbm")
    logger.info(f"Training {validator_type} validator...")

    # Get validator from registry
    validator_cls = sf.get_component(type=sf.SfComponentType.VALIDATOR, name=validator_type)

    # Get validator config from flow config
    validator_config = config.get("validator", {}).copy()
    validator_config.pop("type", None)
    validator_config.pop("model_path", None)

    validator = validator_cls(**validator_config)

    # Train the validator
    validator.fit(X_train, y_train, X_val, y_val)

    # Log training results
    logger.success(f"Validator {validator_type} trained successfully")
    if hasattr(validator, "feature_columns"):
        logger.info(f"Feature columns: {len(validator.feature_columns)}")

    return validator


def save_trained_model(
    config: dict[str, Any],
    validator: Any,
) -> None:
    """Save trained validator model.

    Args:
        config: Flow configuration
        validator: Trained validator
    """
    flow_id = config["flow_id"]
    validator_type = config.get("validator_type", "validator/lightgbm")
    validator_name = validator_type.split("/")[-1]

    output_dir = Path(f"data/06_models/{flow_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"validator_{validator_name}.pkl"

    # Save the model
    validator.save(model_path)

    # Log detailed results
    logger.info("=" * 50)
    logger.success(f"Validator Training: {config.get('flow_name', config['flow_id'])}")
    logger.info("-" * 50)

    logger.info(f"  Validator type:  {validator_type}")
    logger.info(f"  Model saved:     {model_path}")

    if hasattr(validator, "feature_columns"):
        logger.info(f"  Features:        {len(validator.feature_columns)}")

    if hasattr(validator, "model") and validator.model is not None:
        model = validator.model
        if hasattr(model, "n_estimators"):
            logger.info(f"  Estimators:      {model.n_estimators}")
        if hasattr(model, "best_score_"):
            logger.info(f"  Best score:      {model.best_score_:.4f}")

    logger.info("=" * 50)

    # Send to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config.get("enabled", False):
        _send_telegram_notification(telegram_config, config, validator, model_path)


def _send_telegram_notification(
    telegram_config: dict,
    config: dict,
    validator: Any,
    model_path: Path,
) -> None:
    """Send Telegram notification about training completion."""
    try:
        from sf_kedro.utils.telegram import send_message_to_telegram

        validator_type = config.get("validator_type", "unknown")

        message = f"""
🎓 <b>SignalFlow Validator Training Complete</b>

🔍 Flow: {config.get("flow_name", config["flow_id"])}
🤖 Validator: {validator_type}
💾 Model: {model_path.name}
"""
        if hasattr(validator, "feature_columns"):
            message += f"📊 Features: {len(validator.feature_columns)}\n"

        send_message_to_telegram(
            message=message,
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )
        logger.info("Sent Telegram notification")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}")
