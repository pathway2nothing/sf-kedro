"""Analyze pipeline nodes - signal and feature analysis.

Usage:
    kedro run --pipeline=analyze --params='flow_id=grid_sma'
    kedro run --pipeline=analyze --params='flow_id:grid_sma,level=signals'
"""

from __future__ import annotations

import json
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


def load_analysis_data(
    flow_id: str,
    level: str = "all",
) -> tuple[dict[str, Any], sf.RawData, sf.Signals | None]:
    """Load data for analysis.

    Args:
        flow_id: Flow identifier
        level: Analysis level (features/signals/all)

    Returns:
        Tuple of (config, raw_data, signals)
    """
    config = load_flow_config(flow_id)
    config["analysis_level"] = level

    logger.info(f"Loading data for analysis: {flow_id}, level={level}")

    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    period_config = data_config.get("period", {})

    start_date = _parse_date(period_config.get("start", {}))
    end_date = _parse_date(period_config.get("end", {}))
    pairs = data_config.get("pairs", ["BTCUSDT"])
    db_path = store_config.get("db_path", "data/01_raw/market.duckdb")

    raw_data = sf.load(source=db_path, pairs=pairs, start=start_date, end=end_date)
    logger.success(f"Loaded {raw_data.data['spot'].height} rows")

    signals = None
    if level in ["signals", "all"]:
        from signalflow.feature import FeaturePipeline

        detector_config = config.get("detector", {}).copy()
        detector_type = detector_config.pop("type", "example/sma_cross")
        detector = sf.get_component(type=sf.SfComponentType.DETECTOR, name=detector_type)(**detector_config)

        # Extract spot data and compute features
        data_key = config.get("strategy", {}).get("data_key", "spot")
        df = raw_data.data[data_key]

        if hasattr(detector, "features") and detector.features:
            pipeline = FeaturePipeline(features=detector.features)
            df = pipeline.compute(df)

        signals = detector.detect(df)
        logger.info(f"Detected {signals.value.height} signals for analysis")

    return config, raw_data, signals


def analyze_features(
    config: dict[str, Any],
    raw_data: sf.RawData,
) -> dict[str, Any]:
    """Analyze feature informativeness.

    Args:
        config: Flow configuration
        raw_data: Market data

    Returns:
        Feature analysis results
    """
    level = config.get("analysis_level", "all")
    if level not in ["features", "all"]:
        return {}

    logger.info("Analyzing features...")

    spot_data = raw_data.data.get("spot")
    if spot_data is None:
        return {"error": "No spot data available"}

    # Basic statistics
    numeric_cols = [c for c in spot_data.columns if spot_data[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    stats = {}
    for col in numeric_cols[:20]:  # Limit to first 20 columns
        col_data = spot_data[col].drop_nulls()
        if col_data.len() > 0:
            stats[col] = {
                "mean": float(col_data.mean()) if col_data.mean() is not None else None,
                "std": float(col_data.std()) if col_data.std() is not None else None,
                "min": float(col_data.min()) if col_data.min() is not None else None,
                "max": float(col_data.max()) if col_data.max() is not None else None,
                "null_pct": float((spot_data[col].null_count() / spot_data.height) * 100),
            }

    results = {
        "n_rows": spot_data.height,
        "n_pairs": len(config.get("data", {}).get("pairs", [])),
        "n_columns": len(spot_data.columns),
        "columns": spot_data.columns,
        "numeric_columns": numeric_cols,
        "feature_stats": stats,
    }

    logger.success(f"Analyzed {len(numeric_cols)} numeric features")

    return results


def analyze_signals(
    config: dict[str, Any],
    raw_data: sf.RawData,
    signals: sf.Signals | None,
) -> dict[str, Any]:
    """Analyze signal quality and profiling.

    Args:
        config: Flow configuration
        raw_data: Market data
        signals: Detected signals

    Returns:
        Signal analysis results
    """
    level = config.get("analysis_level", "all")
    if level not in ["signals", "all"] or signals is None:
        return {}

    logger.info("Analyzing signals...")

    signals_df = signals.value
    spot_data = raw_data.data.get("spot")

    # Basic signal statistics
    n_signals = signals_df.height
    if "signal_type" in signals_df.columns:
        signal_types = signals_df.select("signal_type").unique().to_series().to_list()
    else:
        signal_types = []

    # Count by signal type
    type_counts = {}
    if "signal_type" in signals_df.columns:
        counts = signals_df.group_by("signal_type").len()
        for row in counts.iter_rows():
            type_counts[str(row[0])] = row[1]

    # Count by pair
    pair_counts = {}
    if "pair" in signals_df.columns:
        counts = signals_df.group_by("pair").len()
        for row in counts.iter_rows():
            pair_counts[row[0]] = row[1]

    # Signal profiling - analyze price movement after signals
    profiling = {}
    if spot_data is not None and "timestamp" in signals_df.columns and "pair" in signals_df.columns:
        profiling = _profile_signals(signals_df, spot_data, config)

    results = {
        "n_signals": n_signals,
        "signal_types": signal_types,
        "type_counts": type_counts,
        "pair_counts": pair_counts,
        "profiling": profiling,
        "signals_per_day": _calc_signals_per_day(signals_df),
    }

    logger.success(f"Analyzed {n_signals} signals across {len(pair_counts)} pairs")

    return results


def _profile_signals(signals_df: pl.DataFrame, spot_data: pl.DataFrame, config: dict) -> dict:
    """Profile signals by analyzing price movement after each signal."""
    horizons = [1, 5, 10, 30, 60, 120]
    profiling = {"horizons": horizons, "returns": {}}

    try:
        joined = signals_df.join(
            spot_data.select(["pair", "timestamp", "close"]),
            on=["pair", "timestamp"],
            how="left",
        )

        if joined.height == 0 or "close" not in joined.columns:
            return profiling

        for horizon in horizons:
            profiling["returns"][horizon] = {
                "analyzed_signals": joined.height,
                "horizon_bars": horizon,
            }

    except Exception as e:
        logger.warning(f"Signal profiling failed: {e}")

    return profiling


def _calc_signals_per_day(signals_df: pl.DataFrame) -> float:
    """Calculate average signals per day."""
    if "timestamp" not in signals_df.columns or signals_df.height == 0:
        return 0.0

    try:
        ts_col = signals_df["timestamp"]
        if ts_col.dtype == pl.Datetime:
            min_ts = ts_col.min()
            max_ts = ts_col.max()
            if min_ts is not None and max_ts is not None:
                days = (max_ts - min_ts).days
                if days > 0:
                    return signals_df.height / days
    except Exception:
        pass

    return 0.0


def save_analysis_report(
    config: dict[str, Any],
    feature_results: dict[str, Any],
    signal_results: dict[str, Any],
) -> None:
    """Save analysis report.

    Args:
        config: Flow configuration
        feature_results: Feature analysis results
        signal_results: Signal analysis results
    """
    flow_id = config["flow_id"]
    output_dir = Path(f"data/08_reporting/{flow_id}/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature analysis
    if feature_results:
        feature_path = output_dir / "feature_analysis.json"
        serializable_features = {k: v for k, v in feature_results.items() if k not in ["columns", "numeric_columns"]}
        serializable_features["n_columns"] = feature_results.get("n_columns", 0)

        with open(feature_path, "w") as f:
            json.dump(serializable_features, f, indent=2, default=str)
        logger.info(f"Feature analysis saved to: {feature_path}")

    # Save signal analysis
    if signal_results:
        signal_path = output_dir / "signal_analysis.json"
        with open(signal_path, "w") as f:
            json.dump(signal_results, f, indent=2, default=str)
        logger.info(f"Signal analysis saved to: {signal_path}")

    # Log detailed results
    logger.info("=" * 50)
    logger.success(f"Analysis Complete: {config.get('flow_name', config['flow_id'])}")
    logger.info("-" * 50)

    if feature_results:
        logger.info(f"  Data rows:       {feature_results.get('n_rows', 0):,}")
        logger.info(f"  Columns:         {feature_results.get('n_columns', 0)}")
        logger.info(f"  Numeric cols:    {len(feature_results.get('numeric_columns', []))}")

    if signal_results:
        n_signals = signal_results.get("n_signals", 0)
        signals_per_day = signal_results.get("signals_per_day", 0)
        type_counts = signal_results.get("type_counts", {})
        pair_counts = signal_results.get("pair_counts", {})

        logger.info(f"  Total signals:   {n_signals:,}")
        logger.info(f"  Signals/day:     {signals_per_day:.1f}")

        if type_counts:
            logger.info("  By type:")
            for sig_type, count in type_counts.items():
                logger.info(f"    - {sig_type}: {count}")

        if pair_counts:
            logger.info(f"  Pairs analyzed:  {len(pair_counts)}")

    logger.info(f"  Report saved:    {output_dir}")
    logger.info("=" * 50)

    # Send to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config.get("enabled", False):
        _send_telegram_notification(telegram_config, config, feature_results, signal_results)


def _send_telegram_notification(
    telegram_config: dict,
    config: dict,
    feature_results: dict,
    signal_results: dict,
) -> None:
    """Send Telegram notification with analysis results."""
    try:
        from sf_kedro.utils.telegram import send_message_to_telegram

        message = f"""
📊 <b>SignalFlow Analysis Complete</b>

🔍 Flow: {config.get("flow_name", config["flow_id"])}
"""
        if feature_results:
            message += f"📈 Features: {feature_results.get('n_columns', 0)} columns\n"
        if signal_results:
            message += f"🎯 Signals: {signal_results.get('n_signals', 0)} total\n"
            message += f"📅 Rate: {signal_results.get('signals_per_day', 0):.1f} signals/day\n"

        send_message_to_telegram(
            message=message,
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )
        logger.info("Sent Telegram notification")
    except Exception as e:
        logger.warning(f"Failed to send Telegram notification: {e}")
