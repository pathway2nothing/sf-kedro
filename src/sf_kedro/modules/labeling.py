"""Labeling and data splitting."""

from typing import Dict, Tuple
import polars as pl
import mlflow

import signalflow as sf


def create_labels(
    raw_data: sf.core.RawData,
    signals: sf.core.Signals,
    labeler_config: Dict,
) -> pl.DataFrame:
    """
    Create labels for validation training.
    
    Args:
        raw_data: Raw market data
        signals: Detected signals
        labeler_config: Labeler configuration
        
    Returns:
        DataFrame with labels
    """
    # Create labeler directly
    labeler_name = labeler_config.get('name')
    
    if labeler_name == 'fixed_horizon':
        from signalflow.target import FixedHorizonLabeler
        
        labeler = FixedHorizonLabeler(
            price_col=labeler_config.get('price_col', 'close'),
            horizon=labeler_config.get('horizon', 10),
            include_meta=labeler_config.get('include_meta', True)
        )
    else:
        raise ValueError(f"Unknown labeler: {labeler_name}")
    
    # Create labels
    raw_df = raw_data.get("spot")
    labeled_df = labeler.compute(df=raw_df, signals=signals)
    
    # Log label distribution
    label_dist = (
        labeled_df
        .group_by("label")
        .agg(pl.count().alias("count"))
    )
    
    mlflow.log_params({
        "labeler.type": labeler.__class__.__name__,
        "labeler.horizon": labeler_config.get('horizon', 10),
    })
    
    for row in label_dist.iter_rows(named=True):
        mlflow.log_metric(f"labels.{row['label']}", row['count'])
    
    return labeled_df


def split_train_val_test(
    labeled_data: pl.DataFrame,
    features: pl.DataFrame,
    split_config: Dict,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into train/val/test sets.
    
    Args:
        labeled_data: DataFrame with labels
        features: Feature DataFrame
        split_config: Split ratios configuration
        
    Returns:
        Tuple of (train_data, val_data, test_data) dicts
    """
    train_ratio = split_config.get('train_ratio', 0.7)
    val_ratio = split_config.get('val_ratio', 0.15)
    
    # Join features with labels
    full_data = features.join(
        labeled_data.select(["timestamp", "pair", "label"]),
        on=["timestamp", "pair"],
        how="inner"
    )
    
    # Remove NaN values
    full_data = full_data.drop_nulls()
    
    n = full_data.height
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = {
        "full": full_data.slice(0, train_end),
    }
    
    val_data = {
        "full": full_data.slice(train_end, val_end - train_end),
    }
    
    test_data = {
        "full": full_data.slice(val_end, None),
    }
    
    # Log split statistics
    mlflow.log_params({
        "split.train_ratio": train_ratio,
        "split.val_ratio": val_ratio,
        "split.test_ratio": 1 - train_ratio - val_ratio,
    })
    
    mlflow.log_metrics({
        "split.train_samples": train_data["full"].height,
        "split.val_samples": val_data["full"].height,
        "split.test_samples": test_data["full"].height,
    })
    
    return train_data, val_data, test_data