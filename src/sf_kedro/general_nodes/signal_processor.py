"""Signal detection and processing."""

from typing import Dict
import polars as pl
import mlflow

import signalflow as sf


def detect_signals(
    raw_data: sf.core.RawData,
    detector_config: Dict,
) -> sf.core.Signals:
    """
    Detect trading signals.
    
    Args:
        raw_data: Raw market data
        detector_config: Detector configuration
        
    Returns:
        Detected signals
    """
    detector_name = detector_config.get('name')
    
    detector_type = sf.core.default_registry.get(component_type=sf.core.SfComponentType.DETECTOR, name=detector_name)
    detector: sf.detector.SignalDetector = detector_type(**detector_config.get('params', {}))
    raw_data_view = sf.core.RawDataView(raw_data)
    signals = detector.run(raw_data_view=raw_data_view, context=None)
    
    mlflow.log_params({
        "detector.type": detector_name,
        "detector.params": detector_config.get('params', {}),
    })
    
    return signals


def calculate_signal_metrics(
    signals: sf.core.Signals,
    raw_data: sf.core.RawData,
) -> Dict:
    """
    Calculate signal quality metrics.
    
    Args:
        signals: Detected signals
        raw_data: Raw market data
        
    Returns:
        Dictionary of metrics
    """
    signal_df = signals.value
    
    # Signal type distribution
    signal_counts = (
        signal_df
        .group_by("signal_type")
        .agg(pl.count().alias("count"))
    )
    
    signal_type_dict = {
        row['signal_type']: row['count']
        for row in signal_counts.iter_rows(named=True)
    }
    
    # Per-pair distribution
    pair_counts = (
        signal_df
        .group_by("pair")
        .agg(pl.count().alias("count"))
    )
    
    metrics = {
        "total_signals": signal_df.height,
        "rise_signals": signal_type_dict.get("rise", 0),
        "fall_signals": signal_type_dict.get("fall", 0),
        "none_signals": signal_type_dict.get("none", 0),
        "unique_pairs": signal_df.select("pair").n_unique(),
    }
    
    # Log to MLflow
    mlflow.log_metrics({
        f"signals.{k}": v for k, v in metrics.items()
    })
    
    return metrics


def validate_signals(
    raw_signals: sf.core.Signals,
    features: pl.DataFrame,
    validator,
) -> sf.core.Signals:
    """
    Apply validator to filter/score signals.
    
    Args:
        raw_signals: Raw detected signals
        features: Feature DataFrame
        validator: Trained validator (sklearn or nn)
        
    Returns:
        Validated signals with probabilities
    """
    validated_signals = validator.validate_signals(raw_signals, features)
    
    # Calculate filtering statistics
    val_df = validated_signals.value
    
    # Count high-confidence signals (assuming prob > 0.6)
    if 'probability_rise' in val_df.columns:
        high_conf = val_df.filter(
            (pl.col("probability_rise") > 0.6) | 
            (pl.col("probability_fall") > 0.6)
        ).height
        
        filter_rate = 1 - (high_conf / val_df.height) if val_df.height > 0 else 0
        
        mlflow.log_metrics({
            "validation.high_confidence_signals": high_conf,
            "validation.filter_rate": filter_rate,
        })
    
    return validated_signals