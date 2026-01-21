"""Signal detection and processing."""

from typing import Dict
import polars as pl
import mlflow

import signalflow as sf
from sf_kedro.custom_modules import SignalMetricsProcessor
import tempfile
from pathlib import Path
import plotly.io as pio

def detect_signals(
    raw_data: sf.RawData,
    detector_config: Dict,
) -> sf.Signals:
    """
    Detect trading signals.
    
    Args:
        raw_data: Raw market data
        detector_config: Detector configuration
        
    Returns:
        Detected signals
    """
    detector_name = detector_config.get('name')
    
    detector_type = sf.default_registry.get(component_type=sf.SfComponentType.DETECTOR, name=detector_name)
    detector: sf.detector.SignalDetector = detector_type(**detector_config.get('params', {}))
    raw_data_view = sf.RawDataView(raw_data)
    signals = detector.run(raw_data_view=raw_data_view, context=None)
    
    mlflow.log_params({
        "detector.type": detector_name,
        "detector.params": detector_config.get('params', {}),
    })
    
    return signals


def validate_signals(
    raw_signals: sf.Signals,
    features: pl.DataFrame,
    validator,
) -> sf.Signals:
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