"""Signal detection and processing."""

from typing import Dict
import polars as pl
import mlflow

import signalflow as sf
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
    detector_name = detector_config.pop('type')
    
    detector = sf.default_registry.get(component_type=sf.SfComponentType.DETECTOR, name=detector_name)(**detector_config)
    raw_data_view = sf.RawDataView(raw_data)
    signals = detector.run(raw_data_view=raw_data_view, context=None)
    
    mlflow.log_params({
        "detector.type": detector_name,
        "detector.params": detector_config,
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
    
    val_df = validated_signals.value
    
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