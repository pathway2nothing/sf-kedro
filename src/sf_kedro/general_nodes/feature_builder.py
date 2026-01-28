"""Feature extraction utilities."""

from typing import Dict, List
import polars as pl
import mlflow

import signalflow as sf


def create_feature_set(feature_configs: List[Dict]) -> sf.feature.FeatureSet:
    """
    Create FeatureSet from configurations.
    
    Args:
        feature_configs: List of feature extractor configs
        
    Returns:
        FeatureSet instance
    """
    extractors = []
    
    for config in feature_configs.get('extractors', []):
        extractor_name = config.pop('type')
        if extractor_name and 'custom' in extractor_name:
            continue
        
        extractor_type = sf.default_registry.get(
            component_type=sf.SfComponentType.FEATURE_EXTRACTOR,
            name=extractor_name)
        
        extractor = extractor_type(**config)
        extractors.append(extractor)

    return sf.feature.FeatureSet(extractors=extractors)


def extract_validation_features(
    raw_data: sf.RawData,
    raw_signals: pl.DataFrame,
    feature_configs: List[Dict],
) -> pl.DataFrame:
    """
    Extract features for validation model.
    
    Args:
        raw_data: Raw market data
        feature_configs: List of feature extractor configurations
        
    Returns:
        DataFrame with features
    """

    feature_set = create_feature_set(feature_configs)
    
    raw_data_view = sf.RawDataView(raw_data)
    features_df = feature_set.extract(raw_data_view)
    
    feature_cols = [col for col in features_df.columns 
                    if col not in ['timestamp', 'pair']]
    
    mlflow.log_params({
        "features.num_features": len(feature_cols),
        "features.names": ",".join(feature_cols[:10]), 
    })
    
    mlflow.log_metrics({
        "features.total_rows": features_df.height,
        "features.null_ratio": features_df.null_count().sum() / (features_df.height * len(feature_cols)) if len(feature_cols) > 0 else 0,
    })
    
    return features_df