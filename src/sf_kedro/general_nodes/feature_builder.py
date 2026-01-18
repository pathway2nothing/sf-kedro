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
    
    for config in feature_configs:
        # Get extractor name and remove from config
        extractor_name = config.get('name')
        
        # For custom extractors, you need to import them
        # For now, skip if not recognized
        if extractor_name and 'custom' in extractor_name:
            # Custom extractors need to be registered separately
            # Skip for now or implement custom logic
            continue
        
        # Create extractor instance directly
        # This is a placeholder - adjust based on your actual extractors
        pass
    
    return sf.feature.FeatureSet(extractors=extractors)


def extract_validation_features(
    raw_data: sf.core.RawData,
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

    from dataclasses import dataclass
    
    @dataclass
    @sf.core.sf_component(name='custom/log_return', override=True)
    class LogReturnExtractor(sf.feature.FeatureExtractor):
        """Calculates logarithmic returns: ln(Pt / Pt-1)."""
        
        price_col: str = 'close'
        out_col: str = 'log_ret'
        period: int = 1

        def compute_group(self, group_df: pl.DataFrame, data_context: dict | None) -> pl.DataFrame:
            if self.price_col not in group_df.columns:
                raise ValueError(f"Missing required column: {self.price_col}")

            log_ret = (
                pl.col(self.price_col)
                .log()
                .diff(n=self.period)
                .alias(self.out_col)
            )
            
            return group_df.with_columns(log_ret)
    
    # Create extractors from configs
    extractors = []
    for config in feature_configs:
        extractor = LogReturnExtractor(
            price_col=config.get('price_col', 'close'),
            out_col=config.get('out_col', 'log_ret'),
            period=config.get('period', 1)
        )
        extractors.append(extractor)
    
    feature_set = sf.feature.FeatureSet(extractors=extractors)
    
    raw_data_view = sf.core.RawDataView(raw_data)
    features_df = feature_set.extract(raw_data_view)
    
    # Log feature statistics
    feature_cols = [col for col in features_df.columns 
                    if col not in ['timestamp', 'pair']]
    
    mlflow.log_params({
        "features.num_features": len(feature_cols),
        "features.names": ",".join(feature_cols[:10]),  # First 10
    })
    
    mlflow.log_metrics({
        "features.total_rows": features_df.height,
        "features.null_ratio": features_df.null_count().sum() / (features_df.height * len(feature_cols)) if len(feature_cols) > 0 else 0,
    })
    
    return features_df