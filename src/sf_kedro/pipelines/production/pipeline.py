"""Production pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.modules import (
    load_raw_data_from_storage,
    detect_signals,
    extract_validation_features,
    load_validator_from_registry,
    validate_signals,
    run_backtest,
    calculate_backtest_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create production simulation pipeline with namespace."""
    
    base_pipeline = pipeline([
        # Data loading
        node(
            func=load_raw_data_from_storage,
            inputs=[
                "params:data.storage_path",
                "params:data.pairs",
                "params:data.date_range",
            ],
            outputs="raw_data",
            name="load_raw_data",
            tags=["data_loading"],
        ),
        
        # Signal detection
        node(
            func=detect_signals,
            inputs=["raw_data", "params:detector"],
            outputs="raw_signals",
            name="detect_signals",
            tags=["signal_detection"],
        ),
        
        # Feature extraction
        node(
            func=extract_validation_features,
            inputs=["raw_data", "params:features.extractors"],
            outputs="validation_features",
            name="extract_validation_features",
            tags=["feature_engineering"],
        ),
        
        # Load pre-trained validator
        node(
            func=load_validator_from_registry,
            inputs=[
                "params:model.name",
                "params:model.stage",
            ],
            outputs="production_validator",
            name="load_production_validator",
            tags=["model_loading"],
        ),
        
        # Validate signals
        node(
            func=validate_signals,
            inputs=["raw_signals", "validation_features", "production_validator"],
            outputs="validated_signals",
            name="validate_signals",
            tags=["signal_validation"],
        ),
        
        # Backtest
        node(
            func=run_backtest,
            inputs=["raw_data", "validated_signals", "params:strategy"],
            outputs="backtest_results",
            name="run_backtest",
            tags=["backtesting"],
        ),
        node(
            func=calculate_backtest_metrics,
            inputs="backtest_results",
            outputs="backtest_metrics",
            name="calculate_backtest_metrics",
            tags=["metrics", "backtest_metrics"],
        ),
    ])
    
    # Return with namespace
    return pipeline(
        base_pipeline,
        namespace="production",
        parameters={
            "params:data",
            "params:detector",
            "params:features",
            "params:model",
            "params:strategy"
        }
    )