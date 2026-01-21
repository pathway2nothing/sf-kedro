"""NN validated pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.modules import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    compute_signal_metrics,
    extract_validation_features,
    create_labels,
    split_train_val_test,
    create_nn_validator,
    validate_signals,
    run_backtest,
    calculate_backtest_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create NN validated pipeline with namespace."""
    
    base_pipeline = pipeline([
        # Data pipeline
        node(
            func=download_market_data,
            inputs=[
                "params:data.pairs",
                "params:data.date_range",
                "params:data.storage_path",
            ],
            outputs="download_status",
            name="download_market_data",
            tags=["data_download"],
        ),
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
        
        # Signal detection pipeline
        node(
            func=detect_signals,
            inputs=["raw_data", "params:detector"],
            outputs="raw_signals",
            name="detect_signals",
            tags=["signal_detection"],
        ),
        node(
            func=compute_signal_metrics,
            inputs=["raw_signals", "raw_data"],
            outputs="raw_signal_metrics",
            name="calculate_raw_signal_metrics",
            tags=["metrics", "signal_metrics"],
        ),
        
        # Feature engineering pipeline
        node(
            func=extract_validation_features,
            inputs=["raw_data", "params:features.extractors"],
            outputs="validation_features",
            name="extract_validation_features",
            tags=["feature_engineering"],
        ),
        
        # Labeling pipeline
        node(
            func=create_labels,
            inputs=["raw_data", "raw_signals", "params:labeler"],
            outputs="labeled_data",
            name="create_labels",
            tags=["labeling"],
        ),
        node(
            func=split_train_val_test,
            inputs=["labeled_data", "validation_features", "params:split"],
            outputs=["train_data", "val_data", "test_data"],
            name="split_train_val_test",
            tags=["data_split"],
        ),
        
        # Model training pipeline (NN)
        node(
            func=create_nn_validator,
            inputs=[
                "train_data",
                "val_data",
                "params:nn_validator.model",
                "params:nn_validator.trainer",
            ],
            outputs="trained_validator",
            name="train_neural_validator",
            tags=["model_training", "neural_network"],
        ),
        
        # Signal validation pipeline
        node(
            func=validate_signals,
            inputs=["raw_signals", "validation_features", "trained_validator"],
            outputs="validated_signals",
            name="validate_signals",
            tags=["signal_validation"],
        ),
        node(
            func=compute_signal_metrics,
            inputs=["validated_signals", "raw_data"],
            outputs="validated_signal_metrics",
            name="calculate_validated_signal_metrics",
            tags=["metrics", "signal_metrics"],
        ),
        
        # Backtesting pipeline
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
        namespace="nn_validated",
        parameters={
            "params:data",
            "params:detector",
            "params:features",
            "params:labeler",
            "params:split",
            "params:nn_validator",
            "params:strategy"
        }
    )