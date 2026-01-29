"""
ML Validated Pipeline for SignalFlow-Kedro

This pipeline implements signal validation using classical ML models (sklearn-based).
It follows the structure:
1. Download/Load market data
2. Detect raw signals
3. Extract validation features
4. Create labels for training
5. Split data into train/val/test
6. Train sklearn validator
7. Validate signals
8. Run backtest with validated signals
9. Compute and save metrics

The pipeline is designed to work with the existing SignalFlow components
and follows the same pattern as the baseline pipeline but adds ML validation.
"""

from kedro.pipeline import Pipeline, node, pipeline
from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    extract_validation_features,
    create_feature_set,
    create_labels,
    split_train_val_test,
    create_sklearn_validator,
    validate_signals,
    run_backtest,
    compute_signal_metrics,
    save_signal_plots,
    compute_strategy_metrics,
    save_strategy_plots,
    log_last_state_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the ML validated pipeline.
    
    This pipeline trains a classical ML model to validate signals detected
    by the detector, then backtests the validated signals.
    
    Returns:
        Pipeline: A Kedro pipeline with all nodes for ML-validated signal processing
    """
    
    base_pipeline = pipeline([
         node(
                func=download_market_data,
                inputs=[
                    "params:ml_validated.data.store",
                    "params:ml_validated.data.loader",
                    "params:ml_validated.data.period",
                    "params:ml_validated.data.pairs",
                ],
                outputs="ml_store_path",
                name="download_market_data",
                tags=["data_download"],
        ),
        node(
            func=load_raw_data_from_storage,
            inputs=[
                "params:ml_validated.data.store",
                "params:ml_validated.data.period", 
                "params:ml_validated.data.pairs",
                "ml_store_path"
            ],
            outputs="ml_raw_data",
            name="load_raw_data",
        ),
        node(
            func=detect_signals,
            inputs=["ml_raw_data", "params:ml_validated.detector"],
            outputs="ml_raw_signals",
            name="detect_signals",
        ),

        node(
            func=extract_validation_features,
            inputs=[
                'ml_raw_data',
                'ml_raw_signals',
                'params:ml_validated.features',
            ],
            outputs='ml_features',
            name='create_feature_set',
            tags=['features', 'preparation'],
        ),
        
        node(
            func=create_labels,
            inputs=[
                'ml_raw_data',
                'ml_raw_signals',
                'params:ml_validated.labeling',
            ],
            outputs='ml_labels',
            name='create_labels',
            tags=['labeling'],
        ),
        
        node(
            func=split_train_val_test,
            inputs=[
                'ml_labels',
                'ml_features',
                'params:ml_validated.split',
            ],
            outputs=['ml_train_data', 'ml_val_data', 'ml_test_data'],
            name='split_train_val_test',
            tags=['data', 'splitting'],
        ),
        
        node(
            func=create_sklearn_validator,
            inputs=[
                'ml_train_data',
                'ml_val_data',
                'params:ml_validated.validator',
            ],
            outputs='ml_trained_validator',
            name='train_ml_validator',
            tags=['ml', 'training', 'validator'],
        ),
        
        node(
            func=validate_signals,
            inputs=[
                'ml_raw_signals',
                'ml_features',
                'ml_trained_validator',
            ],
            outputs='ml_validated_signals',
            name='validate_signals',
            tags=['signals', 'validation'],
        ),
        
        node(
            func=compute_signal_metrics,
            inputs={
                 "params": "params:ml_validated.signal_metrics",
                 "raw_data": "ml_raw_data",
                 "signals": "ml_validated_signals",
                 "labels": "ml_labels",
                 "telegram_config": "params:telegram",
                 "strategy_name": "params:strategy_name",
            },
            outputs='ml_signal_metrics_results',
            name='compute_signal_metrics',
            tags=['metrics', 'signal_metrics'],
        ),
        
        node(
            func=save_signal_plots,
            inputs=[
                'ml_signal_metrics_results',
                'params:ml_validated.signal_metrics',
            ],
            outputs='ml_signal_metrics_plots',
            name='save_signal_plots',
            tags=['metrics', 'signal_metrics', 'plots'],
        ),
        
        node(
            func=run_backtest,
            inputs=[
                'ml_raw_data',
                'ml_validated_signals',
                'params:ml_validated.strategy',
            ],
            outputs='ml_backtest_state',
            name='run_backtest',
            tags=['backtest', 'strategy'],
        ),
        
        node(
                func=log_last_state_metrics,
                inputs="ml_backtest_state",
                outputs="ml_backtest_metrics",
                name="log_backtest_metrics",
                tags=["metrics", "backtest_metrics"],
            ),
        node(
            func=compute_strategy_metrics,
            inputs={
                "backtest_results": "ml_backtest_results",
                "params": "params:ml_validated.strategy_metrics",
                "telegram_config": "params:telegram",
                "strategy_name": "params:strategy_name",
                "raw_data": "ml_raw_data",
                "state": "ml_backtest_state",
            },
            outputs=["ml_strategy_metrics_results", "ml_strategy_metrics_plots"],
            name='compute_strategy_metrics',
            tags=['metrics', 'strategy_metrics'],
        ),
        node(
            func=save_strategy_plots,
            inputs={
                "plots": "ml_strategy_metrics_plots",
                "output_dir": "params:ml_validated.strategy_plots_output_dir"
            },
            outputs=None,
            name='save_strategy_plots',
            tags=['reporting'],
        ),
        
        node(
            func=log_last_state_metrics,
            inputs=[
                'ml_backtest_state',
            ],
            outputs=None,
            name='log_mlflow_metrics',
            tags=['logging', 'mlflow'],
        ),
    ])
    
    return pipeline(
        base_pipeline,
        namespace="ml_validated",
        parameters={
            "params:ml_validated.data.store",           
            "params:ml_validated.data.loader",          
            "params:ml_validated.data.pairs",         
            "params:ml_validated.data.period",          
            "params:ml_validated.detector",             
            "params:ml_validated.features",             
            "params:ml_validated.labeling",             
            "params:ml_validated.split",                
            "params:ml_validated.validator",            
            "params:ml_validated.signal_metrics",       
            #"params:ml_validated.signal_plots_output_dir",  
            "params:ml_validated.strategy",             
            "params:ml_validated.strategy_metrics",    
            #"params:ml_validated.strategy_plots_output_dir",  
            "params:telegram",            
            "params:strategy_name",       
        }
    )