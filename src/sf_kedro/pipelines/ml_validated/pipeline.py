from kedro.pipeline import Pipeline, node, pipeline
from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    extract_validation_features,
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
    base_pipeline = pipeline(
        [
            # 1. Download market data
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
            # 2. Load raw data from storage
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:ml_validated.data.store",
                    "params:ml_validated.data.period",
                    "params:ml_validated.data.pairs",
                    "ml_store_path",
                ],
                outputs="ml_raw_data",
                name="load_raw_data_node",
            ),
            # 3. Detect signals
            node(
                func=detect_signals,
                inputs=["ml_raw_data", "params:ml_validated.detector"],
                outputs="ml_raw_signals",
                name="detect_signals_node",
            ),
            # 4. Extract validation features
            node(
                func=extract_validation_features,
                inputs=[
                    "ml_raw_data",
                    "ml_raw_signals",
                    "params:ml_validated.features",
                ],
                outputs="ml_features",
                name="extract_validation_features_node",
                tags=["features", "preparation"],
            ),
            # 5. Create labels
            node(
                func=create_labels,
                inputs=[
                    "ml_raw_data",
                    "ml_raw_signals",
                    "params:ml_validated.labeling",
                ],
                outputs="ml_labels",
                name="generate_labels_node",
            ),
            # 6. Split train/val/test
            node(
                func=split_train_val_test,
                inputs=[
                    "ml_labels",
                    "ml_features",
                    "params:ml_validated.split",
                ],
                outputs=["ml_train_data", "ml_val_data", "ml_test_data"],
                name="split_train_val_test_node",
                tags=["data", "splitting"],
            ),
            # 7. Train ML validator
            node(
                func=create_sklearn_validator,
                inputs=[
                    "ml_train_data",
                    "ml_val_data",
                    "params:ml_validated.validator",
                ],
                outputs="ml_trained_validator",
                name="train_ml_validator",
                tags=["ml", "training", "validator"],
            ),
            # 8. Validate signals using trained validator
            node(
                func=validate_signals,
                inputs=[
                    "ml_raw_signals",
                    "ml_features",
                    "ml_trained_validator",
                ],
                outputs="ml_validated_signals",
                name="validate_signals_node",
                tags=["signals", "validation"],
            ),
            # 9. Compute signal metrics (returns both results and plots like baseline)
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
                outputs=["ml_signal_metrics_results", "ml_signal_plots"],
                name="compute_signal_metrics",
            ),
            # 10. Save signal plots
            node(
                func=save_signal_plots,
                inputs={
                    "plots": "ml_signal_plots",
                    "output_dir": "params:ml_validated.signal_plots_output_dir",
                },
                outputs=None,
                name="save_signal_plots",
            ),
            # 11. Run backtest (returns both results and state like baseline)
            node(
                func=run_backtest,
                inputs=[
                    "ml_raw_data",
                    "ml_validated_signals",
                    "params:ml_validated.strategy",
                ],
                outputs=["ml_backtest_results", "ml_backtest_state"],
                name="run_backtest",
                tags=["backtesting"],
            ),
            # 12. Log backtest metrics
            node(
                func=log_last_state_metrics,
                inputs="ml_backtest_results",
                outputs="ml_backtest_metrics",
                name="log_backtest_metrics",
                tags=["metrics", "backtest_metrics"],
            ),
            # 13. Compute strategy metrics
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
                outputs=["ml_strategy_metrics_results", "ml_strategy_plots"],
                name="compute_strategy_metrics",
                tags=["metrics", "strategy_metrics"],
            ),
            # 14. Save strategy plots
            node(
                func=save_strategy_plots,
                inputs={
                    "plots": "ml_strategy_plots",
                    "output_dir": "params:ml_validated.strategy_plots_output_dir",
                },
                outputs=None,
                name="save_strategy_plots",
                tags=["reporting"],
            ),
        ]
    )

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
            "params:ml_validated.signal_plots_output_dir",
            "params:ml_validated.strategy",
            "params:ml_validated.strategy_metrics",
            "params:ml_validated.strategy_plots_output_dir",
            "params:telegram",
            "params:strategy_name",
        },
    )
