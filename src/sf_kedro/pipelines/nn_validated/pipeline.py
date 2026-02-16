"""NN validated pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    compute_signal_metrics,
    compute_strategy_metrics,
    create_labels,
    create_nn_validator,
    detect_signals,
    download_market_data,
    extract_validation_features,
    load_raw_data_from_storage,
    log_last_state_metrics,
    run_backtest,
    save_signal_plots,
    save_strategy_plots,
    split_train_val_test,
    validate_signals,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create NN validated pipeline with namespace."""

    base_pipeline = pipeline(
        [
            # 1. Download market data
            node(
                func=download_market_data,
                inputs=[
                    "params:nn_validated.data.store",
                    "params:nn_validated.data.loader",
                    "params:nn_validated.data.period",
                    "params:nn_validated.data.pairs",
                ],
                outputs="nn_store_path",
                name="download_market_data",
                tags=["data_download"],
            ),
            # 2. Load raw data from storage
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:nn_validated.data.store",
                    "params:nn_validated.data.period",
                    "params:nn_validated.data.pairs",
                    "nn_store_path",
                ],
                outputs="nn_raw_data",
                name="load_raw_data_node",
            ),
            # 3. Detect signals
            node(
                func=detect_signals,
                inputs=["nn_raw_data", "params:nn_validated.detector"],
                outputs="nn_raw_signals",
                name="detect_signals_node",
            ),
            # 4. Extract validation features
            node(
                func=extract_validation_features,
                inputs=[
                    "nn_raw_data",
                    "nn_raw_signals",
                    "params:nn_validated.features",
                ],
                outputs="nn_features",
                name="extract_validation_features_node",
                tags=["features", "preparation"],
            ),
            # 5. Create labels
            node(
                func=create_labels,
                inputs=[
                    "nn_raw_data",
                    "nn_raw_signals",
                    "params:nn_validated.labeling",
                ],
                outputs="nn_labels",
                name="generate_labels_node",
            ),
            # 6. Split train/val/test
            node(
                func=split_train_val_test,
                inputs=[
                    "nn_labels",
                    "nn_features",
                    "params:nn_validated.split",
                ],
                outputs=["nn_train_data", "nn_val_data", "nn_test_data"],
                name="split_train_val_test_node",
                tags=["data", "splitting"],
            ),
            # 7. Train NN validator
            node(
                func=create_nn_validator,
                inputs=[
                    "nn_train_data",
                    "nn_val_data",
                    "params:nn_validated.nn_validator.model",
                    "params:nn_validated.nn_validator.trainer",
                ],
                outputs="nn_trained_validator",
                name="train_nn_validator",
                tags=["nn", "training", "validator"],
            ),
            # 8. Validate signals using trained validator
            node(
                func=validate_signals,
                inputs=[
                    "nn_raw_signals",
                    "nn_features",
                    "nn_trained_validator",
                ],
                outputs="nn_validated_signals",
                name="validate_signals_node",
                tags=["signals", "validation"],
            ),
            # 9. Compute signal metrics
            node(
                func=compute_signal_metrics,
                inputs={
                    "params": "params:nn_validated.signal_metrics",
                    "raw_data": "nn_raw_data",
                    "signals": "nn_validated_signals",
                    "labels": "nn_labels",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                },
                outputs=["nn_signal_metrics_results", "nn_signal_plots"],
                name="compute_signal_metrics",
            ),
            # 10. Save signal plots
            node(
                func=save_signal_plots,
                inputs={
                    "plots": "nn_signal_plots",
                    "output_dir": "params:nn_validated.signal_plots_output_dir",
                },
                outputs=None,
                name="save_signal_plots",
            ),
            # 11. Run backtest
            node(
                func=run_backtest,
                inputs=[
                    "nn_raw_data",
                    "nn_validated_signals",
                    "params:nn_validated.strategy",
                ],
                outputs=["nn_backtest_results", "nn_backtest_state"],
                name="run_backtest",
                tags=["backtesting"],
            ),
            # 12. Log backtest metrics
            node(
                func=log_last_state_metrics,
                inputs="nn_backtest_results",
                outputs="nn_backtest_metrics",
                name="log_backtest_metrics",
                tags=["metrics", "backtest_metrics"],
            ),
            # 13. Compute strategy metrics
            node(
                func=compute_strategy_metrics,
                inputs={
                    "backtest_results": "nn_backtest_results",
                    "params": "params:nn_validated.strategy_metrics",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                    "raw_data": "nn_raw_data",
                    "state": "nn_backtest_state",
                },
                outputs=["nn_strategy_metrics_results", "nn_strategy_plots"],
                name="compute_strategy_metrics",
                tags=["metrics", "strategy_metrics"],
            ),
            # 14. Save strategy plots
            node(
                func=save_strategy_plots,
                inputs={
                    "plots": "nn_strategy_plots",
                    "output_dir": "params:nn_validated.strategy_plots_output_dir",
                },
                outputs=None,
                name="save_strategy_plots",
                tags=["reporting"],
            ),
        ]
    )

    return pipeline(
        base_pipeline,
        namespace="nn_validated",
        parameters={
            "params:nn_validated.data.store",
            "params:nn_validated.data.loader",
            "params:nn_validated.data.pairs",
            "params:nn_validated.data.period",
            "params:nn_validated.detector",
            "params:nn_validated.features",
            "params:nn_validated.labeling",
            "params:nn_validated.split",
            "params:nn_validated.nn_validator.model",
            "params:nn_validated.nn_validator.trainer",
            "params:nn_validated.signal_metrics",
            "params:nn_validated.signal_plots_output_dir",
            "params:nn_validated.strategy",
            "params:nn_validated.strategy_metrics",
            "params:nn_validated.strategy_plots_output_dir",
            "params:telegram",
            "params:strategy_name",
        },
    )
