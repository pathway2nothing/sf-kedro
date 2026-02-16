"""Production pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    compute_signal_metrics,
    compute_strategy_metrics,
    detect_signals,
    extract_validation_features,
    load_raw_data_from_storage,
    load_validator_from_registry,
    log_last_state_metrics,
    run_backtest,
    save_signal_plots,
    save_strategy_plots,
    validate_signals,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create production simulation pipeline with namespace."""

    base_pipeline = pipeline(
        [
            # 1. Load raw data from storage (no download — data already present)
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:production.data.store",
                    "params:production.data.period",
                    "params:production.data.pairs",
                    "params:production.data.store_path",
                ],
                outputs="prod_raw_data",
                name="load_raw_data_node",
            ),
            # 2. Detect signals
            node(
                func=detect_signals,
                inputs=["prod_raw_data", "params:production.detector"],
                outputs="prod_raw_signals",
                name="detect_signals_node",
            ),
            # 3. Extract validation features
            node(
                func=extract_validation_features,
                inputs=[
                    "prod_raw_data",
                    "prod_raw_signals",
                    "params:production.features",
                ],
                outputs="prod_features",
                name="extract_validation_features_node",
                tags=["features"],
            ),
            # 4. Load pre-trained validator from MLflow registry
            node(
                func=load_validator_from_registry,
                inputs=[
                    "params:production.model.name",
                    "params:production.model.stage",
                ],
                outputs="prod_validator",
                name="load_production_validator",
                tags=["model_loading"],
            ),
            # 5. Validate signals
            node(
                func=validate_signals,
                inputs=[
                    "prod_raw_signals",
                    "prod_features",
                    "prod_validator",
                ],
                outputs="prod_validated_signals",
                name="validate_signals_node",
                tags=["signals", "validation"],
            ),
            # 6. Compute signal metrics
            node(
                func=compute_signal_metrics,
                inputs={
                    "params": "params:production.signal_metrics",
                    "raw_data": "prod_raw_data",
                    "signals": "prod_validated_signals",
                    "labels": "params:production.empty_labels",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                },
                outputs=["prod_signal_metrics_results", "prod_signal_plots"],
                name="compute_signal_metrics",
            ),
            # 7. Save signal plots
            node(
                func=save_signal_plots,
                inputs={
                    "plots": "prod_signal_plots",
                    "output_dir": "params:production.signal_plots_output_dir",
                },
                outputs=None,
                name="save_signal_plots",
            ),
            # 8. Run backtest
            node(
                func=run_backtest,
                inputs=[
                    "prod_raw_data",
                    "prod_validated_signals",
                    "params:production.strategy",
                ],
                outputs=["prod_backtest_results", "prod_backtest_state"],
                name="run_backtest",
                tags=["backtesting"],
            ),
            # 9. Log backtest metrics
            node(
                func=log_last_state_metrics,
                inputs="prod_backtest_results",
                outputs="prod_backtest_metrics",
                name="log_backtest_metrics",
                tags=["metrics", "backtest_metrics"],
            ),
            # 10. Compute strategy metrics
            node(
                func=compute_strategy_metrics,
                inputs={
                    "backtest_results": "prod_backtest_results",
                    "params": "params:production.strategy_metrics",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                    "raw_data": "prod_raw_data",
                    "state": "prod_backtest_state",
                },
                outputs=["prod_strategy_metrics_results", "prod_strategy_plots"],
                name="compute_strategy_metrics",
                tags=["metrics", "strategy_metrics"],
            ),
            # 11. Save strategy plots
            node(
                func=save_strategy_plots,
                inputs={
                    "plots": "prod_strategy_plots",
                    "output_dir": "params:production.strategy_plots_output_dir",
                },
                outputs=None,
                name="save_strategy_plots",
                tags=["reporting"],
            ),
        ]
    )

    return pipeline(
        base_pipeline,
        namespace="production",
        parameters={
            "params:production.data.store",
            "params:production.data.period",
            "params:production.data.pairs",
            "params:production.data.store_path",
            "params:production.detector",
            "params:production.features",
            "params:production.model.name",
            "params:production.model.stage",
            "params:production.signal_metrics",
            "params:production.signal_plots_output_dir",
            "params:production.strategy",
            "params:production.strategy_metrics",
            "params:production.strategy_plots_output_dir",
            "params:production.empty_labels",
            "params:telegram",
            "params:strategy_name",
        },
    )
