# sf_kedro/pipelines/baseline/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    create_labels,
    compute_signal_metrics,
    save_signal_plots,
    run_backtest,
    log_last_state_metrics, 
    save_strategy_plots,
    compute_strategy_metrics
)


def create_pipeline(**kwargs) -> Pipeline:
    base_pipeline = pipeline(
        [        
            node(
                func=download_market_data,
                inputs=[
                    "params:baseline.data.store",
                    "params:baseline.data.loader",
                    "params:baseline.data.period",
                    "params:baseline.data.pairs",
                ],
                outputs="base_store_path",
                name="download_market_data",
                tags=["data_download"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:baseline.data.store",
                    "params:baseline.data.period", 
                    "params:baseline.data.pairs",
                    "base_store_path"
                ],
                outputs="base_raw_data",
                name="load_raw_data_node",
            ),
            node(
                func=detect_signals,
                inputs=["base_raw_data", "params:baseline.detector"],
                outputs="base_signals",
                name="detect_signals_node",
            ),
            node(
                func=create_labels,
                inputs=["base_raw_data", "base_signals", "params:baseline.labeling"],
                outputs="base_labels",
                name="generate_labels_node",
            ),
            node(
                func=compute_signal_metrics,
                inputs={
                    "params": "params:baseline.signal_metrics",
                    "raw_data": "base_raw_data",
                    "signals": "base_signals",
                    "labels": "base_labels",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                },
                outputs=["base_signal_metrics_results", "base_signal_plots"],
                name="compute_signal_metrics",
            ),
            node(
                func=save_signal_plots,
                inputs={
                    "plots": "base_signal_plots",
                    "output_dir": "params:baseline.signal_plots_output_dir",
                },
                outputs=None,
                name="save_signal_plots",
            ),
            node(
                func=run_backtest,
                inputs=["base_raw_data", "base_signals", "params:baseline.strategy"],
                outputs=["base_backtest_results", "base_backtest_state"],
                name="run_backtest",
                tags=["backtesting"],
            ),
            node(
                func=log_last_state_metrics,
                inputs="base_backtest_results",
                outputs="base_backtest_metrics",
                name="log_backtest_metrics",
                tags=["metrics", "backtest_metrics"],
            ),
            node(
                func=compute_strategy_metrics,
                inputs={
                    "backtest_results": "base_backtest_results",
                    "params": "params:baseline.strategy_metrics",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                    "raw_data": "base_raw_data",
                    "state": "base_backtest_state",
                },
                outputs=["base_strategy_metrics_results", "base_strategy_plots"], 
                name="compute_strategy_metrics",
                tags=["metrics", "strategy_metrics"],
            ),
            node(
                func=save_strategy_plots,
                inputs={
                    "plots": "base_strategy_plots",
                    "output_dir": "params:baseline.strategy_plots_output_dir"
                },
                outputs=None,
                name="save_strategy_plots",
                tags=["reporting"],
            ),
        ]
    )

    return pipeline(
        base_pipeline,
        namespace="baseline",
        parameters={
            "params:baseline.data.store",
            "params:baseline.data.loader",
            "params:baseline.data.pairs",
            "params:baseline.data.period",
            "params:baseline.detector",
            "params:baseline.labeling",
            "params:baseline.signal_metrics",
            "params:baseline.signal_plots_output_dir",
            "params:baseline.strategy",
            "params:baseline.strategy_metrics",
            "params:baseline.strategy_plots_output_dir",          
            "params:telegram",
            "params:strategy_name",
        }
    )