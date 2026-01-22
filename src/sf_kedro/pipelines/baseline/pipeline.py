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
    calculate_backtest_metrics, 
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
                outputs="store_path",
                name="download_market_data",
                tags=["data_download"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:baseline.data.store",
                    "params:baseline.data.period", 
                    "params:baseline.data.pairs",
                    "store_path"
                ],
                outputs="raw_data",
                name="load_raw_data_node",
            ),
            node(
                func=detect_signals,
                inputs=["raw_data", "params:baseline.detector"],
                outputs="signals",
                name="detect_signals_node",
            ),
            node(
                func=create_labels,
                inputs=["raw_data", "signals", "params:baseline.labeling"],
                outputs="labels",
                name="generate_labels_node",
            ),
            node(
                func=compute_signal_metrics,
                inputs={
                    "params": "params:baseline.signal_metrics",
                    "raw_data": "raw_data",
                    "signals": "signals",
                    "labels": "labels",
                    "telegram_config": "params:telegram",
                    "strategy_name": "params:strategy_name",
                },
                outputs=["signal_metrics_results", "signal_plots"],
                name="compute_signal_metrics",
            ),
            node(
                func=save_signal_plots,
                inputs={
                    "plots": "signal_plots",
                    "output_dir": "params:baseline.signal_plots_output_dir",
                },
                outputs=None,
                name="save_signal_plots",
            ),
            node(
                func=run_backtest,
                inputs=["raw_data", "signals", "params:baseline.strategy"],
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
            "params:baseline.strategy_plots_output_dir  ",          
            "params:telegram",
            "params:strategy_name",
        }
    )