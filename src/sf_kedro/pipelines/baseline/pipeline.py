from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    calculate_signal_metrics,
    run_backtest,
    calculate_backtest_metrics,
)

def create_pipeline(**kwargs) -> Pipeline:
    base = pipeline(
        [
            node(
                func=download_market_data,
                inputs=["params:data.store", "params:data.loader", "params:data.period", "params:data.pairs"],
                outputs="store_path",
                name="download_market_data",
                tags=["data", "data_download"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=["params:data.store", "params:data.period", "params:data.pairs", "store_path"],
                outputs="raw_data",
                name="load_raw_data",
                tags=["data", "data_loading"],
            ),
            node(
                func=detect_signals,
                inputs=["raw_data", "params:baseline.detector"],
                outputs="raw_signals",
                name="detect_signals",
                tags=["signal_detection"],
            ),
            node(
                func=calculate_signal_metrics,
                inputs=["raw_signals", "raw_data"],
                outputs="signal_metrics",
                name="calculate_signal_metrics",
                tags=["metrics", "signal_metrics"],
            ),
            node(
                func=run_backtest,
                inputs=["raw_data", "raw_signals", "params:baseline.strategy"],
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
        base,
        namespace="baseline",
        parameters={
            "params:data.store",
            "params:data.loader",
            "params:data.period",
            "params:data.pairs",
            "params:baseline.strategy",
            "params:baseline.detector",
        },
    )
