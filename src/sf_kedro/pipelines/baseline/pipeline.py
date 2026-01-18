"""Baseline pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.modules import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    calculate_signal_metrics,
    run_backtest,
    calculate_backtest_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create baseline pipeline without namespace."""
    
    return pipeline([
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
        
        # Signal detection
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
        
        # Backtesting
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
    ])
