# sf_kedro/pipelines/backtest/pipeline.py
"""Universal backtest pipeline.

Runs any flow configuration from conf/base/flows/*.yml

Usage:
    kedro run --pipeline=backtest
    kedro run --pipeline=backtest --params='flow_id=grid_sma'
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.pipelines.backtest.nodes import (
    compute_metrics,
    load_flow_data,
    run_flow_backtest,
    run_flow_detection,
    save_flow_plots,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_flow_data,
                inputs=["params:flow_id", "params:common"],
                outputs=["flow_config", "flow_raw_data"],
                name="load_flow_data",
                tags=["data"],
            ),
            node(
                func=run_flow_detection,
                inputs=["flow_config", "flow_raw_data"],
                outputs="flow_signals",
                name="run_flow_detection",
                tags=["detection"],
            ),
            node(
                func=run_flow_backtest,
                inputs=["flow_config", "flow_raw_data", "flow_signals"],
                outputs=["flow_results", "flow_state"],
                name="run_flow_backtest",
                tags=["backtesting"],
            ),
            node(
                func=compute_metrics,
                inputs=["flow_config", "flow_results", "flow_state", "flow_raw_data"],
                outputs="flow_metrics",
                name="compute_metrics",
                tags=["metrics"],
            ),
            node(
                func=save_flow_plots,
                inputs=["flow_config", "flow_metrics"],
                outputs=None,
                name="save_flow_plots",
                tags=["reporting"],
            ),
        ]
    )
