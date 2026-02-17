"""Tune pipeline - parameter optimization with Optuna.

Usage:
    kedro run --pipeline=tune
    kedro run --pipeline=tune --params='flow_id=grid_sma,level=detector'
    kedro run --pipeline=tune --params='flow_id=grid_sma,level=strategy,n_trials=100'
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.pipelines.tune.nodes import (
    load_tune_data,
    run_optimization,
    save_best_params,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_tune_data,
                inputs=["params:flow_id", "params:level", "params:n_trials"],
                outputs=["tune_config", "tune_raw_data"],
                name="load_tune_data",
            ),
            node(
                func=run_optimization,
                inputs=["tune_config", "tune_raw_data"],
                outputs="optimization_results",
                name="run_optimization",
            ),
            node(
                func=save_best_params,
                inputs=["tune_config", "optimization_results"],
                outputs=None,
                name="save_best_params",
            ),
        ]
    )
