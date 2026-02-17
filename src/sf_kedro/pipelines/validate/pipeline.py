"""Validate pipeline - walk-forward validation.

Usage:
    kedro run --pipeline=validate
    kedro run --pipeline=validate --params='flow_id=grid_sma,n_folds=5'
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.pipelines.validate.nodes import (
    load_validation_data,
    run_walk_forward,
    save_validation_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_validation_data,
                inputs=["params:flow_id", "params:n_folds"],
                outputs=["validate_config", "validate_raw_data"],
                name="load_validation_data",
            ),
            node(
                func=run_walk_forward,
                inputs=["validate_config", "validate_raw_data"],
                outputs="validation_results",
                name="run_walk_forward",
            ),
            node(
                func=save_validation_report,
                inputs=["validate_config", "validation_results"],
                outputs=None,
                name="save_validation_report",
            ),
        ]
    )
