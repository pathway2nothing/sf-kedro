"""Train pipeline - validator training.

Usage:
    kedro run --pipeline=train
    kedro run --pipeline=train --params='flow_id=grid_sma,validator_type=xgboost'
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.pipelines.train.nodes import (
    load_training_data,
    prepare_features,
    save_trained_model,
    train_validator,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_training_data,
                inputs=["params:flow_id", "params:validator_type"],
                outputs=["train_config", "train_raw_data", "train_signals", "train_labeled_data"],
                name="load_training_data",
            ),
            node(
                func=prepare_features,
                inputs=["train_config", "train_raw_data", "train_signals", "train_labeled_data"],
                outputs=["X_train", "y_train", "X_val", "y_val"],
                name="prepare_features",
            ),
            node(
                func=train_validator,
                inputs=["train_config", "X_train", "y_train", "X_val", "y_val"],
                outputs="trained_validator",
                name="train_validator",
            ),
            node(
                func=save_trained_model,
                inputs=["train_config", "trained_validator"],
                outputs=None,
                name="save_trained_model",
            ),
        ]
    )
