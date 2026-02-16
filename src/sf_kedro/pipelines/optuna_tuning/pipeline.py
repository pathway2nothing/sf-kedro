"""Optuna hyperparameter tuning pipeline for SignalFlow.

This pipeline provides three tuning modes:
1. Validator tuning - Optimize ML model hyperparameters
2. Detector tuning - Optimize signal detection parameters
3. Strategy tuning - Optimize entry/exit parameters

Each mode can be run independently or combined for full pipeline optimization.

Example usage:
    # Run validator tuning only
    kedro run --pipeline optuna_tuning --tags validator_tuning

    # Run full optimization
    kedro run --pipeline optuna_tuning

Configuration in parameters.yml:
    optuna_tuning:
      study:
        study_name: "signalflow_optimization"
        direction: "maximize"
        sampler:
          type: "tpe"
          seed: 42
        pruner:
          type: "median"
      validator:
        n_trials: 50
        metric: "accuracy"
        model_size: "medium"
      detector:
        n_trials: 30
        detector_name: "example/sma_cross"
        metric: "precision"
      strategy:
        n_trials: 50
        metric: "sharpe"
        initial_capital: 10000
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    extract_validation_features,
    create_labels,
    split_train_val_test,
)

from .nodes import (
    create_optuna_study,
    tune_validator,
    tune_detector,
    tune_strategy,
    save_optuna_study,
    apply_best_params,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Optuna tuning pipeline.

    Returns:
        Kedro Pipeline with all tuning nodes
    """
    # Data preparation nodes (shared)
    data_prep = pipeline(
        [
            node(
                func=download_market_data,
                inputs=[
                    "params:optuna_tuning.data.store",
                    "params:optuna_tuning.data.loader",
                    "params:optuna_tuning.data.period",
                    "params:optuna_tuning.data.pairs",
                ],
                outputs="optuna_store_path",
                name="download_market_data",
                tags=["data_download", "preparation"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:optuna_tuning.data.store",
                    "params:optuna_tuning.data.period",
                    "params:optuna_tuning.data.pairs",
                    "optuna_store_path",
                ],
                outputs="optuna_raw_data",
                name="load_raw_data_node",
                tags=["preparation"],
            ),
            node(
                func=detect_signals,
                inputs=["optuna_raw_data", "params:optuna_tuning.detector"],
                outputs="optuna_raw_signals",
                name="detect_initial_signals",
                tags=["preparation"],
            ),
            node(
                func=extract_validation_features,
                inputs=[
                    "optuna_raw_data",
                    "optuna_raw_signals",
                    "params:optuna_tuning.features",
                ],
                outputs="optuna_features",
                name="extract_features_node",
                tags=["preparation", "features"],
            ),
            node(
                func=create_labels,
                inputs=[
                    "optuna_raw_data",
                    "optuna_raw_signals",
                    "params:optuna_tuning.labeling",
                ],
                outputs="optuna_labels",
                name="create_labels_node",
                tags=["preparation"],
            ),
            node(
                func=split_train_val_test,
                inputs=[
                    "optuna_labels",
                    "optuna_features",
                    "params:optuna_tuning.split",
                ],
                outputs=["optuna_train_data", "optuna_val_data", "optuna_test_data"],
                name="split_data_node",
                tags=["preparation", "splitting"],
            ),
        ]
    )

    # Validator tuning pipeline
    validator_tuning = pipeline(
        [
            node(
                func=create_optuna_study,
                inputs="params:optuna_tuning.study",
                outputs="validator_study",
                name="create_validator_study",
                tags=["validator_tuning", "optuna"],
            ),
            node(
                func=tune_validator,
                inputs=[
                    "validator_study",
                    "optuna_train_data",
                    "optuna_val_data",
                    "params:optuna_tuning.validator_tuning",
                ],
                outputs=["best_validator_params", "tuned_validator"],
                name="tune_validator_node",
                tags=["validator_tuning", "optuna", "ml"],
            ),
            node(
                func=save_optuna_study,
                inputs=[
                    "validator_study",
                    "params:optuna_tuning.output_dir",
                ],
                outputs="validator_study_path",
                name="save_validator_study",
                tags=["validator_tuning", "reporting"],
            ),
        ]
    )

    # Detector tuning pipeline
    detector_tuning = pipeline(
        [
            node(
                func=create_optuna_study,
                inputs="params:optuna_tuning.detector_study",
                outputs="detector_study",
                name="create_detector_study",
                tags=["detector_tuning", "optuna"],
            ),
            node(
                func=tune_detector,
                inputs=[
                    "detector_study",
                    "optuna_raw_data",
                    "optuna_labels",
                    "params:optuna_tuning.detector_tuning",
                ],
                outputs=["best_detector_params", "tuned_signals"],
                name="tune_detector_node",
                tags=["detector_tuning", "optuna"],
            ),
            node(
                func=save_optuna_study,
                inputs=[
                    "detector_study",
                    "params:optuna_tuning.output_dir",
                ],
                outputs="detector_study_path",
                name="save_detector_study",
                tags=["detector_tuning", "reporting"],
            ),
        ]
    )

    # Strategy tuning pipeline
    strategy_tuning = pipeline(
        [
            node(
                func=create_optuna_study,
                inputs="params:optuna_tuning.strategy_study",
                outputs="strategy_study",
                name="create_strategy_study",
                tags=["strategy_tuning", "optuna"],
            ),
            node(
                func=tune_strategy,
                inputs=[
                    "strategy_study",
                    "optuna_raw_data",
                    "optuna_raw_signals",
                    "params:optuna_tuning.strategy_tuning",
                ],
                outputs=["best_strategy_params", "tuning_backtest_results"],
                name="tune_strategy_node",
                tags=["strategy_tuning", "optuna", "backtest"],
            ),
            node(
                func=save_optuna_study,
                inputs=[
                    "strategy_study",
                    "params:optuna_tuning.output_dir",
                ],
                outputs="strategy_study_path",
                name="save_strategy_study",
                tags=["strategy_tuning", "reporting"],
            ),
        ]
    )

    # Combine all pipelines
    full_pipeline = data_prep + validator_tuning + detector_tuning + strategy_tuning

    return pipeline(
        full_pipeline,
        namespace="optuna_tuning",
        parameters={
            "params:optuna_tuning.data.store",
            "params:optuna_tuning.data.loader",
            "params:optuna_tuning.data.pairs",
            "params:optuna_tuning.data.period",
            "params:optuna_tuning.detector",
            "params:optuna_tuning.features",
            "params:optuna_tuning.labeling",
            "params:optuna_tuning.split",
            "params:optuna_tuning.study",
            "params:optuna_tuning.detector_study",
            "params:optuna_tuning.strategy_study",
            "params:optuna_tuning.validator_tuning",
            "params:optuna_tuning.detector_tuning",
            "params:optuna_tuning.strategy_tuning",
            "params:optuna_tuning.output_dir",
        },
    )


def create_validator_only_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline for validator tuning only.

    Use when you only need to tune the ML validator without
    re-tuning detector or strategy parameters.
    """
    base = create_pipeline(**kwargs)
    return base.only_nodes_with_tags("preparation", "validator_tuning")


def create_detector_only_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline for detector tuning only."""
    base = create_pipeline(**kwargs)
    return base.only_nodes_with_tags("preparation", "detector_tuning")


def create_strategy_only_pipeline(**kwargs) -> Pipeline:
    """Create a pipeline for strategy tuning only."""
    base = create_pipeline(**kwargs)
    return base.only_nodes_with_tags("preparation", "strategy_tuning")
