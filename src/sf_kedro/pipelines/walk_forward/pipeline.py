"""Walk-Forward validation pipeline for SignalFlow.

Walk-Forward validation (rolling window validation) provides more realistic
backtesting by:

1. Training on historical data
2. Testing on the immediately following period
3. Rolling forward and repeating

This avoids look-ahead bias and provides out-of-sample performance estimates.

Example usage:
    kedro run --pipeline walk_forward

Configuration in parameters.yml:
    walk_forward:
      windows:
        train_size: 60  # days
        test_size: 14   # days
        step_size: 14   # days
        n_windows: null # auto-calculate from data
      validator:
        model_type: "lightgbm"
        model_params: {}
      strategy:
        initial_capital: 10000
        entry:
          position_size_pct: 0.1
          max_positions: 5
        exit:
          take_profit_pct: 0.02
          stop_loss_pct: 0.01
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
    detect_signals,
    extract_validation_features,
    create_labels,
)

from .nodes import (
    create_walk_forward_windows,
    run_walk_forward_validation,
    save_walk_forward_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the Walk-Forward validation pipeline.

    Returns:
        Kedro Pipeline for walk-forward validation
    """
    base_pipeline = pipeline(
        [
            # Data preparation
            node(
                func=download_market_data,
                inputs=[
                    "params:walk_forward.data.store",
                    "params:walk_forward.data.loader",
                    "params:walk_forward.data.period",
                    "params:walk_forward.data.pairs",
                ],
                outputs="wf_store_path",
                name="download_market_data",
                tags=["data_download", "preparation"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:walk_forward.data.store",
                    "params:walk_forward.data.period",
                    "params:walk_forward.data.pairs",
                    "wf_store_path",
                ],
                outputs="wf_raw_data",
                name="load_raw_data_node",
                tags=["preparation"],
            ),
            node(
                func=detect_signals,
                inputs=["wf_raw_data", "params:walk_forward.detector"],
                outputs="wf_signals",
                name="detect_signals_node",
                tags=["preparation"],
            ),
            node(
                func=extract_validation_features,
                inputs=[
                    "wf_raw_data",
                    "wf_signals",
                    "params:walk_forward.features",
                ],
                outputs="wf_features",
                name="extract_features_node",
                tags=["preparation", "features"],
            ),
            node(
                func=create_labels,
                inputs=[
                    "wf_raw_data",
                    "wf_signals",
                    "params:walk_forward.labeling",
                ],
                outputs="wf_labels",
                name="create_labels_node",
                tags=["preparation"],
            ),
            # Walk-forward specific nodes
            node(
                func=create_walk_forward_windows,
                inputs=[
                    "wf_raw_data",
                    "params:walk_forward.windows",
                ],
                outputs="wf_windows",
                name="create_windows_node",
                tags=["walk_forward", "setup"],
            ),
            node(
                func=run_walk_forward_validation,
                inputs=[
                    "wf_raw_data",
                    "wf_signals",
                    "wf_features",
                    "wf_labels",
                    "wf_windows",
                    "params:walk_forward.validation",
                ],
                outputs="wf_result",
                name="run_walk_forward_node",
                tags=["walk_forward", "validation"],
            ),
            node(
                func=save_walk_forward_results,
                inputs=[
                    "wf_result",
                    "params:walk_forward.output_dir",
                ],
                outputs="wf_output_path",
                name="save_results_node",
                tags=["walk_forward", "reporting"],
            ),
        ]
    )

    return pipeline(
        base_pipeline,
        namespace="walk_forward",
        parameters={
            "params:walk_forward.data.store",
            "params:walk_forward.data.loader",
            "params:walk_forward.data.pairs",
            "params:walk_forward.data.period",
            "params:walk_forward.detector",
            "params:walk_forward.features",
            "params:walk_forward.labeling",
            "params:walk_forward.windows",
            "params:walk_forward.validation",
            "params:walk_forward.output_dir",
        },
    )


def create_anchored_walk_forward_pipeline(**kwargs) -> Pipeline:
    """Create anchored walk-forward validation pipeline.

    In anchored walk-forward, the training window always starts from
    the beginning of the data (expanding window), rather than rolling.

    This is useful when you want to use all available historical data
    for training.
    """
    # Uses same pipeline but with anchored=True in windows config
    return create_pipeline(**kwargs)


def create_purged_walk_forward_pipeline(**kwargs) -> Pipeline:
    """Create purged walk-forward validation pipeline.

    Purged walk-forward adds a gap between train and test periods
    to avoid data leakage from lagged features or labels.

    Configure with:
        windows:
          purge_gap: 5  # days between train_end and test_start
    """
    return create_pipeline(**kwargs)
