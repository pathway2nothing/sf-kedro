"""Visualization pipeline definition."""

from kedro.pipeline import Pipeline, node

from sf_kedro.pipelines.viz.nodes import (
    generate_flow_dag_visualization,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the visualization pipeline.

    This pipeline generates visualizations for SignalFlow flows.

    Usage:
        kedro run --pipeline=viz --params='flow_id=grid_sma'
        kedro run --pipeline=viz --params='flow_id=grid_sma,viz.output_format=mermaid'

    Nodes:
        - generate_flow_dag_visualization: Flow DAG visualization
    """
    return Pipeline(
        [
            node(
                func=generate_flow_dag_visualization,
                inputs={
                    "flow_id": "params:flow_id",
                    "output_dir": "params:viz.output_dir",
                    "output_format": "params:viz.output_format",
                },
                outputs="viz_result",
                name="generate_flow_dag_visualization",
            ),
        ]
    )
