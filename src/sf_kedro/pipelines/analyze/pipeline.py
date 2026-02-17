"""Analyze pipeline - feature and signal analysis.

Usage:
    kedro run --pipeline=analyze
    kedro run --pipeline=analyze --params='flow_id=grid_sma,level=signals'
"""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.pipelines.analyze.nodes import (
    analyze_features,
    analyze_signals,
    load_analysis_data,
    save_analysis_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_analysis_data,
                inputs=["params:flow_id", "params:level"],
                outputs=["analysis_config", "analysis_raw_data", "analysis_signals"],
                name="load_analysis_data",
            ),
            node(
                func=analyze_features,
                inputs=["analysis_config", "analysis_raw_data"],
                outputs="feature_analysis_results",
                name="analyze_features",
            ),
            node(
                func=analyze_signals,
                inputs=["analysis_config", "analysis_raw_data", "analysis_signals"],
                outputs="signal_analysis_results",
                name="analyze_signals",
            ),
            node(
                func=save_analysis_report,
                inputs=["analysis_config", "feature_analysis_results", "signal_analysis_results"],
                outputs=None,
                name="save_analysis_report",
            ),
        ]
    )
