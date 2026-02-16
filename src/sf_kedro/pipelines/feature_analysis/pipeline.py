"""Feature analysis pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from sf_kedro.general_nodes import (
    download_market_data,
    load_raw_data_from_storage,
)

from .nodes import (
    build_feature_analysis_plots,
    extract_features_for_analysis,
    save_feature_analysis_plots,
)


def create_pipeline(**kwargs) -> Pipeline:
    base_pipeline = pipeline(
        [
            node(
                func=download_market_data,
                inputs=[
                    "params:feature_analysis.data.store",
                    "params:feature_analysis.data.loader",
                    "params:feature_analysis.data.period",
                    "params:feature_analysis.data.pairs",
                ],
                outputs="fa_store_path",
                name="download_market_data",
                tags=["data_download"],
            ),
            node(
                func=load_raw_data_from_storage,
                inputs=[
                    "params:feature_analysis.data.store",
                    "params:feature_analysis.data.period",
                    "params:feature_analysis.data.pairs",
                    "fa_store_path",
                ],
                outputs="fa_raw_data",
                name="load_raw_data_node",
            ),
            node(
                func=extract_features_for_analysis,
                inputs=[
                    "fa_raw_data",
                    "params:feature_analysis.features",
                ],
                outputs="fa_features",
                name="extract_features_node",
            ),
            node(
                func=build_feature_analysis_plots,
                inputs={
                    "raw_data": "fa_raw_data",
                    "features_df": "fa_features",
                    "analysis_params": "params:feature_analysis.analysis",
                    "telegram_config": "params:telegram",
                },
                outputs="fa_plots",
                name="build_feature_analysis_plots",
            ),
            node(
                func=save_feature_analysis_plots,
                inputs={
                    "plots": "fa_plots",
                    "output_dir": "params:feature_analysis.plots_output_dir",
                },
                outputs=None,
                name="save_feature_plots",
                tags=["reporting"],
            ),
        ]
    )

    return pipeline(
        base_pipeline,
        namespace="feature_analysis",
        parameters={
            "params:feature_analysis.data.store",
            "params:feature_analysis.data.loader",
            "params:feature_analysis.data.pairs",
            "params:feature_analysis.data.period",
            "params:feature_analysis.features",
            "params:feature_analysis.analysis",
            "params:feature_analysis.plots_output_dir",
            "params:telegram",
        },
    )
