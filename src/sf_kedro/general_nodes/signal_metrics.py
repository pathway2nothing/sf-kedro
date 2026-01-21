# # sf_kedro/pipelines/baseline/nodes.py

# from typing import Dict, Any, List, Tuple
# import polars as pl
# import plotly.graph_objects as go
# import mlflow
# import tempfile
# from pathlib import Path
# from loguru import logger
# import signalflow as sf
# from sf_kedro.custom_modules import SignalMetricsProcessor
# from sf_kedro.utils.telegram import send_plots_to_telegram

# def compute_signal_metrics(
#     params: Dict[str, Dict[str, Any]],
#     raw_data: sf.RawData,
#     signals: sf.Signals,
#     labels: pl.DataFrame | None = None,
# ) -> Tuple[Dict[str, Any], Dict[str, List[go.Figure]]]:
#     """
#     Compute signal metrics and log to MLflow.
    
#     Args:
#         params: Dictionary with metric configurations
#         raw_data: Raw OHLCV data
#         signals: Signals container
#         labels: Optional labels container
        
#     Returns:
#         Tuple of (results_dict, plots_dict)
#     """
#     logger.info(f"Computing signal metrics for {len(params)} metric types")
    
#     metrics: List[SignalMetricsProcessor] = []
#     for metric_type, metric_params in params.items():
#         logger.debug(f"Loading metric processor: {metric_type}")
#         metric_processor = sf.get_component(
#             type=sf.SfComponentType.SIGNAL_METRIC,
#             name=metric_type
#         )(**metric_params)
#         metrics.append(metric_processor)
    
#     plots = {}
#     results = {}
    
#     for metric in metrics:
#         metric_name = metric.__class__.__name__
#         logger.info(f"Processing metric: {metric_name}")
        
#         computed_metrics, metric_plots = metric(
#             raw_data=raw_data,
#             signals=signals,
#             labels=labels,
#         )

#         plots[metric_name] = metric_plots
#         results[metric_name] = computed_metrics
        
#         logger.info(
#             f"Metric {metric_name}: computed {len(computed_metrics)} entries, "
#             f"generated {len(metric_plots)if isinstance(metric_plots, list) else 1} plots"
#         )

#     if mlflow.active_run():
#         logger.info("Logging signal metrics to MLflow")
#         _log_metrics_to_mlflow(results, plots)
#     else:
#         logger.debug("No active MLflow run, skipping MLflow logging")
    
#     logger.info(f"Successfully computed {len(results)} metric types")
#     return results, plots


# def save_signal_plots(
#     plots: Dict[str, List[go.Figure]],
#     output_dir: str,
# ) -> None:
#     """
#     Save signal plots to disk as PNG and HTML files.
    
#     Args:
#         plots: Dictionary with {metric_name: [figures]}
#         output_dir: Base directory for saving plots
#     """
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     total_saved = 0
#     for metric_name, figures in plots.items():
#         metric_dir = output_path / metric_name
#         metric_dir.mkdir(exist_ok=True)
        
#         logger.info(f"Saving {len(figures if isinstance(figures, list) else [figures])} plots for {metric_name}")
        
#         for i, fig in enumerate(figures if isinstance(figures, list) else [figures]):
#             png_path = metric_dir / f"{i}.png"
#             fig.write_image(str(png_path), width=1200, height=600, scale=2)
            
#             html_path = metric_dir / f"{i}.html"
#             fig.write_html(str(html_path))
            
#             total_saved += 1
        
#         logger.info(f"Saved {len(figures if isinstance(figures, list) else [figures])} plots to {metric_dir}")
    
#     logger.info(f"Total saved {total_saved} plots to {output_path}")


# def _log_metrics_to_mlflow(
#     results: Dict[str, Any],
#     plots: Dict[str, List[go.Figure] | go.Figure | None]
# ) -> None:
#     """Log computed metrics and plots to MLflow."""
    
#     for metric_name, metric_data in results.items():
#         if isinstance(metric_data, dict):
#             for key, value in metric_data.items():
#                 if isinstance(value, dict):
#                     for subkey, subvalue in value.items():
#                         if isinstance(subvalue, (int, float)):
#                             mlflow.log_metric(
#                                 f"{metric_name}.{key}.{subkey}",
#                                 subvalue
#                             )
#                 elif isinstance(value, (int, float)):
#                     mlflow.log_metric(f"{metric_name}.{key}", value)
    
#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmpdir_path = Path(tmpdir)
        
#         for metric_name, figures in plots.items():
#             metric_dir = tmpdir_path / metric_name
#             metric_dir.mkdir(exist_ok=True)
#             if figures is None:
#                 continue
#             for i, fig in enumerate(figures if isinstance(figures, list) else [figures]):
#                 png_path = metric_dir / f"{i}.png"
#                 fig.write_image(str(png_path), width=1200, height=600, scale=2)
                
#                 html_path = metric_dir / f"{i}.html"
#                 fig.write_html(str(html_path))
            
#             mlflow.log_artifacts(str(metric_dir), artifact_path=f"signal_metrics/{metric_name}")

# =====


# # sf_kedro/pipelines/baseline/nodes.py
# sf_kedro/pipelines/baseline/nodes.py

from typing import Dict, Any, List, Tuple, Optional
import polars as pl
import plotly.graph_objects as go
import mlflow
import tempfile
from pathlib import Path
from datetime import datetime
from loguru import logger
import signalflow as sf
from sf_kedro.custom_modules import SignalMetricsProcessor
from sf_kedro.utils.telegram import send_plots_to_telegram


def compute_signal_metrics(
    params: Dict[str, Dict[str, Any]],
    raw_data: sf.RawData,
    signals: sf.Signals,
    labels: pl.DataFrame | None = None,
    telegram_config: Dict[str, Any] | None = None,
    strategy_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[go.Figure]]]:
    """
    Compute signal metrics and log to MLflow.
    
    Args:
        params: Dictionary with metric configurations
        raw_data: Raw OHLCV data
        signals: Signals container
        labels: Optional labels container
        telegram_config: Optional telegram notification config
        strategy_name: Optional strategy name for header
        
    Returns:
        Tuple of (results_dict, plots_dict)
    """
    logger.info(f"Computing signal metrics for {len(params)} metric types")
    
    metrics: List[SignalMetricsProcessor] = []
    for metric_type, metric_params in params.items():
        logger.debug(f"Loading metric processor: {metric_type}")
        metric_processor = sf.get_component(
            type=sf.SfComponentType.SIGNAL_METRIC,
            name=metric_type
        )(**metric_params)
        metrics.append(metric_processor)
    
    plots = {}
    results = {}
    
    for metric in metrics:
        metric_name = metric.__class__.__name__
        logger.info(f"Processing metric: {metric_name}")
        
        try:
            computed_metrics, metric_plots = metric(
                raw_data=raw_data,
                signals=signals,
                labels=labels,
            )

            plots[metric_name] = metric_plots
            results[metric_name] = computed_metrics
            
            n_plots = len(metric_plots) if isinstance(metric_plots, list) else (1 if metric_plots else 0)
            n_metrics = len(computed_metrics) if computed_metrics else 0
            
            logger.info(
                f"Metric {metric_name}: computed {n_metrics} entries, "
                f"generated {n_plots} plots"
            )
        except Exception as e:
            logger.error(f"Failed to process metric {metric_name}: {e}")
            plots[metric_name] = None
            results[metric_name] = None

    # MLflow logging
    if mlflow.active_run():
        logger.info("Logging signal metrics to MLflow")
        _log_metrics_to_mlflow(results, plots)
    else:
        logger.debug("No active MLflow run, skipping MLflow logging")
    
    # Telegram notification
    if telegram_config and telegram_config.get("enabled", False):
        logger.info("Preparing Telegram notification")
        try:
            # Build header message
            header_template = telegram_config.get(
                "header_message",
                "📊 <b>SignalFlow Metrics Report</b>"
            )
            
            # Get date range from signals
            signals_df = signals.value
            date_min = signals_df["timestamp"].min()
            date_max = signals_df["timestamp"].max()
            n_signals = signals_df.filter(pl.col("signal") != 0).height
            n_pairs = signals_df["pair"].n_unique()
            
            header = header_template.format(
                strategy_name=strategy_name or "Unknown",
                period=f"{date_min} to {date_max}",
                n_signals=n_signals,
                n_pairs=n_pairs,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Add summary stats
            summary_lines = [header, ""]
            for metric_name, metric_data in results.items():
                if metric_data and isinstance(metric_data, dict) and "quant" in metric_data:
                    quant = metric_data["quant"]
                    summary_lines.append(f"<b>{metric_name}:</b>")
                    if "n_signals" in quant:
                        summary_lines.append(f"  • Signals: {quant['n_signals']}")
                    if "final_mean" in quant:
                        summary_lines.append(f"  • Final Mean: {quant['final_mean']:.2f}%")
                    if "avg_max_uplift" in quant:
                        summary_lines.append(f"  • Avg Max: {quant['avg_max_uplift']:.2f}%")
            
            full_header = "\n".join(summary_lines)
            
            send_plots_to_telegram(
                plots=plots,
                bot_token=telegram_config.get("bot_token"),
                chat_id=telegram_config.get("chat_id"),
                header_message=full_header,
                image_width=telegram_config.get("image_width", 1400),
                image_height=telegram_config.get("image_height", 900),
            )
            logger.info("Successfully sent Telegram notification")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            if telegram_config.get("raise_on_error", False):
                raise
    
    logger.info(f"Successfully computed {len(results)} metric types")
    return results, plots


def save_signal_plots(
    plots: Dict[str, List[go.Figure]],
    output_dir: str,
) -> None:
    """
    Save signal plots to disk as PNG and HTML files.
    
    Args:
        plots: Dictionary with {metric_name: [figures]}
        output_dir: Base directory for saving plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    for metric_name, figures in plots.items():
        if figures is None:
            logger.warning(f"Skipping {metric_name}: no figures")
            continue
            
        metric_dir = output_path / metric_name
        metric_dir.mkdir(exist_ok=True)
        
        figs_list = figures if isinstance(figures, list) else [figures]
        logger.info(f"Saving {len(figs_list)} plots for {metric_name}")
        
        for i, fig in enumerate(figs_list):
            png_path = metric_dir / f"{i}.png"
            fig.write_image(str(png_path), width=1400, height=900, scale=2)
            
            html_path = metric_dir / f"{i}.html"
            fig.write_html(str(html_path))
            
            total_saved += 1
        
        logger.info(f"Saved {len(figs_list)} plots to {metric_dir}")
    
    logger.info(f"Total saved {total_saved} plots to {output_path}")


def _log_metrics_to_mlflow(
    results: Dict[str, Any],
    plots: Dict[str, List[go.Figure] | go.Figure | None]
) -> None:
    """Log computed metrics and plots to MLflow."""
    
    for metric_name, metric_data in results.items():
        if metric_data is None:
            continue
            
        if isinstance(metric_data, dict):
            for key, value in metric_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            mlflow.log_metric(
                                f"{metric_name}.{key}.{subkey}",
                                subvalue
                            )
                elif isinstance(value, (int, float)):
                    mlflow.log_metric(f"{metric_name}.{key}", value)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        for metric_name, figures in plots.items():
            if figures is None:
                continue
                
            metric_dir = tmpdir_path / metric_name
            metric_dir.mkdir(exist_ok=True)
            
            figs_list = figures if isinstance(figures, list) else [figures]
            
            for i, fig in enumerate(figs_list):
                png_path = metric_dir / f"{i}.png"
                fig.write_image(str(png_path), width=1400, height=900, scale=2)
                
                html_path = metric_dir / f"{i}.html"
                fig.write_html(str(html_path))
            
            mlflow.log_artifacts(str(metric_dir), artifact_path=f"signal_metrics/{metric_name}")