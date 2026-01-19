# sf_kedro/pipelines/baseline/nodes.py

from sf_kedro.custom_modules import SignalMetrics, SignalMetricsConfig
import mlflow
import tempfile
from pathlib import Path
import plotly.io as pio


def compute_signal_metrics(
    raw_df: pl.DataFrame,
    signals_df: pl.DataFrame,
    params: dict
) -> dict:
    """
    Compute signal metrics and log to MLflow
    
    Args:
        raw_df: Raw OHLCV data
        signals_df: Signals dataframe
        params: Parameters from conf
        
    Returns:
        Dictionary with scalar metrics
    """
    metrics_config = SignalMetricsConfig(
        look_ahead=params.get("look_ahead", 1440),
        quantiles=tuple(params.get("quantiles", [0.25, 0.75])),
        pairs_to_plot=params.get("pairs_to_plot", ["BTCUSDT", "ETHUSDT"])
    )
    
    metrics_computer = SignalMetrics(metrics_config)
    
    # Compute all metrics
    results = metrics_computer.compute_all_metrics(raw_df, signals_df)
    
    # Log scalar metrics to MLflow
    if mlflow.active_run():
        mlflow.log_metrics(results["scalar_metrics"])
        
        # Log figures as artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            for fig_name, fig in results["figures"].items():
                # Save as HTML (interactive)
                html_path = tmpdir_path / f"{fig_name}.html"
                pio.write_html(fig, str(html_path))
                mlflow.log_artifact(str(html_path), artifact_path="plots")
                
                # Save as PNG (static)
                png_path = tmpdir_path / f"{fig_name}.png"
                pio.write_image(fig, str(png_path), width=1400, height=900)
                mlflow.log_artifact(str(png_path), artifact_path="plots")
    
    return results["scalar_metrics"]