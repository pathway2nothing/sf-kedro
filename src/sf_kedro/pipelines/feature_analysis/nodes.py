"""Feature analysis pipeline nodes."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

import signalflow as sf
from sf_kedro.general_nodes.feature_builder import create_feature_set
from sf_kedro.utils.telegram import TelegramNotifier

# Color palette for pairs (consistent with project style)
PAIR_COLORS = [
    "#2171b5",
    "#e6550d",
    "#31a354",
    "#756bb1",
    "#d62728",
    "#ff7f0e",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
]


def extract_features_for_analysis(
    raw_data: sf.RawData,
    feature_configs: Dict,
) -> pl.DataFrame:
    """
    Extract features from raw data for analysis.

    Args:
        raw_data: Raw OHLCV market data
        feature_configs: Feature extractor configurations

    Returns:
        DataFrame with columns: timestamp, pair, feature_1, ...
    """
    feature_set = create_feature_set(feature_configs)
    raw_data_view = sf.RawDataView(raw_data)
    features_df = feature_set.run(raw_data_view)

    feature_cols = [c for c in features_df.columns if c not in ("timestamp", "pair")]
    logger.info(
        f"Extracted {len(feature_cols)} features: {feature_cols}, "
        f"{features_df.height} rows, "
        f"{features_df['pair'].n_unique()} pairs"
    )
    return features_df


def build_feature_analysis_plots(
    raw_data: sf.RawData,
    features_df: pl.DataFrame,
    analysis_params: Dict[str, Any],
    telegram_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[go.Figure]]:
    """
    Build 4 feature analysis plots and optionally send to Telegram.

    Plots:
      1. Feature across pairs (line chart)
      2. Feature vs price for one pair (dual y-axis)
      3. Feature value distribution (histogram)
      4. Normalized feature across pairs (z-score)

    Args:
        raw_data: Raw OHLCV data
        features_df: Extracted features
        analysis_params: Dict with feature_name, price_pair, price_col, n_bins, indicator_type
        telegram_config: Optional Telegram config

    Returns:
        Dict with {"feature_analysis": [fig1, fig2, fig3, fig4]}
    """
    feature_name = analysis_params["feature_name"]
    price_pair = analysis_params.get("price_pair", "BTCUSDT")
    price_col = analysis_params.get("price_col", "close")
    n_bins = analysis_params.get("n_bins", 50)
    indicator_type = analysis_params.get("indicator_type", None)

    pairs = sorted(features_df["pair"].unique().to_list())
    logger.info(
        f"Building feature analysis plots: feature={feature_name}, "
        f"price_pair={price_pair}, pairs={pairs}"
    )

    # Calculate statistics
    stats = _calculate_feature_statistics(features_df, feature_name, pairs)

    # Check if we have valid data
    if not stats or stats.get("count", 0) == 0:
        logger.error(
            f"No valid data found for feature '{feature_name}'. "
            "Skipping plot generation and Telegram notification."
        )
        # Return empty plots dict to prevent downstream errors
        return {"feature_analysis": []}

    # Calculate correlations and quality metrics for comprehensive stats
    correlations = _calculate_pair_correlations(
        features_df, raw_data, feature_name, pairs, price_col
    )
    quality_metrics = _calculate_data_quality(features_df, feature_name, pairs)

    # Save comprehensive statistics to file for batch reporting
    _save_statistics_to_file(
        feature_name=feature_name,
        indicator_type=indicator_type,
        stats=stats,
        correlations=correlations,
        quality_metrics=quality_metrics,
        output_dir="data/08_reporting/feature_analysis",
    )

    fig1 = _plot_feature_across_pairs(features_df, feature_name, pairs)
    fig2 = _plot_feature_vs_price(
        raw_data, features_df, feature_name, price_pair, price_col
    )
    fig3 = _plot_feature_distribution(features_df, feature_name, pairs, n_bins)
    fig4 = _plot_feature_normalized(features_df, feature_name, pairs)

    plots = {"feature_analysis": [fig1, fig2, fig3, fig4]}

    # Generate text report if Telegram is enabled
    text_report = None
    if telegram_config and telegram_config.get("enabled", False):
        # Check if text report is enabled (default: True)
        if telegram_config.get("send_text_report", True):
            logger.info("Generating detailed text report...")
            text_report = _generate_text_report(
                features_df=features_df,
                raw_data=raw_data,
                feature_name=feature_name,
                pairs=pairs,
                price_col=price_col,
                stats=stats,
                indicator_type=indicator_type,
            )

        _send_to_telegram(
            plots,
            telegram_config,
            feature_name,
            pairs,
            indicator_type,
            stats,
            text_report,
        )

    return plots


def save_feature_analysis_plots(
    plots: Dict[str, List[go.Figure]],
    output_dir: str,
) -> None:
    """
    Save feature analysis plots to disk as PNG and HTML.

    Args:
        plots: Dict with {name: [figures]}
        output_dir: Base directory for saving
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for name, figures in plots.items():
        if figures is None or len(figures) == 0:
            logger.info(f"No plots to save for '{name}' (empty or None)")
            continue

        name_dir = output_path / name
        name_dir.mkdir(exist_ok=True)

        for i, fig in enumerate(figures):
            fig.write_image(str(name_dir / f"{i}.png"), width=1400, height=900, scale=2)
            fig.write_html(str(name_dir / f"{i}.html"))
            total_saved += 1

        logger.info(f"Saved {len(figures)} plots to {name_dir}")

    logger.info(f"Total saved {total_saved} plots to {output_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _calculate_feature_statistics(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
) -> Dict[str, Any]:
    """Calculate statistical metrics for the feature across all pairs."""
    # Check if feature exists in dataframe
    if feature_name not in features_df.columns:
        logger.warning(
            f"Feature '{feature_name}' not found in features_df. Available columns: {features_df.columns}"
        )
        return {}

    all_values = (
        features_df.drop_nulls(subset=[feature_name])[feature_name]
        .to_numpy()
        .astype(float)
    )

    # Filter out NaN and inf values
    all_values = all_values[np.isfinite(all_values)]

    if len(all_values) == 0:
        logger.warning(
            f"No valid (non-NaN, non-inf) values found for feature '{feature_name}'"
        )
        return {}

    stats = {
        "count": len(all_values),
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "max": float(np.max(all_values)),
        "q25": float(np.percentile(all_values, 25)),
        "q50": float(np.percentile(all_values, 50)),
        "q75": float(np.percentile(all_values, 75)),
    }

    # Calculate skewness and kurtosis
    from scipy import stats as scipy_stats

    # Check if we have enough data points
    if len(all_values) < 3:
        logger.warning(
            f"Not enough data points ({len(all_values)}) to calculate skewness/kurtosis"
        )
        stats["skewness"] = 0.0
        stats["kurtosis"] = 0.0
    else:
        stats["skewness"] = float(scipy_stats.skew(all_values))
        stats["kurtosis"] = float(scipy_stats.kurtosis(all_values))

    # Determine if transformation is needed
    stats["needs_transform"] = (
        abs(stats["skewness"]) > 1.0 or abs(stats["kurtosis"]) > 3.0
    )
    stats["transform_reason"] = []

    if abs(stats["skewness"]) > 1.0:
        direction = "right" if stats["skewness"] > 0 else "left"
        stats["transform_reason"].append(
            f"High skewness ({stats['skewness']:.2f}, {direction}-skewed)"
        )

    if abs(stats["kurtosis"]) > 3.0:
        tail_type = "heavy" if stats["kurtosis"] > 0 else "light"
        stats["transform_reason"].append(
            f"Extreme kurtosis ({stats['kurtosis']:.2f}, {tail_type} tails)"
        )

    return stats


def _calculate_pair_correlations(
    features_df: pl.DataFrame,
    raw_data: sf.RawData,
    feature_name: str,
    pairs: List[str],
    price_col: str,
) -> Dict[str, Optional[float]]:
    """
    Calculate Pearson correlation between feature and price for each pair.

    Args:
        features_df: Feature data with columns [timestamp, pair, feature_name]
        raw_data: Raw OHLCV data
        feature_name: Name of feature being analyzed
        pairs: List of trading pairs
        price_col: Price column to use (e.g., 'close')

    Returns:
        Dict mapping pair to correlation coefficient or None if insufficient data
    """
    correlations = {}

    # Get price data once
    price_df = raw_data.get("spot")

    for pair in pairs:
        try:
            # Get feature data for this pair
            feat_pair = (
                features_df.filter(pl.col("pair") == pair)
                .select(["timestamp", feature_name])
                .drop_nulls()
                .sort("timestamp")
            )

            # Get price data for this pair
            price_pair = (
                price_df.filter(pl.col("pair") == pair)
                .select(["timestamp", price_col])
                .sort("timestamp")
            )

            # Inner join on timestamp to align data
            merged = feat_pair.join(price_pair, on="timestamp", how="inner")

            if merged.height < 2:
                correlations[pair] = None
                continue

            # Calculate Pearson correlation using NumPy
            feature_vals = merged[feature_name].to_numpy().astype(float)
            price_vals = merged[price_col].to_numpy().astype(float)

            # Filter out NaN/inf
            valid_mask = np.isfinite(feature_vals) & np.isfinite(price_vals)
            if valid_mask.sum() < 2:
                correlations[pair] = None
                continue

            corr_matrix = np.corrcoef(feature_vals[valid_mask], price_vals[valid_mask])
            correlations[pair] = float(corr_matrix[0, 1])

        except Exception as e:
            logger.warning(f"Failed to calculate correlation for {pair}: {e}")
            correlations[pair] = None

    return correlations


def _calculate_data_quality(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate data quality metrics for each pair.

    Args:
        features_df: Feature data with columns [timestamp, pair, feature_name]
        feature_name: Name of feature being analyzed
        pairs: List of trading pairs

    Returns:
        Dict mapping pair to quality metrics dict with keys:
        - total_rows: Total number of rows
        - nan_count: Number of NaN values
        - valid_count: Number of valid values
        - valid_pct: Percentage of valid values
        - time_start: First timestamp
        - time_end: Last timestamp
        - time_span: Time span duration
    """
    quality_metrics = {}

    for pair in pairs:
        try:
            pair_df = features_df.filter(pl.col("pair") == pair)

            # Total rows for this pair
            total_rows = pair_df.height

            # NaN count
            nan_count = pair_df[feature_name].null_count()

            # Valid data count
            valid_count = total_rows - nan_count

            # Valid percentage
            valid_pct = (valid_count / total_rows * 100) if total_rows > 0 else 0.0

            # Time range
            timestamps = pair_df.sort("timestamp")["timestamp"]
            if timestamps.height > 0:
                time_start = timestamps[0]
                time_end = timestamps[-1]
                time_span = time_end - time_start
            else:
                time_start = None
                time_end = None
                time_span = None

            quality_metrics[pair] = {
                "total_rows": total_rows,
                "nan_count": nan_count,
                "valid_count": valid_count,
                "valid_pct": valid_pct,
                "time_start": time_start,
                "time_end": time_end,
                "time_span": time_span,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate data quality for {pair}: {e}")
            quality_metrics[pair] = {
                "total_rows": 0,
                "nan_count": 0,
                "valid_count": 0,
                "valid_pct": 0.0,
                "time_start": None,
                "time_end": None,
                "time_span": None,
            }

    return quality_metrics


def _format_text_report(
    feature_name: str,
    pairs: List[str],
    stats: Dict[str, Any],
    correlations: Dict[str, Optional[float]],
    quality_metrics: Dict[str, Dict[str, Any]],
    indicator_type: Optional[str] = None,
) -> str:
    """
    Format all metrics into readable HTML text for Telegram.

    Args:
        feature_name: Name of feature being analyzed
        pairs: List of trading pairs
        stats: Global statistics from _calculate_feature_statistics()
        correlations: Per-pair correlations from _calculate_pair_correlations()
        quality_metrics: Per-pair quality metrics from _calculate_data_quality()
        indicator_type: Optional indicator type (e.g., "momentum/torque")

    Returns:
        HTML-formatted text report for Telegram
    """
    lines = []

    # Header
    lines.append("<b>📊 Feature Analysis Report</b>")
    lines.append(f"<b>Feature:</b> <code>{feature_name}</code>")
    if indicator_type:
        lines.append(f"<b>Type:</b> <code>{indicator_type}</code>")
    lines.append(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Section 1: Global Statistics
    lines.append("<b>═══ GLOBAL STATISTICS ═══</b>")
    lines.append(f"<b>Sample Count:</b> {stats.get('count', 0):,}")
    lines.append(f"<b>Mean:</b> {stats.get('mean', 0.0):.6f}")
    lines.append(f"<b>Std Dev:</b> {stats.get('std', 0.0):.6f}")
    lines.append(f"<b>Skewness:</b> {stats.get('skewness', 0.0):.4f}")
    lines.append(f"<b>Kurtosis:</b> {stats.get('kurtosis', 0.0):.4f}")
    lines.append("")
    lines.append("<b>Quartiles:</b>")
    lines.append(f"  Min:  {stats.get('min', 0.0):.6f}")
    lines.append(f"  Q25:  {stats.get('q25', 0.0):.6f}")
    lines.append(f"  Q50:  {stats.get('q50', 0.0):.6f}")
    lines.append(f"  Q75:  {stats.get('q75', 0.0):.6f}")
    lines.append(f"  Max:  {stats.get('max', 0.0):.6f}")
    lines.append("")

    # Section 2: Correlation with Price
    lines.append("<b>═══ PRICE CORRELATION ═══</b>")
    lines.append("<pre>")
    lines.append(f"{'Pair':<12} {'Correlation':>12}")
    lines.append("-" * 26)

    # Sort by correlation (absolute value)
    sorted_pairs = sorted(
        pairs,
        key=lambda p: (
            abs(correlations.get(p, 0.0)) if correlations.get(p) is not None else -1
        ),
        reverse=True,
    )

    for pair in sorted_pairs:
        corr = correlations.get(pair)
        if corr is None:
            corr_str = "N/A"
        else:
            corr_str = f"{corr:>+.4f}"
        lines.append(f"{pair:<12} {corr_str:>12}")

    lines.append("</pre>")
    lines.append("")

    # Section 3: Data Quality
    lines.append("<b>═══ DATA QUALITY & COVERAGE ═══</b>")
    lines.append("<pre>")
    lines.append(f"{'Pair':<12} {'Valid%':>8} {'NaNs':>8} {'Total':>8}")
    lines.append("-" * 38)

    for pair in pairs:
        qm = quality_metrics.get(pair, {})
        valid_pct = qm.get("valid_pct", 0.0)
        nan_count = qm.get("nan_count", 0)
        total_rows = qm.get("total_rows", 0)

        lines.append(f"{pair:<12} {valid_pct:>7.2f}% {nan_count:>8,} {total_rows:>8,}")

    lines.append("</pre>")
    lines.append("")

    # Section 4: Time Coverage
    lines.append("<b>Time Coverage:</b>")
    for pair in pairs:
        qm = quality_metrics.get(pair, {})
        time_start = qm.get("time_start")
        time_end = qm.get("time_end")

        if time_start and time_end:
            start_str = time_start.strftime("%Y-%m-%d %H:%M")
            end_str = time_end.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  <code>{pair}:</code> {start_str} → {end_str}")

    lines.append("")

    # Section 5: Transformation recommendation
    if stats.get("needs_transform"):
        lines.append("<b>⚠️ TRANSFORMATION RECOMMENDATION</b>")
        for reason in stats.get("transform_reason", []):
            lines.append(f"  • {reason}")
        lines.append("")

    report = "\n".join(lines)

    # Check Telegram message length limit
    if len(report) > 4000:
        logger.warning(
            f"Text report too long ({len(report)} chars), truncating to fit Telegram limit"
        )
        report = report[:3900] + "\n\n... (truncated)"

    return report


def _generate_text_report(
    features_df: pl.DataFrame,
    raw_data: sf.RawData,
    feature_name: str,
    pairs: List[str],
    price_col: str,
    stats: Dict[str, Any],
    indicator_type: Optional[str] = None,
) -> str:
    """
    Generate comprehensive text report for feature analysis.

    Orchestrates calculation of correlations, quality metrics, and formatting.

    Args:
        features_df: Feature data with columns [timestamp, pair, feature_name]
        raw_data: Raw OHLCV data
        feature_name: Name of feature being analyzed
        pairs: List of trading pairs
        price_col: Price column to use for correlation (e.g., 'close')
        stats: Pre-calculated statistics from _calculate_feature_statistics()
        indicator_type: Optional indicator type (e.g., "momentum/torque")

    Returns:
        HTML-formatted text report ready for Telegram
    """
    # Calculate per-pair correlations
    correlations = _calculate_pair_correlations(
        features_df, raw_data, feature_name, pairs, price_col
    )

    # Calculate data quality metrics
    quality_metrics = _calculate_data_quality(features_df, feature_name, pairs)

    # Format into text report
    report = _format_text_report(
        feature_name, pairs, stats, correlations, quality_metrics, indicator_type
    )

    return report


def _save_statistics_to_file(
    feature_name: str,
    indicator_type: Optional[str],
    stats: Dict[str, Any],
    correlations: Dict[str, Optional[float]],
    quality_metrics: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> None:
    """
    Save comprehensive statistics to JSON file for batch reporting.

    Args:
        feature_name: Name of feature being analyzed
        indicator_type: Indicator type (e.g., "momentum/rsi")
        stats: Global statistics
        correlations: Per-pair correlations
        quality_metrics: Per-pair quality metrics
        output_dir: Directory to save stats file
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats_file = output_path / "latest_stats.json"

        # Prepare data for JSON serialization
        def serialize_value(v):
            """Convert numpy/polars types to Python types."""
            if v is None:
                return None
            elif isinstance(v, (np.integer, np.floating)):
                val = float(v)
                # Handle NaN and inf
                if np.isnan(val):
                    return None
                elif np.isinf(val):
                    return None
                return val
            elif hasattr(v, "isoformat"):  # datetime
                return v.isoformat()
            return v

        # Calculate average correlation
        valid_corrs = [c for c in correlations.values() if c is not None]
        avg_corr = float(np.mean(valid_corrs)) if valid_corrs else None

        # Calculate average data quality
        total_valid = sum(qm.get("valid_count", 0) for qm in quality_metrics.values())
        total_rows = sum(qm.get("total_rows", 0) for qm in quality_metrics.values())
        avg_valid_pct = (total_valid / total_rows * 100) if total_rows > 0 else 0.0

        # Build comprehensive stats dict
        comprehensive_stats = {
            "feature_name": feature_name,
            "indicator_type": indicator_type,
            "timestamp": datetime.now().isoformat(),
            "global_stats": {
                k: serialize_value(v)
                for k, v in stats.items()
                if k not in ["transform_reason"]  # Skip list field for now
            },
            "transform_needed": stats.get("needs_transform", False),
            "transform_reasons": stats.get("transform_reason", []),
            "correlations": {
                pair: serialize_value(corr) for pair, corr in correlations.items()
            },
            "avg_correlation": serialize_value(avg_corr),
            "data_quality": {
                "avg_valid_pct": serialize_value(avg_valid_pct),
                "per_pair": {
                    pair: {
                        "valid_pct": serialize_value(qm.get("valid_pct", 0)),
                        "nan_count": serialize_value(qm.get("nan_count", 0)),
                        "total_rows": serialize_value(qm.get("total_rows", 0)),
                    }
                    for pair, qm in quality_metrics.items()
                },
            },
        }

        # Write to file (use allow_nan=False to ensure valid JSON)
        # Replace NaN with null before writing
        import math

        def replace_nan(obj):
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(item) for item in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        comprehensive_stats = replace_nan(comprehensive_stats)

        with open(stats_file, "w") as f:
            json.dump(comprehensive_stats, f, indent=2)

        logger.info(f"Statistics saved to: {stats_file}")

    except Exception as e:
        logger.error(f"Failed to save statistics to file: {e}")


def _plot_feature_across_pairs(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
) -> go.Figure:
    """Plot 1: Feature values over time, one line per pair."""
    fig = go.Figure()

    for i, pair in enumerate(pairs):
        pair_df = (
            features_df.filter(pl.col("pair") == pair)
            .sort("timestamp")
            .drop_nulls(subset=[feature_name])
        )
        if pair_df.height == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=pair_df["timestamp"].to_list(),
                y=pair_df[feature_name].to_list(),
                mode="lines",
                name=pair,
                line=dict(color=PAIR_COLORS[i % len(PAIR_COLORS)], width=1.5),
                hovertemplate=(
                    f"<b>{pair}</b><br>"
                    "Time: %{x}<br>"
                    f"{feature_name}: %{{y:.2f}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>{feature_name.upper()} across pairs</b>",
            font=dict(color="#333333", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Time",
        yaxis_title=feature_name.upper(),
    )
    return fig


def _plot_feature_vs_price(
    raw_data: sf.RawData,
    features_df: pl.DataFrame,
    feature_name: str,
    price_pair: str,
    price_col: str,
) -> go.Figure:
    """Plot 2: Feature vs close price for a single pair (dual y-axis)."""
    feat_pair = (
        features_df.filter(pl.col("pair") == price_pair)
        .sort("timestamp")
        .drop_nulls(subset=[feature_name])
    )

    spot_df = raw_data.get("spot")
    price_df = (
        spot_df.filter(pl.col("pair") == price_pair)
        .sort("timestamp")
        .select(["timestamp", price_col])
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=price_df["timestamp"].to_list(),
            y=price_df[price_col].to_list(),
            mode="lines",
            name=f"{price_pair} Price",
            line=dict(color="#2171b5", width=1.5),
            hovertemplate=(
                f"<b>{price_pair} Price</b><br>"
                "Time: %{x}<br>"
                "Price: %{y:,.2f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=feat_pair["timestamp"].to_list(),
            y=feat_pair[feature_name].to_list(),
            mode="lines",
            name=feature_name.upper(),
            line=dict(color="#e6550d", width=1.5),
            hovertemplate=(
                f"<b>{feature_name.upper()}</b><br>"
                "Time: %{x}<br>"
                "Value: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>{feature_name.upper()} vs Price ({price_pair})</b>",
            font=dict(color="#333333", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Time",
    )
    fig.update_yaxes(title_text=f"Price ({price_col})", secondary_y=False)
    fig.update_yaxes(title_text=feature_name.upper(), secondary_y=True)

    return fig


def _plot_feature_distribution(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
    n_bins: int,
) -> go.Figure:
    """Plot 3: Histogram of feature values."""
    fig = go.Figure()

    for i, pair in enumerate(pairs):
        vals = (
            features_df.filter(pl.col("pair") == pair)
            .drop_nulls(subset=[feature_name])[feature_name]
            .to_numpy()
        )
        if len(vals) == 0:
            continue

        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=n_bins,
                name=pair,
                marker_color=PAIR_COLORS[i % len(PAIR_COLORS)],
                opacity=0.6,
                hovertemplate=(
                    f"<b>{pair}</b><br>"
                    f"{feature_name}: %{{x:.2f}}<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(),
        barmode="overlay",
        title=dict(
            text=f"<b>{feature_name.upper()} Distribution</b>",
            font=dict(color="#333333", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=feature_name.upper(),
        yaxis_title="Count",
    )
    return fig


def _plot_feature_normalized(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
) -> go.Figure:
    """Plot 4: Z-score normalized feature across pairs."""
    fig = go.Figure()

    for i, pair in enumerate(pairs):
        pair_df = (
            features_df.filter(pl.col("pair") == pair)
            .sort("timestamp")
            .drop_nulls(subset=[feature_name])
        )
        if pair_df.height == 0:
            continue

        values = pair_df[feature_name].to_numpy().astype(float)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - mean) / std

        fig.add_trace(
            go.Scatter(
                x=pair_df["timestamp"].to_list(),
                y=normalized.tolist(),
                mode="lines",
                name=pair,
                line=dict(color=PAIR_COLORS[i % len(PAIR_COLORS)], width=1.5),
                hovertemplate=(
                    f"<b>{pair}</b><br>"
                    "Time: %{x}<br>"
                    "Z-score: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>{feature_name.upper()} Normalized (Z-score)</b>",
            font=dict(color="#333333", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Time",
        yaxis_title="Z-score",
    )
    return fig


def _base_layout() -> Dict[str, Any]:
    """Shared layout settings consistent with existing project plots."""
    return dict(
        height=900,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#ffffff",
    )


def _send_to_telegram(
    plots: Dict[str, List[go.Figure]],
    telegram_config: Dict[str, Any],
    feature_name: str,
    pairs: List[str],
    indicator_type: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    text_report: Optional[str] = None,
) -> None:
    """Send all plots as one Telegram media group with hashtags and statistics, followed by optional text report."""
    try:
        notifier = TelegramNotifier(
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )

        # Generate hashtags from indicator type (e.g., "momentum/rsi" -> "#momentum #rsi")
        hashtags = ""
        if indicator_type:
            tags = indicator_type.split("/")
            hashtags = " ".join(f"#{tag}" for tag in tags)

        # Build caption with statistics
        caption_parts = [f"<b>Feature Analysis: {feature_name.upper()}</b>"]

        if indicator_type:
            caption_parts.append(f"Indicator: <code>{indicator_type}</code>")

        caption_parts.append(f"Pairs: {', '.join(pairs)}")

        caption = "\n".join(caption_parts) + "\n"

        # Add statistics if available
        if stats:
            caption += (
                f"\n<b>Statistics:</b>\n"
                f"Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f}\n"
                f"Skew: {stats['skewness']:.3f} | Kurt: {stats['kurtosis']:.3f}\n"
            )
            if stats.get("needs_transform"):
                caption += (
                    f"⚠️ Needs transform: {', '.join(stats['transform_reason'])}\n"
                )

        caption += (
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{hashtags}"
        )

        all_figures = plots.get("feature_analysis", [])

        # Check if we have any figures to send
        if not all_figures or len(all_figures) == 0:
            logger.warning("No plots to send to Telegram (empty plots list)")
            return

        notifier.send_plots_group(
            figures=all_figures,
            metric_name=caption,
            width=telegram_config.get("image_width", 1400),
            height=telegram_config.get("image_height", 900),
        )
        logger.info("Feature analysis plots sent to Telegram")

        # Send detailed text report separately
        if text_report:
            notifier.send_message(text_report)
            logger.info("Feature analysis text report sent to Telegram")

    except Exception as e:
        logger.error(f"Failed to send feature analysis to Telegram: {e}")
        if telegram_config.get("raise_on_error", False):
            raise
