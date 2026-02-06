"""Feature analysis pipeline nodes."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

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
    "#2171b5", "#e6550d", "#31a354", "#756bb1", "#d62728",
    "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b",
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
        analysis_params: Dict with feature_name, price_pair, price_col, n_bins
        telegram_config: Optional Telegram config

    Returns:
        Dict with {"feature_analysis": [fig1, fig2, fig3, fig4]}
    """
    feature_name = analysis_params["feature_name"]
    price_pair = analysis_params.get("price_pair", "BTCUSDT")
    price_col = analysis_params.get("price_col", "close")
    n_bins = analysis_params.get("n_bins", 50)

    pairs = sorted(features_df["pair"].unique().to_list())
    logger.info(
        f"Building feature analysis plots: feature={feature_name}, "
        f"price_pair={price_pair}, pairs={pairs}"
    )

    fig1 = _plot_feature_across_pairs(features_df, feature_name, pairs)
    fig2 = _plot_feature_vs_price(raw_data, features_df, feature_name, price_pair, price_col)
    fig3 = _plot_feature_distribution(features_df, feature_name, pairs, n_bins)
    fig4 = _plot_feature_normalized(features_df, feature_name, pairs)

    plots = {"feature_analysis": [fig1, fig2, fig3, fig4]}

    if telegram_config and telegram_config.get("enabled", False):
        _send_to_telegram(plots, telegram_config, feature_name, pairs)

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
        if figures is None:
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

def _plot_feature_across_pairs(
    features_df: pl.DataFrame,
    feature_name: str,
    pairs: List[str],
) -> go.Figure:
    """Plot 1: Feature values over time, one line per pair."""
    fig = go.Figure()

    for i, pair in enumerate(pairs):
        pair_df = (
            features_df
            .filter(pl.col("pair") == pair)
            .sort("timestamp")
            .drop_nulls(subset=[feature_name])
        )
        if pair_df.height == 0:
            continue

        fig.add_trace(go.Scatter(
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
        ))

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>{feature_name.upper()} across pairs</b>",
            font=dict(color="#333333", size=18),
            x=0.5, xanchor="center",
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
        features_df
        .filter(pl.col("pair") == price_pair)
        .sort("timestamp")
        .drop_nulls(subset=[feature_name])
    )

    spot_df = raw_data.get("spot")
    price_df = (
        spot_df
        .filter(pl.col("pair") == price_pair)
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
            x=0.5, xanchor="center",
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
            features_df
            .filter(pl.col("pair") == pair)
            .drop_nulls(subset=[feature_name])
            [feature_name]
            .to_numpy()
        )
        if len(vals) == 0:
            continue

        fig.add_trace(go.Histogram(
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
        ))

    fig.update_layout(
        **_base_layout(),
        barmode="overlay",
        title=dict(
            text=f"<b>{feature_name.upper()} Distribution</b>",
            font=dict(color="#333333", size=18),
            x=0.5, xanchor="center",
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
            features_df
            .filter(pl.col("pair") == pair)
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

        fig.add_trace(go.Scatter(
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
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))

    fig.update_layout(
        **_base_layout(),
        title=dict(
            text=f"<b>{feature_name.upper()} Normalized (Z-score)</b>",
            font=dict(color="#333333", size=18),
            x=0.5, xanchor="center",
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
) -> None:
    """Send all plots as one Telegram media group."""
    try:
        notifier = TelegramNotifier(
            bot_token=telegram_config.get("bot_token"),
            chat_id=telegram_config.get("chat_id"),
        )

        header = (
            f"<b>Feature Analysis: {feature_name.upper()}</b>\n"
            f"Pairs: {', '.join(pairs)}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        notifier.send_message(header)

        all_figures = plots["feature_analysis"]
        notifier.send_plots_group(
            figures=all_figures,
            metric_name=f"Feature: {feature_name.upper()}",
            width=telegram_config.get("image_width", 1400),
            height=telegram_config.get("image_height", 900),
        )
        logger.info("Feature analysis plots sent to Telegram")
    except Exception as e:
        logger.error(f"Failed to send feature analysis to Telegram: {e}")
        if telegram_config.get("raise_on_error", False):
            raise
