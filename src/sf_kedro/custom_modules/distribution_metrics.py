### TODO: fix this logic
from typing import Dict, Any, Tuple
from dataclasses import dataclass

import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from .signal_metrics import SignalMetricsProcessor
import signalflow as sf



@dataclass
@sf.sf_component(name="distribution")
class SignalDistributionMetrics(SignalMetricsProcessor):
    """Analyze signal distribution across pairs and time.
    
    Provides insights into:
    - How signals are distributed across different trading pairs
    - Signal frequency patterns over time
    - Concentration of trading activity
    """
    
    n_bars: int = 1  # Number of bins for histogram
    rolling_window_minutes: int = 60  # Window for rolling signal count
    ma_window_hours: int = 12  # Moving average window in hours
    chart_height: int = 1200
    chart_width: int = 1400
    
    def compute(
        self,
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute signal distribution metrics."""
        
        signals_df = signals.value
        
        # Count signals per pair (excluding neutral signals)
        signals_per_pair = (
            signals_df
            .filter(pl.col("signal") != 0)
            .group_by("pair")
            .agg(pl.count().alias("signal_count"))
            .sort("signal_count", descending=True)
        )
        
        if signals_per_pair.height == 0:
            logger.warning("No non-zero signals found")
            return None, {}
        
        # Get statistics
        signal_counts = signals_per_pair["signal_count"].to_numpy()
        min_count = signal_counts.min()
        max_count = signal_counts.max()
        mean_count = signal_counts.mean()
        median_count = np.median(signal_counts)
        
        # Create histogram bins
        if min_count == max_count:
            bin_edges = [min_count - 0.5, max_count + 0.5]
            bin_labels = [f"{int(min_count)}"]
        else:
            bin_edges = np.linspace(min_count, max_count, self.n_bars + 1)
            bin_labels = []
            for i in range(self.n_bars):
                lower = int(np.floor(bin_edges[i]))
                upper = int(np.ceil(bin_edges[i + 1]))
                
                if lower == upper:
                    label = f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}"
                else:
                    label = f"{lower}–{upper}"
                
                bin_labels.append(label)
        
        # Bin the data
        binned = np.digitize(signal_counts, bin_edges[:-1]) - 1
        binned = np.clip(binned, 0, len(bin_labels) - 1)
        
        # Group pairs by bins
        grouped_data = []
        for i, label in enumerate(bin_labels):
            mask = binned == i
            pairs_in_bin = signals_per_pair.filter(
                pl.Series(mask)
            )["pair"].to_list()
            
            if pairs_in_bin:
                grouped_data.append({
                    "category": label,
                    "num_columns": len(pairs_in_bin),
                    "columns_in_group": "<br>".join(pairs_in_bin),
                })
        
        # Compute rolling signal count over time
        # First, create time series with 1-minute granularity
        signals_by_time = (
            signals_df
            .filter(pl.col("signal") != 0)
            .group_by_dynamic("timestamp", every="1m")
            .agg(pl.count().alias("signal_count"))
            .sort("timestamp")
        )
        
        # Rolling sum using integer window (number of rows)
        # Since we have 1-minute granularity, window = rolling_window_minutes
        signals_rolling = (
            signals_by_time
            .with_columns(
                pl.col("signal_count")
                .rolling_sum(
                    window_size=self.rolling_window_minutes,
                    min_periods=1,
                    center=False,
                )
                .alias("rolling_sum")
            )
        )
        
        # Moving average for smoothing
        ma_window_minutes = self.ma_window_hours * 60
        if signals_rolling.height > ma_window_minutes:
            signals_rolling = signals_rolling.with_columns(
                pl.col("rolling_sum")
                .rolling_mean(
                    window_size=ma_window_minutes,
                    min_periods=1,
                    center=True,
                )
                .alias("ma")
            )
        else:
            signals_rolling = signals_rolling.with_columns(
                pl.lit(None).alias("ma")
            )
        
        mean_rolling = signals_rolling["rolling_sum"].mean()
        max_rolling = signals_rolling["rolling_sum"].max()
        
        computed_metrics = {
            "quant": {
                "mean_signals_per_pair": float(mean_count),
                "median_signals_per_pair": float(median_count),
                "min_signals_per_pair": int(min_count),
                "max_signals_per_pair": int(max_count),
                "total_pairs": signals_per_pair.height,
                "mean_rolling_signals": float(mean_rolling) if mean_rolling else 0.0,
                "max_rolling_signals": int(max_rolling) if max_rolling else 0,
            },
            "series": {
                "grouped": grouped_data,
                "signals_per_pair": signals_per_pair,
                "signals_rolling": signals_rolling,
            },
        }
        
        plots_context = {
            "bin_labels": bin_labels,
            "rolling_window": self.rolling_window_minutes,
            "ma_window": self.ma_window_hours,
        }
        
        logger.info(
            f"Distribution computed: {signals_per_pair.height} pairs, "
            f"mean {mean_count:.1f} signals/pair, "
            f"max rolling {max_rolling} signals/{self.rolling_window_minutes}min"
        )
        
        return computed_metrics, plots_context
    
    def plot(
        self,
        computed_metrics: Dict[str, Any],
        plots_context: Dict[str, Any],
        raw_data: sf.RawData,
        signals: sf.Signals,
        labels: pl.DataFrame | None = None,
    ) -> go.Figure:
        """Generate distribution visualization."""
        
        if computed_metrics is None:
            logger.error("No metrics available for plotting")
            return None
        
        fig = self._create_figure()
        
        self._add_histogram(fig, computed_metrics)
        self._add_sorted_signals(fig, computed_metrics)
        self._add_rolling_signals(fig, computed_metrics, plots_context)
        self._add_summary_annotation(fig, computed_metrics, plots_context)
        self._update_layout(fig, plots_context)
        
        return fig
    
    @staticmethod
    def _create_figure():
        """Create subplot structure."""
        return make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            row_heights=[0.3, 0.35, 0.35],
            vertical_spacing=0.08,
            subplot_titles=[
                "Pairs Distribution by Signal Count",
                "Signal Count per Pair (Sorted)",
                "Temporal Signal Density",
            ],
        )
    
    @staticmethod
    def _add_histogram(fig, metrics):
        """Add histogram of signal distribution."""
        grouped = metrics["series"]["grouped"]
        
        if not grouped:
            return
        
        categories = [g["category"] for g in grouped]
        counts = [g["num_columns"] for g in grouped]
        hovertexts = [g["columns_in_group"] for g in grouped]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=counts,
                customdata=[[ht] for ht in hovertexts],
                marker=dict(
                    color=counts,
                    colorscale="Viridis",
                    showscale=False,
                    line=dict(color="black", width=1),
                ),
                hovertemplate=(
                    "<b>Signal Range:</b> %{x}<br>"
                    "<b>Number of Pairs:</b> %{y}<br>"
                    "<b>Pairs:</b><br>%{customdata[0]}"
                    "<extra></extra>"
                ),
                name="Pair Count",
            ),
            row=1,
            col=1,
        )
    
    @staticmethod
    def _add_sorted_signals(fig, metrics):
        """Add sorted signal counts per pair."""
        signals_per_pair = metrics["series"]["signals_per_pair"]
        
        pairs = signals_per_pair["pair"].to_list()
        counts = signals_per_pair["signal_count"].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pairs))),
                y=counts,
                mode="lines+markers",
                line=dict(color="indianred", width=2),
                marker=dict(size=5, color="darkred"),
                text=pairs,
                hovertemplate=(
                    "<b>Rank:</b> %{x}<br>"
                    "<b>Pair:</b> %{text}<br>"
                    "<b>Signals:</b> %{y}"
                    "<extra></extra>"
                ),
                name="Signal Count",
            ),
            row=2,
            col=1,
        )
        
        # Add mean line
        mean_count = metrics["quant"]["mean_signals_per_pair"]
        fig.add_hline(
            y=mean_count,
            line=dict(color="green", dash="dash", width=2),
            annotation_text=f"Mean: {mean_count:.1f}",
            annotation_position="right",
            row=2,
            col=1,
        )
    
    @staticmethod
    def _add_rolling_signals(fig, metrics, plots_context):
        """Add rolling signal count over time."""
        signals_rolling = metrics["series"]["signals_rolling"]
        
        if signals_rolling.height == 0:
            return
        
        timestamps = signals_rolling["timestamp"].to_list()
        rolling_sum = signals_rolling["rolling_sum"].to_list()
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_sum,
                mode="lines",
                line=dict(color="royalblue", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(65, 105, 225, 0.15)",
                hovertemplate=(
                    "<b>Time:</b> %{x}<br>"
                    f"<b>{plots_context['rolling_window']}min Signals:</b> %{{y:.0f}}"
                    "<extra></extra>"
                ),
                name=f"{plots_context['rolling_window']}min Rolling",
            ),
            row=3,
            col=1,
        )
        
        # Add moving average if available
        if "ma" in signals_rolling.columns:
            ma_values = signals_rolling["ma"].to_list()
            if any(v is not None for v in ma_values):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=ma_values,
                        mode="lines",
                        line=dict(color="darkblue", width=2.5, dash="dash"),
                        hovertemplate=(
                            "<b>Time:</b> %{x}<br>"
                            f"<b>{plots_context['ma_window']}h MA:</b> %{{y:.1f}}"
                            "<extra></extra>"
                        ),
                        name=f"{plots_context['ma_window']}h MA",
                    ),
                    row=3,
                    col=1,
                )
    
    @staticmethod
    def _add_summary_annotation(fig, metrics, plots_context):
        """Add summary statistics annotation."""
        quant = metrics["quant"]
        
        summary_text = (
            f"<b>Distribution Statistics</b><br>"
            f"Total Pairs: {quant['total_pairs']}<br>"
            f"Mean Signals/Pair: {quant['mean_signals_per_pair']:.1f}<br>"
            f"Median Signals/Pair: {quant['median_signals_per_pair']:.1f}<br>"
            f"Range: [{quant['min_signals_per_pair']}, {quant['max_signals_per_pair']}]<br>"
            f"<br>"
            f"<b>Temporal Density</b><br>"
            f"Mean ({plots_context['rolling_window']}min): {quant['mean_rolling_signals']:.1f}<br>"
            f"Max ({plots_context['rolling_window']}min): {quant['max_rolling_signals']}"
        )
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=summary_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=8,
            bgcolor="white",
            opacity=0.9,
            align="left",
            font=dict(size=11),
        )
    
    def _update_layout(self, fig, plots_context):
        """Update figure layout and axes."""
        fig.update_xaxes(
            title_text="Signal Count Range",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Number of Pairs",
            row=1,
            col=1,
        )
        
        fig.update_xaxes(
            title_text="Pair Rank (Sorted by Signal Count)",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Signal Count",
            row=2,
            col=1,
        )
        
        fig.update_xaxes(
            title_text="Time",
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title_text=f"Signals ({plots_context['rolling_window']}min Rolling)",
            row=3,
            col=1,
        )
        
        fig.update_layout(
            title=dict(
                text=(
                    "SignalFlow: Signal Distribution Analysis<br>"
                    f"<sub>Temporal window: {plots_context['rolling_window']}min rolling, "
                    f"{plots_context['ma_window']}h moving average</sub>"
                ),
                font=dict(color="black", size=16),
                x=0.5,
                xanchor="center",
            ),
            height=self.chart_height,
            width=self.chart_width,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )