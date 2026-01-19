from typing import Dict, Any, Optional, List, Tuple
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SignalMetricsConfig:
    """Configuration for signal metrics computation"""
    look_ahead: int = 1440  # minutes
    quantiles: Tuple[float, float] = (0.25, 0.75)
    pairs_to_plot: Optional[List[str]] = None
    min_signals_threshold: int = 10  # Minimum signals to compute metrics
    

class SignalMetrics:
    """Compute and visualize signal performance metrics"""
    
    def __init__(self, config: SignalMetricsConfig):
        self.config = config
        
    def compute_all_metrics(
        self, 
        raw_df: pl.DataFrame,
        signals_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute all metrics for the signals
        
        Args:
            raw_df: Raw OHLCV data with columns [timestamp, pair, close, ...]
            signals_df: Signals with columns [timestamp, pair, signal]
            
        Returns:
            Dict with 'scalar_metrics' and 'figures' keys
        """
        # Validate inputs
        self._validate_inputs(raw_df, signals_df)
        
        # Scalar metrics for MLflow
        profile_metrics = self.compute_profile_metrics(raw_df, signals_df)
        scalar_metrics = profile_metrics.get("quant", {})
        
        # Additional signal statistics
        signal_stats = self.compute_signal_statistics(signals_df)
        scalar_metrics.update(signal_stats)
        
        # Generate figures
        figures = {}
        
        # Profile plot
        if profile_metrics.get("quant") and profile_metrics["quant"].get("n_signals", 0) > 0:
            profile_fig = self.plot_profile_metrics(profile_metrics)
            if profile_fig:
                figures["profile"] = profile_fig
        else:
            logger.warning("No signals found or insufficient data for profile metrics")
            
        # Price + signals plots for specified pairs
        if self.config.pairs_to_plot:
            for pair in self.config.pairs_to_plot:
                try:
                    fig = self.plot_signals_on_price(raw_df, signals_df, pair)
                    if fig:
                        figures[f"signals_{pair}"] = fig
                except Exception as e:
                    logger.warning(f"Failed to create plot for {pair}: {e}")
        
        return {
            "scalar_metrics": scalar_metrics,
            "figures": figures
        }
    
    def _validate_inputs(self, raw_df: pl.DataFrame, signals_df: pl.DataFrame):
        """Validate input dataframes"""
        # Check required columns
        required_raw_cols = {"timestamp", "pair", "close"}
        required_sig_cols = {"timestamp", "pair", "signal"}
        
        if not required_raw_cols.issubset(raw_df.columns):
            missing = required_raw_cols - set(raw_df.columns)
            raise ValueError(f"raw_df missing columns: {missing}")
            
        if not required_sig_cols.issubset(signals_df.columns):
            missing = required_sig_cols - set(signals_df.columns)
            raise ValueError(f"signals_df missing columns: {missing}")
        
        # Ensure timestamp columns are datetime
        if raw_df["timestamp"].dtype != pl.Datetime:
            logger.warning("Converting raw_df timestamp to datetime")
            
        if signals_df["timestamp"].dtype != pl.Datetime:
            logger.warning("Converting signals_df timestamp to datetime")
    
    def compute_signal_statistics(self, signals_df: pl.DataFrame) -> Dict[str, float]:
        """Compute basic signal statistics"""
        total_signals = signals_df.filter(pl.col("signal") != 0).height
        buy_signals = signals_df.filter(pl.col("signal") == 1).height
        sell_signals = signals_df.filter(pl.col("signal") == -1).height
        
        unique_pairs = signals_df.filter(pl.col("signal") != 0)["pair"].n_unique()
        
        return {
            "total_signals": float(total_signals),
            "buy_signals": float(buy_signals),
            "sell_signals": float(sell_signals),
            "unique_pairs_with_signals": float(unique_pairs),
            "avg_signals_per_pair": float(total_signals / unique_pairs if unique_pairs > 0 else 0)
        }
    
    def compute_profile_metrics(
        self, 
        raw_df: pl.DataFrame,
        signals_df: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute post-signal price change profiles using vectorized operations
        
        This is MUCH faster than iterating through rows
        """
        look_ahead = self.config.look_ahead
        
        # Filter for buy signals only (signal == 1)
        buy_signals = signals_df.filter(pl.col("signal") == 1)
        
        if buy_signals.height == 0:
            logger.warning("No buy signals found")
            return {"quant": {}, "series": {}}
        
        logger.info(f"Processing {buy_signals.height} buy signals")
        
        # Prepare price data - sort and add row number per pair
        price_data = (
            raw_df
            .select(["timestamp", "pair", "close"])
            .sort(["pair", "timestamp"])
            .with_columns([
                pl.col("close").shift(-i).over("pair").alias(f"close_t{i}")
                for i in range(1, look_ahead + 1)
            ])
        )
        
        # Join signals with price data
        signal_prices = buy_signals.join(
            price_data,
            on=["timestamp", "pair"],
            how="inner"
        )
        
        if signal_prices.height == 0:
            logger.warning("No matching timestamps between signals and prices")
            return {"quant": {}, "series": {}}
        
        # Filter out signals without enough future data
        # Check if we have all future prices
        future_cols = [f"close_t{i}" for i in range(1, look_ahead + 1)]
        signal_prices = signal_prices.filter(
            pl.all_horizontal([pl.col(c).is_not_null() for c in future_cols])
        )
        
        if signal_prices.height < self.config.min_signals_threshold:
            logger.warning(
                f"Only {signal_prices.height} valid signals found "
                f"(threshold: {self.config.min_signals_threshold})"
            )
            return {"quant": {}, "series": {}}
        
        logger.info(f"Computing metrics for {signal_prices.height} valid signals")
        
        # Calculate relative changes vectorized
        signal_price = signal_prices["close"].to_numpy().reshape(-1, 1)
        
        # Stack all future prices
        future_prices = np.column_stack([
            signal_prices[col].to_numpy() 
            for col in future_cols
        ])
        
        # Include t=0 (signal price itself, relative change = 0)
        future_prices_with_t0 = np.column_stack([
            signal_price.flatten(),
            future_prices
        ])
        
        # Calculate relative changes
        changes_array = (future_prices_with_t0 / signal_price) - 1.0
        
        # Compute statistics
        mean_profile = changes_array.mean(axis=0)
        std_profile = changes_array.std(axis=0)
        median_profile = np.median(changes_array, axis=0)
        
        lower_quant = np.quantile(changes_array, self.config.quantiles[0], axis=0)
        upper_quant = np.quantile(changes_array, self.config.quantiles[1], axis=0)
        
        # Cumulative metrics
        cummax_array = np.maximum.accumulate(changes_array, axis=1)
        cummax_mean = cummax_array.mean(axis=0)
        cummax_median = np.median(cummax_array, axis=0)
        cummax_lower = np.quantile(cummax_array, self.config.quantiles[0], axis=0)
        cummax_upper = np.quantile(cummax_array, self.config.quantiles[1], axis=0)
        
        cummin_array = np.minimum.accumulate(changes_array, axis=1)
        cummin_mean = cummin_array.mean(axis=0)
        cummin_median = np.median(cummin_array, axis=0)
        cummin_lower = np.quantile(cummin_array, self.config.quantiles[0], axis=0)
        cummin_upper = np.quantile(cummin_array, self.config.quantiles[1], axis=0)
        
        # Scalar metrics
        n_signals = changes_array.shape[0]
        max_uplifts = cummax_array[:, -1]  # Max at end of period
        
        final_mean = float(mean_profile[-1] * 100)
        final_median = float(median_profile[-1] * 100)
        avg_max_uplift = float(np.mean(max_uplifts) * 100)
        median_max_uplift = float(np.median(max_uplifts) * 100)
        max_mean_val = float(mean_profile.max())
        max_mean_idx = int(mean_profile.argmax())
        max_mean_pct = float(max_mean_val * 100)
        
        # Percentage of signals reaching 5% profit
        pct_reaching_5 = float((cummax_array.max(axis=1) >= 0.05).mean() * 100)
        
        # Average time to reach max
        time_to_max = np.argmax(cummax_array, axis=1)
        avg_time_to_max = float(time_to_max.mean())
        median_time_to_max = float(np.median(time_to_max))
        
        return {
            "quant": {
                "n_signals": n_signals,
                "final_mean": final_mean,
                "final_median": final_median,
                "avg_max_uplift": avg_max_uplift,
                "median_max_uplift": median_max_uplift,
                "max_mean_val": max_mean_val,
                "max_mean_idx": max_mean_idx,
                "max_mean_pct": max_mean_pct,
                "pct_reaching_5_percent": pct_reaching_5,
                "avg_time_to_max_minutes": avg_time_to_max,
                "median_time_to_max_minutes": median_time_to_max,
            },
            "series": {
                "mean_profile": mean_profile,
                "std_profile": std_profile,
                "median_profile": median_profile,
                "lower_quant": lower_quant,
                "upper_quant": upper_quant,
                "cummax_mean": cummax_mean,
                "cummax_median": cummax_median,
                "cummax_lower": cummax_lower,
                "cummax_upper": cummax_upper,
                "cummin_mean": cummin_mean,
                "cummin_median": cummin_median,
                "cummin_lower": cummin_lower,
                "cummin_upper": cummin_upper,
            }
        }
    
    def plot_profile_metrics(self, metrics: Dict[str, Any]) -> Optional[go.Figure]:
        """Plot the profile metrics"""
        if not metrics.get("quant") or not metrics.get("series"):
            return None
        
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Average Post-Signal Price Change Profile",
                "Cumulative Maximum Profile",
            ),
        )
        
        look_ahead = self.config.look_ahead
        x_axis = np.arange(look_ahead + 1)
        
        # Row 1: Average profile
        mean = metrics["series"]["mean_profile"]
        std = metrics["series"]["std_profile"]
        median = metrics["series"]["median_profile"]
        lower_q = metrics["series"]["lower_quant"]
        upper_q = metrics["series"]["upper_quant"]
        
        # Std bands (add first for proper fill)
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=mean + std, 
                mode="lines", 
                name="+1 STD",
                line=dict(color="lightblue", dash="dash"),
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=mean - std, 
                mode="lines", 
                name="-1 STD",
                line=dict(color="lightblue", dash="dash"),
                fill="tonexty", 
                fillcolor="rgba(173,216,230,0.2)",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=mean, 
                mode="lines", 
                name="Mean",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        
        # Median
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=median, 
                mode="lines", 
                name="Median",
                line=dict(color="red", dash="dot", width=2)
            ),
            row=1, col=1
        )
        
        # Percentiles
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=upper_q, 
                mode="lines", 
                name="75th %ile",
                line=dict(color="green", dash="dash")
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=lower_q, 
                mode="lines", 
                name="25th %ile",
                line=dict(color="green", dash="dash"),
                fill="tonexty", 
                fillcolor="rgba(0,128,0,0.1)"
            ),
            row=1, col=1
        )
        
        # Max mean point
        max_idx = metrics["quant"]["max_mean_idx"]
        max_val = metrics["quant"]["max_mean_val"]
        max_pct = metrics["quant"]["max_mean_pct"]
        fig.add_trace(
            go.Scatter(
                x=[max_idx], y=[max_val], 
                mode="markers+text",
                name="Max Mean", 
                text=[f"{max_pct:.2f}%"],
                textposition="top center",
                marker=dict(color="purple", size=10, symbol="star")
            ),
            row=1, col=1
        )
        
        # Key time markers
        key_minutes = [60, 120, 360, 720, 1440]
        for km in key_minutes:
            if km <= look_ahead:
                fig.add_vline(
                    x=km, 
                    line_dash="dot", 
                    line_color="gray",
                    opacity=0.5,
                    row=1, col=1
                )
                # Add label
                fig.add_annotation(
                    x=km, 
                    y=mean.max(),
                    text=f"{km}m",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9, color="gray"),
                    row=1, col=1
                )
        
        # Row 2: Cumulative max/min
        cummax_mean = metrics["series"]["cummax_mean"]
        cummax_median = metrics["series"]["cummax_median"]
        cummax_lower = metrics["series"]["cummax_lower"]
        cummax_upper = metrics["series"]["cummax_upper"]
        cummin_mean = metrics["series"]["cummin_mean"]
        cummin_median = metrics["series"]["cummin_median"]
        
        # Cummax percentile band
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=cummax_upper,
                mode="lines",
                name="CumMax 75th %ile",
                line=dict(color="lightcoral", dash="dash"),
                showlegend=True
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=cummax_lower,
                mode="lines",
                name="CumMax 25th %ile",
                line=dict(color="lightcoral", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(255,182,193,0.2)",
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Cummax mean/median
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=cummax_mean, 
                mode="lines",
                name="CumMax Mean", 
                line=dict(color="darkblue", width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=cummax_median, 
                mode="lines",
                name="CumMax Median", 
                line=dict(color="darkred", dash="dot", width=2)
            ),
            row=2, col=1
        )
        
        # Cummin for drawdown
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=cummin_mean, 
                mode="lines",
                name="CumMin Mean (Drawdown)", 
                line=dict(color="darkgreen", width=1)
            ),
            row=2, col=1
        )
        
        # Profit targets
        for target in [0.05, 0.10]:
            fig.add_hline(
                y=target, 
                line_dash="dot", 
                line_color="green",
                opacity=0.5,
                annotation_text=f"{target*100:.0f}%",
                annotation_position="right",
                row=2, col=1
            )
        
        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
        
        # Annotations with metrics
        quant = metrics["quant"]
        summary_text = (
            f"<b>Signal Statistics</b><br>"
            f"Total Signals: {quant['n_signals']}<br>"
            f"Final Mean: {quant['final_mean']:.2f}%<br>"
            f"Final Median: {quant['final_median']:.2f}%<br>"
            f"Avg Max: {quant['avg_max_uplift']:.2f}%<br>"
            f"Median Max: {quant['median_max_uplift']:.2f}%<br>"
            f"Reaching 5%: {quant.get('pct_reaching_5_percent', 0):.1f}%<br>"
            f"Avg Time to Max: {quant.get('avg_time_to_max_minutes', 0):.0f}m"
        )
        fig.add_annotation(
            x=0.02, y=0.98, 
            xref="paper", yref="paper",
            text=summary_text, 
            showarrow=False,
            bordercolor="black", 
            borderwidth=1, 
            borderpad=8,
            bgcolor="white", 
            opacity=0.9, 
            align="left",
            xanchor="left",
            yanchor="top",
            font=dict(size=11)
        )
        
        # Layout
        fig.update_xaxes(title_text="Minutes After Signal", row=2, col=1)
        fig.update_yaxes(title_text="Relative Change", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Change", row=2, col=1)
        
        fig.update_layout(
            title={
                "text": "Signal Performance Profile Analysis",
                "x": 0.5,
                "xanchor": "center"
            },
            template="plotly_white",
            hovermode="x unified",
            height=1000,
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def plot_signals_on_price(
        self,
        raw_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        pair: str
    ) -> Optional[go.Figure]:
        """Plot signals overlaid on price chart for a specific pair"""
        
        # Filter data for the pair
        price_data = raw_df.filter(pl.col("pair") == pair).sort("timestamp")
        sig_data = signals_df.filter(pl.col("pair") == pair).sort("timestamp")
        
        if price_data.height == 0:
            logger.warning(f"No price data found for {pair}")
            return None
        
        # Join signals with prices
        signals_with_price = sig_data.join(
            price_data.select(["timestamp", "close"]),
            on="timestamp",
            how="inner"
        )
        
        fig = go.Figure()
        
        # Price line
        timestamps = price_data["timestamp"].to_list()
        prices = price_data["close"].to_list()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=prices,
            mode='lines',
            name=f'{pair} Price',
            line=dict(color='#2E86C1', width=1.5),
            hovertemplate='%{x}<br>Price: %{y:.2f}<extra></extra>'
        ))
        
        # Buy signals (signal == 1)
        buys = signals_with_price.filter(pl.col("signal") == 1)
        if buys.height > 0:
            fig.add_trace(go.Scatter(
                x=buys["timestamp"].to_list(),
                y=buys["close"].to_list(),
                mode='markers',
                name=f'Buy Signals ({buys.height})',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='#00CC96',
                    line=dict(width=1, color='black')
                ),
                hovertemplate='%{x}<br>Buy: %{y:.2f}<extra></extra>'
            ))
        
        # Sell signals (signal == -1)
        sells = signals_with_price.filter(pl.col("signal") == -1)
        if sells.height > 0:
            fig.add_trace(go.Scatter(
                x=sells["timestamp"].to_list(),
                y=sells["close"].to_list(),
                mode='markers',
                name=f'Sell Signals ({sells.height})',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='#EF553B',
                    line=dict(width=1, color='black')
                ),
                hovertemplate='%{x}<br>Sell: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': f'<b>SignalFlow Analysis: {pair}</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Date',
            yaxis_title='Price (USDT)',
            template='plotly_white',
            hovermode='x unified',
            height=600,
            width=1400,
            xaxis=dict(
                showgrid=True, 
                gridcolor='lightgray',
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='lightgray'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig