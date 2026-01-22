from __future__ import annotations

from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import signalflow as sf


@dataclass
@sf.sf_component(name="result_main", override=True)
class StrategyMainResult(sf.analytic.StrategyMetric):
    """
    Strategy-level visualization based on results['metrics_df'] (Polars DataFrame).

    """
    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float], 
        **kwargs
    ) -> Dict[str, float]:
        """Compute metric values."""
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        metrics_df: pl.DataFrame | None = results.get("metrics_df")
        if metrics_df is None or metrics_df.height == 0:
            logger.warning("No metrics_df to plot")
            return None

        ts = self._timestamps(metrics_df)

        main_fig = self._plot_main(metrics_df=metrics_df, ts=ts, results=results)
        detailed_fig = self._plot_detailed(metrics_df=metrics_df, ts=ts, results=results)

        # повертаємо список, щоб ти міг зберегти/показати обидва
        figs: list[go.Figure] = [main_fig]
        if detailed_fig is not None:
            figs.append(detailed_fig)
        return figs

    def _timestamps(self, metrics_df: pl.DataFrame):
        if "timestamp" in metrics_df.columns:
            return (
                metrics_df.select(
                    pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="s").alias("datetime")
                )
                .get_column("datetime")
                .to_list()
            )
        return list(range(metrics_df.height))

    def _plot_main(self, *, metrics_df: pl.DataFrame, ts: list, results: dict) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Strategy Performance", "Position Metrics", "Balance Allocation"),
            row_heights=[0.45, 0.30, 0.25],
        )

        # Strategy return (%)
        if "total_return" in metrics_df.columns:
            returns_pct = (metrics_df.get_column("total_return") * 100).to_list()
            fig.add_trace(
                go.Scatter(
                    x=ts, y=returns_pct, mode="lines",
                    name="Strategy Return",
                    hovertemplate="Return: %{y:.2f}%<extra></extra>",
                ),
                row=1, col=1,
            )

        fig.add_hline(y=0, line_dash="dash", line_width=1, row=1, col=1)

        # Positions
        if "open_positions" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=metrics_df.get_column("open_positions").to_list(),
                    mode="lines",
                    name="Open Positions",
                    fill="tozeroy",
                    hovertemplate="Open: %{y}<extra></extra>",
                ),
                row=2, col=1,
            )

        if "closed_positions" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=metrics_df.get_column("closed_positions").to_list(),
                    mode="lines",
                    name="Closed Positions",
                    line=dict(dash="dot"),
                    hovertemplate="Closed: %{y}<extra></extra>",
                ),
                row=2, col=1,
            )

        # Allocation + Total Return overlay
        if "cash" in metrics_df.columns and "equity" in metrics_df.columns:
            cash = metrics_df.get_column("cash").to_list()
            equity = metrics_df.get_column("equity").to_list()
            initial_capital = results.get("initial_capital", equity[0] if equity else 10000)

            allocated_pct = [(eq - c) / eq if eq > 0 else 0.0 for eq, c in zip(equity, cash)]
            free_pct = [c / eq if eq > 0 else 0.0 for eq, c in zip(equity, cash)]
            total_balance_pct = [(eq / initial_capital - 1.0) * 100.0 for eq in equity]

            fig.add_trace(
                go.Scatter(
                    x=ts, y=free_pct,
                    mode="lines", name="Free Cash",
                    line=dict(width=0),
                    fill="tozeroy",
                    stackgroup="balance",
                    hovertemplate="Free: %{y:.1%}<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts, y=allocated_pct,
                    mode="lines", name="In Positions",
                    line=dict(width=0),
                    fill="tonexty",
                    stackgroup="balance",
                    hovertemplate="Allocated: %{y:.1%}<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts, y=total_balance_pct,
                    mode="lines", name="Total Return",
                    yaxis="y4",
                    hovertemplate="Total: %{y:.2f}%<extra></extra>",
                ),
                row=3, col=1,
            )

            fig.update_yaxes(title_text="Allocation", row=3, col=1, tickformat=".0%", range=[0, 1])
            fig.update_layout(
                yaxis4=dict(
                    title="Total Return (%)",
                    overlaying="y3",
                    side="right",
                    showgrid=False,
                )
            )

        final_return = results.get("final_return", 0.0) * 100.0
        fig.update_layout(
            title=dict(text=f"Backtest Results | Total Return: {final_return:.2f}%"),
            template="plotly_white",
            height=900,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.01, x=0),
            showlegend=True,
        )
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        return fig

    def _plot_detailed(self, *, metrics_df: pl.DataFrame, ts: list, results: dict) -> go.Figure | None:
        # робимо тільки якщо є релевантні колонки
        has_dd = "current_drawdown" in metrics_df.columns
        has_util = "capital_utilization" in metrics_df.columns
        if not (has_dd or has_util):
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Drawdown Analysis", "Capital Utilization"),
            row_heights=[0.6, 0.4],
        )

        if has_dd:
            drawdown = metrics_df.get_column("current_drawdown").to_list()
            drawdown_pct = [-d * 100 for d in drawdown]
            fig.add_trace(
                go.Scatter(
                    x=ts, y=drawdown_pct, mode="lines",
                    name="Drawdown",
                    fill="tozeroy",
                    hovertemplate="DD: %{y:.2f}%<extra></extra>",
                ),
                row=1, col=1,
            )

            max_dd = results.get("max_drawdown", 0.0) * 100.0
            if max_dd > 0:
                fig.add_hline(
                    y=-max_dd,
                    line_dash="dash",
                    line_width=1.5,
                    annotation_text=f"Max DD: {max_dd:.2f}%",
                    annotation_position="right",
                    row=1, col=1,
                )

        if has_util:
            util = metrics_df.get_column("capital_utilization").to_list()
            util_pct = [u * 100 for u in util]
            fig.add_trace(
                go.Scatter(
                    x=ts, y=util_pct, mode="lines",
                    name="Capital Utilization",
                    fill="tozeroy",
                    hovertemplate="Util: %{y:.1f}%<extra></extra>",
                ),
                row=2, col=1,
            )
            fig.add_hline(y=100, line_dash="dot", line_width=1, row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=600,
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_yaxes(title_text="Utilization (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig


from dataclasses import dataclass, field
from typing import List

from loguru import logger
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signalflow.core import StrategyState, sf_component
from signalflow.analytic.base import StrategyMetric
from signalflow import RawData


@dataclass
@sf_component(name="result_pair", override=True)
class StrategyPairResult(StrategyMetric):
    """
    Plot price for a specific pair and overlay entries/exits from state.
    """

    pairs: List[str] = field(default_factory=list)            
    price_col: str = "close"      
    ts_col: str = "timestamp"     

    def compute(
        self,
        state: StrategyState,
        prices: dict[str, float], 
        **kwargs
    ) -> Dict[str, float]:
        """Compute metric values."""
        return {}

    def plot(
        self,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        if not self.pairs:
            logger.warning("pair is not provided for StrategyPairResult.plot")
            return None

        figs = []
        for pair in self.pairs:
            try:
                fig = self._plot_pair(pair=pair, results=results, state=state, raw_data=raw_data, **kwargs)
                if fig is not None:
                    figs.append(fig) 
            except Exception as e:
                logger.error(f"Failed to plot pair={pair}: {e}")
        return figs

    def _plot_pair(
        self,
        pair: str,
        results: dict,
        state: StrategyState | None = None,
        raw_data: RawData | None = None,
        **kwargs,
    ) -> list[go.Figure] | go.Figure | None:
        pair_df = self._get_pair_df(results=results, raw_data=raw_data, pair=pair)
        if pair_df is None or pair_df.height == 0:
            logger.warning(f"No data to plot for pair={pair}")
            return None

        ts = self._timestamps(pair_df)
        if self.price_col not in pair_df.columns:
            logger.warning(f"price_col='{self.price_col}' not found in pair_df for pair={pair}")
            return None

        price = pair_df.get_column(self.price_col).to_list()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{pair} Price + Trades", "Position Size"),
        )

        fig.add_trace(go.Scatter(x=ts, y=price, mode="lines", name="Price"), row=1, col=1)

        if state is None:
            fig.update_layout(template="plotly_white", height=650, hovermode="x unified")
            return fig

        trades = self._extract_trades(state=state, pair=pair)

        entry_x, entry_y, exit_x, exit_y = [], [], [], []
        size_x, size_y = [], []

        price_by_epoch = {}
        if self.ts_col in pair_df.columns:
            tlist = pair_df.get_column(self.ts_col).cast(pl.Int64).to_list()
            price_by_epoch = {int(t): float(p) for t, p in zip(tlist, price)}

        for tr in trades:
            et = tr.get("entry_ts")
            xt = tr.get("exit_ts")
            size = float(tr.get("size", 0.0))

            if et is not None:
                entry_x.append(self._epoch_to_datetime(et) if self.ts_col in pair_df.columns else et)
                entry_y.append(price_by_epoch.get(int(et), None))
                size_x.append(self._epoch_to_datetime(et) if self.ts_col in pair_df.columns else et)
                size_y.append(size)

            if xt is not None:
                exit_x.append(self._epoch_to_datetime(xt) if self.ts_col in pair_df.columns else xt)
                exit_y.append(price_by_epoch.get(int(xt), None))
                size_x.append(self._epoch_to_datetime(xt) if self.ts_col in pair_df.columns else xt)
                size_y.append(0.0)

        if entry_x:
            fig.add_trace(
                go.Scatter(
                    x=entry_x, y=entry_y,
                    mode="markers",
                    name="Entry",
                    marker=dict(size=9, symbol="triangle-up"),
                ),
                row=1, col=1,
            )

        if exit_x:
            fig.add_trace(
                go.Scatter(
                    x=exit_x, y=exit_y,
                    mode="markers",
                    name="Exit",
                    marker=dict(size=9, symbol="triangle-down"),
                ),
                row=1, col=1,
            )

        if size_x:
            fig.add_trace(
                go.Scatter(x=size_x, y=size_y, mode="lines+markers", name="Position Size"),
                row=2, col=1,
            )

        fig.update_layout(template="plotly_white", height=700, hovermode="x unified", showlegend=True)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Size", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        return fig

    def _get_pair_df(self, *, results: dict, raw_data: RawData | None, pair: str) -> pl.DataFrame | None:
        pair_dfs = results.get("pair_dfs")
        if isinstance(pair_dfs, dict) and pair in pair_dfs:
            return pair_dfs[pair]

        if raw_data is not None:
            spot = None
            if hasattr(raw_data, "__getitem__"):
                try:
                    spot = raw_data["spot"]
                except Exception:
                    spot = None
            if spot is None and hasattr(raw_data, "data"):
                spot = raw_data.data.get("spot") if isinstance(raw_data.data, dict) else None

            if isinstance(spot, pl.DataFrame) and "pair" in spot.columns:
                return spot.filter(pl.col("pair") == pair)

        return None

    def _timestamps(self, df: pl.DataFrame) -> list[Any]:
        if self.ts_col in df.columns:
            return (
                df.select(pl.from_epoch(pl.col(self.ts_col).cast(pl.Int64), time_unit="s").alias("datetime"))
                .get_column("datetime")
                .to_list()
            )
        return list(range(df.height))

    def _epoch_to_datetime(self, t: int):
        import datetime as dt
        return dt.datetime.utcfromtimestamp(int(t))

    def _extract_trades(self, *, state: StrategyState, pair: str) -> list[dict[str, Any]]:
        """
        Приводимо позиції до формату:
        { entry_ts: int|None, exit_ts: int|None, size: float }

        Тут 100% треба під твої класи підкрутити.
        """
        out: list[dict[str, Any]] = []

        positions = getattr(getattr(state, "portfolio", None), "positions", None)
        if not isinstance(positions, dict):
            return out

        for p in positions.values():
            if getattr(p, "pair", None) != pair:
                continue

            entry_time = getattr(p, "entry_time", None)
            exit_time = getattr(p, "last_time", None) if getattr(p, "is_closed", False) else None

            entry_ts = self._to_epoch(entry_time)
            exit_ts = self._to_epoch(exit_time)

            size = getattr(p, "quantity", None)
            if size is None:
                size = getattr(p, "size", 0.0)

            out.append(
                {
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "size": float(size or 0.0),
                }
            )

        return out

    def _to_epoch(self, t: Any) -> int | None:
        if t is None:
            return None
        if hasattr(t, "timestamp"):
            return int(t.timestamp())
        if isinstance(t, (int,)):
            return int(t)
        return None
