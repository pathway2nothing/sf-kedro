from dataclasses import dataclass
import polars as pl
import signalflow as sf
from signalflow import sf_component
from typing import Any


@dataclass
@sf_component(name='global/log_return')
class MarketLogReturnFeature(sf.feature.GlobalFeature):
    """Market log-return (stationary feature)."""
    
    price_col: str = "close"
    
    requires = ["{price_col}"]
    outputs = ["market_log_ret"]
    
    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute mean log-returns across all pairs per timestamp."""
        
        df = df.with_columns(
            pl.col(self.price_col)
              .log()
              .diff()
              .over(self.group_col)
              .alias("__log_ret")
        )
        
        market_ret = df.group_by(self.ts_col).agg(
            pl.col("__log_ret").mean().alias("market_log_ret")
        )
        
        return (
            df.join(market_ret, on=self.ts_col, how="left")
              .with_columns(pl.col("market_log_ret").fill_null(0))
              .drop("__log_ret")
        )