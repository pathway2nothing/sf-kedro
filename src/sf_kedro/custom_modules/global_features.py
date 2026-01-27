from abc import ABC, abstractmethod
from typing import Any, ClassVar

import polars as pl
from dataclasses import dataclass

from signalflow.core import SfComponentType
from signalflow.feature import FeatureExtractor

@dataclass
class GlobalFeatureExtractor(FeatureExtractor, ABC):
    """Base class for features that require cross-sectional (global) data context.
    
    Unlike standard FeatureExtractors which process each pair independently,
    GlobalFeatureExtractors aggregate data across ALL pairs for each timestamp.
    
    Use cases:
        - Market Indices (Benchmarks)
        - Global Market Breadth (e.g., % of stocks above SMA200)
        - Correlation/Covariance Matrices
        - PCA/Factor Analysis features

    Pipeline:
        1. Validate & Sort Input (by timestamp, pair).
        2. Compute Global Features (User implemented `compute_global`).
        3. Join Global Features back to original data (on timestamp).
        4. Project columns (clean up).
    """

    join_on_col: str = "timestamp"

    def extract(self, df: pl.DataFrame, data_context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Orchestrates the global extraction process."""
        if not isinstance(df, pl.DataFrame):
             raise TypeError(f"Input must be pl.DataFrame, got {type(df)}")
        
        self._validate_input(df)

        df_sorted = df.sort([self.ts_col, self.pair_col])


        global_features = self.compute_global(df_sorted, data_context)

        if self.ts_col not in global_features.columns:
            raise ValueError(f"compute_global output must contain '{self.ts_col}' column for joining.")


        out = df_sorted.join(global_features, on=self.ts_col, how="left")

        if self.keep_input_columns:
            return out
        
        input_cols = set(df_sorted.columns)
        new_cols = [c for c in out.columns if c not in input_cols]
        
        return out.select([self.pair_col, self.ts_col] + new_cols)

    @abstractmethod
    def compute_global(self, df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Compute the cross-sectional features.

        Args:
            df (pl.DataFrame): Input data sorted by [timestamp, pair].
            data_context: Optional context.

        Returns:
            pl.DataFrame: A DataFrame containing at least:
                - timestamp (for joining)
                - new feature columns
            
            Note: This DF usually has 1 row per timestamp (aggregating all pairs),
            but it could theoretically have more if the logic is sector-based.
        """
        raise NotImplementedError

    def compute_group(self, group_df: pl.DataFrame, data_context: dict[str, Any] | None) -> pl.DataFrame:
        """Disabled for Global Extractors."""
        raise NotImplementedError("GlobalFeatureExtractor does not use per-group processing.")

from dataclasses import dataclass
from typing import Any

import polars as pl

from signalflow.core import sf_component
# from signalflow.feature import GlobalFeatureExtractor


@dataclass
@sf_component(name='global/market_log_return', override=True)
class MarketLogReturn(GlobalFeatureExtractor):
    """Market log-return (stationary feature)."""
    
    price_col: str = "close"
    output_col: str = "market_log_ret"
    
    def compute_global(
        self, 
        df: pl.DataFrame, 
        data_context: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        """Compute instantaneous log-returns (stationary)."""
        
        log_returns = df.select([
            pl.col(self.ts_col),
            pl.col(self.pair_col),
            pl.col(self.price_col).log().diff().over(self.pair_col).alias("__log_ret")
        ])

        return (
            log_returns
            .group_by(self.ts_col)
            .agg(pl.col("__log_ret").mean().alias(self.output_col))
            .sort(self.ts_col)
            .with_columns(pl.col(self.output_col).fill_null(0))
        )

feature_set = sf.feature.FeatureSet([
    MarketLogReturn()
])

df_features = feature_set.extract(raw_data_view)