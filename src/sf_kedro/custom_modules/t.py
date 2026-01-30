

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from signalflow.core import RawDataType, sf_component, SfComponentType
from signalflow.feature.base import FeatureExtractor


@dataclass
@sf_component(name="lin_reg")
class LinearRegressionExtractor(FeatureExtractor):
    """Linear regression forecast feature extractor.
    
    Creates a derived feature by predicting the next value of a feature
    using linear regression on lagged values. Can optionally extract the
    base feature first using SignalFlow registry.
    
    Two modes of operation:
        1. Direct mode: source_col already exists in input DataFrame
        2. Composite mode: base_extractor_name specified, extracts feature first
    
    Lag Configuration:
        - n_lags: Number of lagged values for regression
        - lag_step: Step between lags (1 = every bar, 5 = every 5th bar)
        - step_agg: How to aggregate values within step window
    
    Example lag matrix (n_lags=3, lag_step=2, step_agg="mean"):
        Row t uses: [mean(t-6,t-5), mean(t-4,t-3), mean(t-2,t-1)] -> predict t
    
    Attributes:
        source_col: Source feature column to forecast. Required if no base_extractor.
        base_extractor_name: Registry name of base extractor (e.g., "rsi", "sma").
        base_extractor_params: Parameters for base extractor constructor.
        out_col: Output column name. Default: "{source_col}_forecast".
        n_lags: Number of lagged values for regression. Default: 5.
        lag_step: Step between lags in bars. Default: 1.
        step_agg: Aggregation for stepped lags ('last', 'mean', 'sum'). Default: 'last'.
        min_train_samples: Minimum samples before generating forecasts. Default: 30.
        fit_window: Rolling window for fitting. None = expanding window.
        global_feature: If True, compute on reference_pair only, broadcast to all.
        reference_pair: Pair to use when global_feature=True.
        add_residual: Add residual (actual - predicted) column. Default: False.
        add_change: Add predicted change column. Default: True.
    
    Output Columns:
        - {out_col}: Forecasted value
        - {out_col}_change: Predicted change (if add_change=True)
        - {out_col}_residual: Actual - Predicted (if add_residual=True)
    
    Example:
        >>> # Mode 1: Forecast existing column
        >>> extractor = RegressionForecastExtractor(
        ...     source_col="rsi_14",
        ...     n_lags=5,
        ... )
        
        >>> # Mode 2: Extract RSI first, then forecast
        >>> extractor = RegressionForecastExtractor(
        ...     base_extractor_name="rsi",
        ...     base_extractor_params={"period": 14, "price_col": "close"},
        ...     n_lags=5,
        ... )
        
        >>> # Multi-timeframe: 5-bar steps with mean aggregation
        >>> extractor = RegressionForecastExtractor(
        ...     source_col="close",
        ...     n_lags=4,
        ...     lag_step=5,
        ...     step_agg="mean",
        ... )
        
        >>> # Global feature (same forecast for all pairs)
        >>> extractor = RegressionForecastExtractor(
        ...     source_col="btc_dominance",
        ...     global_feature=True,
        ...     reference_pair="BTCUSDT",
        ... )
    
    Note:
        - First min_train_samples rows will have null forecasts
        - Model is refit for each row (stateless)
        - For composite mode, source_col is auto-detected if base produces single column
    """
    
    source_col: str | None = None
    
    base_extractor_name: str | None = None
    base_extractor_params: dict[str, Any] = field(default_factory=dict)
    
    out_col: str | None = None
    
    n_lags: int = 5
    lag_step: int = 1
    step_agg: Literal["last", "mean", "sum"] = "last"
    
    min_train_samples: int = 30
    fit_window: int | None = None
    
    global_feature: bool = False
    reference_pair: str = "BTCUSDT"

    add_residual: bool = False
    add_change: bool = True
    
    _model: LinearRegression = field(default_factory=LinearRegression, repr=False, init=False)
    _base_extractor: FeatureExtractor | None = field(default=None, repr=False, init=False)
    _resolved_source_col: str | None = field(default=None, repr=False, init=False)
    
    def __post_init__(self) -> None:
        """Validate parameters and initialize base extractor if needed."""
        super().__post_init__()
        
        if self.source_col is None and self.base_extractor_name is None:
            raise ValueError(
                "Either source_col or base_extractor_name must be specified"
            )
        
        if self.n_lags < 1:
            raise ValueError(f"n_lags must be >= 1, got {self.n_lags}")
        if self.lag_step < 1:
            raise ValueError(f"lag_step must be >= 1, got {self.lag_step}")
        if self.step_agg not in ("last", "mean", "sum"):
            raise ValueError(f"step_agg must be 'last', 'mean', or 'sum', got {self.step_agg}")
        
        # Minimum samples must cover at least the lag span
        min_required = self.n_lags * self.lag_step + 1
        if self.min_train_samples < min_required:
            self.min_train_samples = min_required
        
        if self.fit_window is not None and self.fit_window < self.min_train_samples:
            raise ValueError(
                f"fit_window ({self.fit_window}) must be >= min_train_samples ({self.min_train_samples})"
            )
        
        self._model = LinearRegression()
        
        # Initialize base extractor from registry if specified
        if self.base_extractor_name is not None:
            from signalflow.core.registry import default_registry
            
            self._base_extractor = default_registry.create(
                SfComponentType.FEATURE_EXTRACTOR,
                self.base_extractor_name,
                **self.base_extractor_params,
            )
    
    def _resolve_output_col(self, source_col: str) -> str:
        """Determine output column name."""
        if self.out_col is not None:
            return self.out_col
        
        step_suffix = f"_{self.lag_step}step" if self.lag_step > 1 else ""
        return f"{source_col}_forecast{step_suffix}"
    
    def _resolve_source_col(self, df: pl.DataFrame, original_cols: set[str]) -> str:
        """Auto-detect source column from base extractor output."""
        if self.source_col is not None:
            return self.source_col
        
        new_cols = set(df.columns) - original_cols - {self.pair_col, self.ts_col, self.offset_col}
        
        if len(new_cols) == 1:
            return new_cols.pop()
        elif len(new_cols) > 1:
            raise ValueError(
                f"Base extractor produced multiple columns {sorted(new_cols)}. "
                f"Specify source_col explicitly."
            )
        else:
            raise ValueError("Base extractor produced no new columns.")
    
    def _aggregate_step(self, values: np.ndarray, start_idx: int, end_idx: int) -> float:
        """Aggregate values within a step window."""
        window = values[start_idx:end_idx]
        
        if len(window) == 0:
            return np.nan
        
        if self.step_agg == "last":
            return window[-1]
        elif self.step_agg == "mean":
            return np.nanmean(window)
        elif self.step_agg == "sum":
            return np.nansum(window)
        else:
            return window[-1]
    
    def _build_lag_features(self, values: np.ndarray, target_idx: int) -> np.ndarray | None:
        """Build lag feature vector for a single prediction."""
        features = np.zeros(self.n_lags)
        
        for lag_num in range(self.n_lags):
            lag_offset = (self.n_lags - lag_num) * self.lag_step
            
            if self.lag_step == 1:
                idx = target_idx - lag_offset
                if idx < 0:
                    return None
                features[lag_num] = values[idx]
            else:
                end_idx = target_idx - (self.n_lags - lag_num - 1) * self.lag_step
                start_idx = end_idx - self.lag_step
                
                if start_idx < 0:
                    return None
                
                features[lag_num] = self._aggregate_step(values, start_idx, end_idx)
        
        return features
    
    def _build_training_data(
        self, 
        values: np.ndarray, 
        current_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training dataset up to current index."""
        if self.fit_window is not None:
            start_idx = max(self.n_lags * self.lag_step, current_idx - self.fit_window)
        else:
            start_idx = self.n_lags * self.lag_step
        
        X_list = []
        y_list = []
        
        for i in range(start_idx, current_idx):
            features = self._build_lag_features(values, i)
            if features is not None:
                X_list.append(features)
                y_list.append(values[i])
        
        if not X_list:
            return np.array([]).reshape(0, self.n_lags), np.array([])
        
        return np.array(X_list), np.array(y_list)
    
    def _compute_forecasts(
        self,
        df: pl.DataFrame,
        source_col: str,
    ) -> pl.DataFrame:
        """Compute regression forecasts for DataFrame."""
        if source_col not in df.columns:
            raise ValueError(
                f"Source column '{source_col}' not found. "
                f"Available: {df.columns}"
            )
        
        n_rows = df.height
        values = df[source_col].to_numpy().astype(np.float64)
        
        forecasts = np.full(n_rows, np.nan, dtype=np.float64)
        
        for i in range(self.min_train_samples, n_rows):
            X_train, y_train = self._build_training_data(values, i)
            
            if len(X_train) < 10:
                continue
            
            try:
                self._model.fit(X_train, y_train)
            except Exception:
                continue
            
            X_pred = self._build_lag_features(values, i)
            if X_pred is None:
                continue
            
            forecasts[i] = self._model.predict(X_pred.reshape(1, -1))[0]
        
        out_col = self._resolve_output_col(source_col)
        
        result = df.with_columns([
            pl.Series(name=out_col, values=forecasts),
        ])
        
        if self.add_change:
            changes = forecasts - values
            result = result.with_columns([
                pl.Series(name=f"{out_col}_change", values=changes),
            ])
        
        if self.add_residual:
            residuals = np.full(n_rows, np.nan, dtype=np.float64)
            for i in range(1, n_rows):
                if not np.isnan(forecasts[i-1]):
                    residuals[i] = values[i] - forecasts[i-1]
            
            result = result.with_columns([
                pl.Series(name=f"{out_col}_residual", values=residuals),
            ])
        
        return result
    
    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Compute regression forecast for a single group."""
        source_col = self._resolved_source_col or self.source_col
        
        if source_col is None:
            raise ValueError("source_col not resolved. Call extract() first.")
        
        return self._compute_forecasts(group_df, source_col)
    
    def extract(
        self, 
        df: pl.DataFrame, 
        data_context: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        """Extract regression forecast features.
        
        If base_extractor is configured, extracts base feature first.
        Then computes forecast, handling global_feature mode if enabled.
        
        Args:
            df: Input DataFrame (OHLCV or with existing features).
            data_context: Optional context.
            
        Returns:
            DataFrame with forecast columns added.
        """
        original_cols = set(df.columns)
        
        if self._base_extractor is not None:
            df = self._base_extractor.extract(df, data_context)
            self._resolved_source_col = self._resolve_source_col(df, original_cols)
        else:
            self._resolved_source_col = self.source_col
        
        if self.global_feature:
            return self._extract_global(df, data_context)
        
        return super().extract(df, data_context)
    
    def _extract_global(
        self, 
        df: pl.DataFrame, 
        data_context: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        """Extract global forecast (single pair, broadcast to all)."""
        source_col = self._resolved_source_col
        
        ref_df = df.filter(pl.col(self.pair_col) == self.reference_pair)
        
        if ref_df.is_empty():
            raise ValueError(
                f"Reference pair '{self.reference_pair}' not found. "
                f"Available: {df[self.pair_col].unique().to_list()}"
            )
        
        ref_forecasts = self._compute_forecasts(ref_df, source_col)
        
        out_col = self._resolve_output_col(source_col)
        forecast_cols = [out_col]
        if self.add_change:
            forecast_cols.append(f"{out_col}_change")
        if self.add_residual:
            forecast_cols.append(f"{out_col}_residual")
        
        ref_to_join = ref_forecasts.select([self.ts_col] + forecast_cols)
        result = df.join(ref_to_join, on=self.ts_col, how="left")
        
        return result