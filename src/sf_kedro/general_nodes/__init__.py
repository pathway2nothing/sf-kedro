"""Reusable modules for SignalFlow Kedro pipelines."""

from .backtest import (
    run_backtest,
)
from .data_loader import (
    download_market_data,
    load_raw_data_from_storage,
)
from .feature_builder import (
    create_feature_set,
    extract_validation_features,
)
from .labeling import (
    create_labels,
    split_train_val_test,
)
from .ml_validator_factory import (
    create_nn_validator,
    create_sklearn_validator,
    load_validator_from_registry,
)
from .signal_metrics import (
    compute_signal_metrics,
    save_signal_plots,
)
from .signal_processor import (
    detect_signals,
    validate_signals,
)
from .strategy_metrics import (
    compute_strategy_metrics,
    log_last_state_metrics,
    save_strategy_plots,
)

__all__ = [
    # Data
    "download_market_data",
    "load_raw_data_from_storage",
    # Features
    "extract_validation_features",
    "create_feature_set",
    # Signals
    "detect_signals",
    "validate_signals",
    # Labeling
    "create_labels",
    "split_train_val_test",
    # Validators
    "create_sklearn_validator",
    "create_nn_validator",
    "load_validator_from_registry",
    # Backtest
    "run_backtest",
    "log_last_state_metrics",
    # Signal metrics
    "compute_signal_metrics",
    "save_signal_plots",
    "compute_strategy_metrics",
    "save_strategy_plots",
]
