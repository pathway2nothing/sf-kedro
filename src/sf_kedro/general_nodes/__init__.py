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
    # Signal metrics
    "compute_signal_metrics",
    "compute_strategy_metrics",
    "create_feature_set",
    # Labeling
    "create_labels",
    "create_nn_validator",
    # Validators
    "create_sklearn_validator",
    # Signals
    "detect_signals",
    # Data
    "download_market_data",
    # Features
    "extract_validation_features",
    "load_raw_data_from_storage",
    "load_validator_from_registry",
    "log_last_state_metrics",
    # Backtest
    "run_backtest",
    "save_signal_plots",
    "save_strategy_plots",
    "split_train_val_test",
    "validate_signals",
]
