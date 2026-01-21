"""Reusable modules for SignalFlow Kedro pipelines."""

from .data_loader import (
    download_market_data,
    load_raw_data_from_storage,
)

from .feature_builder import (
    extract_validation_features,
    create_feature_set,
)

from .signal_processor import (
    detect_signals,
    validate_signals,
)

from .labeling import (
    create_labels,
    split_train_val_test,
)

from .validator_factory import (
    create_sklearn_validator,
    create_nn_validator,
    load_validator_from_registry,
)

from .backtest_engine import (
    run_backtest,
    calculate_backtest_metrics,
)

from .signal_metrics import (
    compute_signal_metrics,
    save_signal_plots,
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
    "calculate_backtest_metrics",
    
    # Signal metrics
    "compute_signal_metrics",
    "save_signal_plots",
]