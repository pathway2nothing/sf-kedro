"""Flow configuration loader.

Re-exports from signalflow.config for backward compatibility.
All new code should use signalflow.config directly.

This module provides sf-kedro specific defaults (conf path relative to sf-kedro).

Example:
    >>> from sf_kedro.utils.flow_config import load_flow_config, load_flow
    >>>
    >>> # Load as dict (legacy)
    >>> config = load_flow_config("grid_sma")
    >>>
    >>> # Load as Flow (DAG)
    >>> flow = load_flow("grid_sma")
    >>> flow.compile()  # Resolve dependencies
    >>> flow.plan()     # Get execution order
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Re-export core utilities from signalflow.config
from signalflow.config import (
    # New names
    Artifact,
    Dependency,
    Flow,
    # Backward compatibility
    Edge,
    FlowConfig,
    FlowDAG,
    # Other
    EntryMode,
    Node,
    SignalReconciliation,
    StrategySubgraph,
    deep_merge,
    load_yaml,
)
from signalflow.config.loader import _resolve_env_vars, get_flow_info

__all__ = [
    # Flow classes (new names)
    "Artifact",
    "Dependency",
    "Flow",
    # Backward compatibility
    "Edge",
    "FlowDAG",
    # Other classes
    "EntryMode",
    "FlowConfig",
    "Node",
    "SignalReconciliation",
    "StrategySubgraph",
    # Functions
    "config_to_dag",
    "config_to_flow",
    "deep_merge",
    "get_flow_info",
    "list_flows",
    "load_flow",
    "load_flow_config",
    "load_flow_dag",
    "load_yaml",
]


def _default_conf_path() -> Path:
    """Get default conf path relative to sf-kedro."""
    return Path(__file__).parent.parent.parent.parent / "conf" / "base"


def load_flow_config(flow_id: str, conf_path: Path | str | None = None) -> dict[str, Any]:
    """Load flow configuration by ID.

    sf-kedro wrapper that defaults to sf-kedro/conf/base.
    For generic usage, use signalflow.config.load_flow_config instead.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma', 'baseline_sma')
        conf_path: Path to conf directory. Defaults to sf-kedro/conf/base

    Returns:
        Merged configuration dictionary
    """
    from signalflow.config import load_flow_config as sf_load_flow_config

    # Use sf-kedro default path if not specified
    if conf_path is None:
        conf_path = _default_conf_path()

    return sf_load_flow_config(flow_id, conf_path)


def list_flows(conf_path: Path | str | None = None) -> list[str]:
    """List available flow configurations.

    sf-kedro wrapper that defaults to sf-kedro/conf/base.

    Args:
        conf_path: Path to conf directory

    Returns:
        List of flow IDs
    """
    from signalflow.config import list_flows as sf_list_flows

    if conf_path is None:
        conf_path = _default_conf_path()

    return sf_list_flows(conf_path)


def config_to_flow(config: dict[str, Any]) -> Flow:
    """Convert a flow config dict to a Flow.

    Transforms the chain-style flow config (detector → strategy)
    into a DAG with explicit nodes and dependencies.

    Supports:
    - Single detector (detector: {...}) or multiple detectors (detectors: [...])
    - training_only flag for detectors used only in validator training
    - Features list (features: [...])
    - Labeler (labeler: {...} or labeling: {...})
    - Validator with ML model config

    Args:
        config: Flow configuration dictionary

    Returns:
        Flow with nodes and inferred dependencies

    Example:
        >>> config = load_flow_config("grid_sma")
        >>> flow = config_to_flow(config)
        >>> flow.compile()
        >>> flow.plan()
    """
    nodes: dict[str, dict[str, Any]] = {}

    flow_id = config.get("flow_id", "flow")

    # Data loader node
    data_config = config.get("data", {})
    store_config = data_config.get("store", {})
    nodes["loader"] = {
        "type": "data/loader",
        "name": store_config.get("type", "duckdb") + "/" + data_config.get("data_type", "spot"),
        "config": {
            "pairs": data_config.get("pairs", ["BTCUSDT"]),
            "timeframe": data_config.get("timeframe", "1h"),
            "start": data_config.get("start"),
            "end": data_config.get("end"),
            **{k: v for k, v in store_config.items() if k != "type"},
        },
    }
    # Remove None values
    nodes["loader"]["config"] = {k: v for k, v in nodes["loader"]["config"].items() if v is not None}

    # Feature nodes
    features = config.get("features", [])
    for i, feat in enumerate(features):
        feat_name = feat.get("name", feat.get("type", "unknown"))
        feat_params = {k: v for k, v in feat.items() if k not in ("name", "type")}
        nodes[f"feature_{i}"] = {
            "type": "feature",
            "name": feat_name,
            "config": feat_params,
        }

    # Detector node(s) - support both single and multiple
    detectors = config.get("detectors", [])
    if not detectors:
        # Legacy single detector format
        detector_config = config.get("detector")
        if detector_config:
            detectors = [detector_config]

    for i, det in enumerate(detectors):
        det_id = det.get("id", f"detector_{i}" if len(detectors) > 1 else "detector")
        det_type = det.get("type", "")
        det_params = {k: v for k, v in det.items() if k not in ("id", "type", "training_only")}
        nodes[det_id] = {
            "type": "signals/detector",
            "name": det_type,
            "config": det_params,
            "training_only": det.get("training_only", False),
        }

    # Labeler node (if present) - support both keys
    labeling_config = config.get("labeler") or config.get("labeling")
    if labeling_config:
        labeler_type = labeling_config.get("type", "fixed_horizon")
        labeler_params = {k: v for k, v in labeling_config.items() if k != "type"}
        nodes["labeler"] = {
            "type": "signals/labeler",
            "name": labeler_type,
            "config": labeler_params,
        }

    # Validator node (if present)
    validator_config = config.get("validator")
    if validator_config:
        validator_type = validator_config.get("type", "lightgbm")
        validator_params = {k: v for k, v in validator_config.items() if k != "type"}
        nodes["validator"] = {
            "type": "signals/validator",
            "name": validator_type,
            "config": validator_params,
        }

    # Strategy node
    strategy_config = config.get("strategy", {})
    strategy_node_config = {
        "entry_rules": strategy_config.get("entry_rules", []),
        "exit_rules": strategy_config.get("exit_rules", []),
        "entry_filters": _extract_entry_filters(strategy_config),
        "metrics": strategy_config.get("metrics", []),
        "signal_reconciliation": strategy_config.get("signal_reconciliation", "any"),
        "entry_mode": strategy_config.get("entry_mode", "sequential"),
    }

    # Optional strategy model with fallbacks
    if strategy_config.get("strategy_model"):
        strategy_node_config["strategy_model"] = strategy_config["strategy_model"]
    if strategy_config.get("fallback_entry"):
        strategy_node_config["fallback_entry"] = strategy_config["fallback_entry"]
    if strategy_config.get("fallback_exit"):
        strategy_node_config["fallback_exit"] = strategy_config["fallback_exit"]
    if strategy_config.get("signal_weights"):
        strategy_node_config["signal_weights"] = strategy_config["signal_weights"]

    nodes["strategy"] = {
        "type": "strategy",
        "config": strategy_node_config,
    }

    return Flow.from_dict({
        "id": flow_id,
        "name": config.get("flow_name", flow_id),
        "nodes": nodes,
        "config": {
            "capital": config.get("capital", 10000.0),
            "fee": config.get("fee", 0.001),
            "slippage": config.get("slippage", 0.0),
        },
    })


# Backward compatibility alias
def config_to_dag(config: dict[str, Any]) -> Flow:
    """Backward compatibility alias for config_to_flow."""
    return config_to_flow(config)


def _extract_entry_filters(strategy_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract entry filters from strategy config."""
    filters = []
    for rule in strategy_config.get("entry_rules", []):
        for f in rule.get("entry_filters", []):
            filters.append(f)
    return filters


def load_flow(flow_id: str, conf_path: Path | str | None = None) -> Flow:
    """Load flow configuration as a Flow.

    Loads the flow config and converts it to a Flow structure
    with explicit nodes and auto-inferred dependencies.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma')
        conf_path: Path to conf directory. Defaults to sf-kedro/conf/base

    Returns:
        Flow with nodes and dependencies

    Example:
        >>> flow = load_flow("grid_sma")
        >>> flow.compile()  # Resolve dependencies
        >>> flow.plan()     # Get execution order
        >>> flow.run()      # Execute backtest
    """
    config = load_flow_config(flow_id, conf_path)
    return config_to_flow(config)


# Backward compatibility alias
def load_flow_dag(flow_id: str, conf_path: Path | str | None = None) -> Flow:
    """Backward compatibility alias for load_flow."""
    return load_flow(flow_id, conf_path)
