"""Flow configuration loader.

Re-exports from signalflow.config for backward compatibility.
All new code should use signalflow.config directly.

This module provides sf-kedro specific defaults (conf path relative to sf-kedro).

Example:
    >>> from sf_kedro.utils.flow_config import load_flow_config, load_flow_dag
    >>>
    >>> # Load as dict (legacy)
    >>> config = load_flow_config("grid_sma")
    >>>
    >>> # Load as DAG (new)
    >>> dag = load_flow_dag("grid_sma")
    >>> dag.get_execution_plan()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Re-export core utilities from signalflow.config
from signalflow.config import (
    FlowConfig,
    FlowDAG,
    Node,
    Edge,
    EntryMode,
    SignalReconciliation,
    StrategySubgraph,
    deep_merge,
    load_yaml,
)
from signalflow.config.loader import _resolve_env_vars, get_flow_info

__all__ = [
    # DAG classes
    "Edge",
    "EntryMode",
    "FlowConfig",
    "FlowDAG",
    "Node",
    "SignalReconciliation",
    "StrategySubgraph",
    # Functions
    "config_to_dag",
    "deep_merge",
    "get_flow_info",
    "list_flows",
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


def config_to_dag(config: dict[str, Any]) -> FlowDAG:
    """Convert a flow config dict to a FlowDAG.

    Transforms the chain-style flow config (detector → strategy)
    into a DAG with explicit nodes and edges.

    Args:
        config: Flow configuration dictionary

    Returns:
        FlowDAG with nodes and inferred edges

    Example:
        >>> config = load_flow_config("grid_sma")
        >>> dag = config_to_dag(config)
        >>> dag.get_execution_plan()
    """
    nodes: dict[str, dict[str, Any]] = {}

    flow_id = config.get("flow_id", "flow")

    # Data loader node
    data_config = config.get("data", {})
    nodes["loader"] = {
        "type": "data/loader",
        "name": data_config.get("store", {}).get("type", "duckdb"),
        "config": {
            "pairs": data_config.get("pairs", ["BTCUSDT"]),
            "timeframe": data_config.get("timeframe", "1h"),
            **data_config.get("store", {}),
            **data_config.get("period", {}),
        },
    }

    # Detector node
    detector_config = config.get("detector")
    if detector_config:
        detector_type = detector_config.get("type", "")
        detector_params = {k: v for k, v in detector_config.items() if k != "type"}
        nodes["detector"] = {
            "type": "signals/detector",
            "name": detector_type,
            "config": detector_params,
        }

    # Validator node (if present)
    validator_config = config.get("validator")
    if validator_config:
        validator_type = validator_config.get("type", "")
        validator_params = {k: v for k, v in validator_config.items() if k != "type"}
        nodes["validator"] = {
            "type": "signals/validator",
            "name": validator_type,
            "config": validator_params,
        }

    # Labeler node (if present)
    labeling_config = config.get("labeling")
    if labeling_config:
        labeler_type = labeling_config.get("type", "")
        labeler_params = {k: v for k, v in labeling_config.items() if k != "type"}
        nodes["labeler"] = {
            "type": "signals/labeler",
            "name": labeler_type,
            "config": labeler_params,
        }

    # Strategy node
    strategy_config = config.get("strategy", {})
    nodes["strategy"] = {
        "type": "strategy",
        "config": {
            "entry_rules": strategy_config.get("entry_rules", []),
            "exit_rules": strategy_config.get("exit_rules", []),
            "entry_filters": _extract_entry_filters(strategy_config),
            "metrics": strategy_config.get("metrics", []),
            "signal_reconciliation": strategy_config.get("signal_reconciliation", "any"),
            "entry_mode": strategy_config.get("entry_mode", "sequential"),
        },
    }

    return FlowDAG.from_dict({
        "id": flow_id,
        "name": config.get("flow_name", flow_id),
        "nodes": nodes,
        "config": {
            "capital": config.get("capital", 10000.0),
            "fee": config.get("fee", 0.001),
        },
    })


def _extract_entry_filters(strategy_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract entry filters from strategy config."""
    filters = []
    for rule in strategy_config.get("entry_rules", []):
        for f in rule.get("entry_filters", []):
            filters.append(f)
    return filters


def load_flow_dag(flow_id: str, conf_path: Path | str | None = None) -> FlowDAG:
    """Load flow configuration as a FlowDAG.

    Loads the flow config and converts it to a DAG structure
    with explicit nodes and auto-inferred edges.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma')
        conf_path: Path to conf directory. Defaults to sf-kedro/conf/base

    Returns:
        FlowDAG with nodes and edges

    Example:
        >>> dag = load_flow_dag("grid_sma")
        >>> print(dag.get_loaders())
        >>> print(dag.get_detectors())
        >>> print(dag.topological_sort())
    """
    config = load_flow_config(flow_id, conf_path)
    return config_to_dag(config)
