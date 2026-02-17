"""Flow configuration loader.

Loads and merges common defaults with flow-specific configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def _default_conf_path() -> Path:
    """Get default conf path relative to this file."""
    return Path(__file__).parent.parent.parent.parent / "conf" / "base"


def _resolve_conf_path(conf_path: Path | str | None) -> Path:
    """Resolve conf path to Path object."""
    return _default_conf_path() if conf_path is None else Path(conf_path)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_flow_config(flow_id: str, conf_path: Path | str | None = None) -> dict[str, Any]:
    """Load flow configuration by ID.

    Merges common defaults with flow-specific config.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma', 'baseline_sma')
        conf_path: Path to conf directory. Defaults to sf-kedro/conf/base

    Returns:
        Merged configuration dictionary

    Raises:
        FileNotFoundError: If flow config doesn't exist
    """
    conf_path = _resolve_conf_path(conf_path)

    # Load common config
    common_path = conf_path / "parameters" / "common.yml"
    common_config: dict = {}
    if common_path.exists():
        with open(common_path) as f:
            common_config = yaml.safe_load(f) or {}
        logger.debug(f"Loaded common config from {common_path}")

    # Load flow-specific config
    flow_path = conf_path / "flows" / f"{flow_id}.yml"
    if not flow_path.exists():
        raise FileNotFoundError(f"Flow config not found: {flow_path}")

    with open(flow_path) as f:
        flow_config = yaml.safe_load(f) or {}
    logger.info(f"Loaded flow config: {flow_id} from {flow_path}")

    # Merge configs: flow overrides common
    defaults = common_config.get("defaults", {})
    merged = deep_merge(defaults, flow_config)

    # Add telegram config
    if "telegram" in common_config:
        merged["telegram"] = common_config["telegram"]

    # Resolve output paths with flow_id
    output = common_config.get("output", {})
    merged["output"] = {
        "signals": output.get("signals", "data/08_reporting/{flow_id}/signals").format(flow_id=flow_id),
        "strategy": output.get("strategy", "data/08_reporting/{flow_id}/strategy").format(flow_id=flow_id),
        "db": output.get("db", "data/07_model_output/strategy_{flow_id}.duckdb").format(flow_id=flow_id),
    }

    # Ensure flow_id is set
    merged["flow_id"] = flow_id

    return merged


def list_flows(conf_path: Path | str | None = None) -> list[str]:
    """List available flow configurations.

    Args:
        conf_path: Path to conf directory

    Returns:
        List of flow IDs
    """
    conf_path = _resolve_conf_path(conf_path)
    flows_dir = conf_path / "flows"
    if not flows_dir.exists():
        return []

    return [f.stem for f in flows_dir.glob("*.yml")]


def get_flow_info(flow_id: str, conf_path: Path | str | None = None) -> dict[str, str]:
    """Get flow metadata (name, description).

    Args:
        flow_id: Flow identifier
        conf_path: Path to conf directory

    Returns:
        Dict with flow_id, flow_name, description
    """
    config = load_flow_config(flow_id, conf_path)
    return {
        "flow_id": config.get("flow_id", flow_id),
        "flow_name": config.get("flow_name", flow_id),
        "description": config.get("description", ""),
    }
