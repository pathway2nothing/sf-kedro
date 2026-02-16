"""Safe MLflow logging utilities.

This module provides wrapper functions for MLflow operations that:
- Don't raise exceptions if MLflow is unavailable
- Can be disabled via MLFLOW_ENABLED environment variable
- Gracefully handle import errors
"""

import os
from typing import Any, Dict, Optional


def _is_mlflow_enabled() -> bool:
    """Check if MLflow logging is enabled."""
    mlflow_enabled = os.getenv("MLFLOW_ENABLED", "true").lower()
    return mlflow_enabled in ("true", "1", "yes")


def _get_mlflow():
    """Get mlflow module if available."""
    if not _is_mlflow_enabled():
        return None
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def log_params(params: Dict[str, Any]) -> None:
    """
    Safely log parameters to MLflow.

    Args:
        params: Dictionary of parameters to log
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.log_params(params)
    except Exception:
        # Silently ignore MLflow errors
        pass


def log_param(key: str, value: Any) -> None:
    """
    Safely log a single parameter to MLflow.

    Args:
        key: Parameter name
        value: Parameter value
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.log_param(key, value)
    except Exception:
        pass


def log_metrics(metrics: Dict[str, float]) -> None:
    """
    Safely log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics to log
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.log_metrics(metrics)
    except Exception:
        pass


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """
    Safely log a single metric to MLflow.

    Args:
        key: Metric name
        value: Metric value
        step: Optional step number
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)
    except Exception:
        pass


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Safely log an artifact to MLflow.

    Args:
        local_path: Path to the artifact file
        artifact_path: Optional path within the artifact store
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path)
    except Exception:
        pass


def log_artifacts(local_dir: str, artifact_path: Optional[str] = None) -> None:
    """
    Safely log a directory of artifacts to MLflow.

    Args:
        local_dir: Path to the directory
        artifact_path: Optional path within the artifact store
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        if artifact_path:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        else:
            mlflow.log_artifacts(local_dir)
    except Exception:
        pass


def set_tag(key: str, value: Any) -> None:
    """
    Safely set a tag in MLflow.

    Args:
        key: Tag name
        value: Tag value
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.set_tag(key, value)
    except Exception:
        pass


def set_tags(tags: Dict[str, Any]) -> None:
    """
    Safely set multiple tags in MLflow.

    Args:
        tags: Dictionary of tags to set
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return

    try:
        mlflow.set_tags(tags)
    except Exception:
        pass
