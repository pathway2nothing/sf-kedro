"""Detection utilities for running detectors with feature computation."""

from __future__ import annotations

from typing import Any

import polars as pl
import signalflow as sf


def run_detection(
    config: dict[str, Any],
    raw_data: sf.RawData,
    data_key: str = "spot",
) -> sf.Signals:
    """Run detector with automatic feature computation.

    Args:
        config: Flow configuration with detector settings
        raw_data: Market data
        data_key: Key for data extraction from RawData

    Returns:
        Detected signals
    """
    from signalflow.feature import FeaturePipeline

    detector_config = config.get("detector", {}).copy()
    detector_type = detector_config.pop("type", "example/sma_cross")

    detector = sf.get_component(
        type=sf.SfComponentType.DETECTOR,
        name=detector_type,
    )(**detector_config)

    # Extract data and compute features
    df = raw_data.data[data_key]

    if hasattr(detector, "features") and detector.features:
        pipeline = FeaturePipeline(features=detector.features)
        df = pipeline.compute(df)

    return detector.detect(df)


def run_detection_with_detector(
    detector: Any,
    raw_data: sf.RawData,
    data_key: str = "spot",
) -> sf.Signals:
    """Run an instantiated detector with automatic feature computation.

    Args:
        detector: Instantiated detector
        raw_data: Market data
        data_key: Key for data extraction from RawData

    Returns:
        Detected signals
    """
    from signalflow.feature import FeaturePipeline

    df = raw_data.data[data_key]

    if hasattr(detector, "features") and detector.features:
        pipeline = FeaturePipeline(features=detector.features)
        df = pipeline.compute(df)

    return detector.detect(df)
