"""
Tests for Kedro project pipelines.
"""

from pathlib import Path

import pytest
from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    """Test that pipelines are registered correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Bootstrap the project before each test."""
        bootstrap_project(Path.cwd())

    def test_pipelines_registered(self):
        """Test that expected pipelines are registered."""
        expected = {
            "__default__",
            "baseline",
            "feature_analysis",
            "ml_validated",
            "nn_validated",
            "optuna_tuning",
            "production",
            "walk_forward",
        }
        assert expected.issubset(set(pipelines.keys()))

    def test_baseline_pipeline_has_nodes(self):
        """Test that baseline pipeline has nodes."""
        baseline = pipelines.get("baseline")
        assert baseline is not None
        assert len(baseline.nodes) > 0

    def test_ml_validated_pipeline_has_nodes(self):
        """Test that ml_validated pipeline has nodes."""
        ml_pipeline = pipelines.get("ml_validated")
        assert ml_pipeline is not None
        assert len(ml_pipeline.nodes) > 0

    def test_walk_forward_pipeline_has_nodes(self):
        """Test that walk_forward pipeline has nodes."""
        wf_pipeline = pipelines.get("walk_forward")
        assert wf_pipeline is not None
        assert len(wf_pipeline.nodes) > 0
