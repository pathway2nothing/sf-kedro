"""Tests for Kedro project pipelines."""

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
            "backtest",
            "analyze",
            "train",
            "tune",
            "validate",
        }
        assert expected.issubset(set(pipelines.keys()))

    def test_backtest_pipeline_has_nodes(self):
        """Test that backtest pipeline has expected nodes."""
        backtest = pipelines.get("backtest")
        assert backtest is not None
        assert len(backtest.nodes) == 5
        node_names = [n.name for n in backtest.nodes]
        assert "load_flow_data" in node_names
        assert "run_flow_detection" in node_names
        assert "run_flow_backtest" in node_names

    def test_analyze_pipeline_has_nodes(self):
        """Test that analyze pipeline has expected nodes."""
        analyze = pipelines.get("analyze")
        assert analyze is not None
        assert len(analyze.nodes) == 4
        node_names = [n.name for n in analyze.nodes]
        assert "load_analysis_data" in node_names
        assert "analyze_features" in node_names

    def test_train_pipeline_has_nodes(self):
        """Test that train pipeline has expected nodes."""
        train = pipelines.get("train")
        assert train is not None
        assert len(train.nodes) == 4
        node_names = [n.name for n in train.nodes]
        assert "load_training_data" in node_names
        assert "train_validator" in node_names

    def test_tune_pipeline_has_nodes(self):
        """Test that tune pipeline has expected nodes."""
        tune = pipelines.get("tune")
        assert tune is not None
        assert len(tune.nodes) == 3
        node_names = [n.name for n in tune.nodes]
        assert "run_optimization" in node_names

    def test_validate_pipeline_has_nodes(self):
        """Test that validate pipeline has expected nodes."""
        validate = pipelines.get("validate")
        assert validate is not None
        assert len(validate.nodes) == 3
        node_names = [n.name for n in validate.nodes]
        assert "run_walk_forward" in node_names


class TestFlowConfig:
    """Test flow configuration loading."""

    def test_list_flows(self):
        """Test listing available flows."""
        from sf_kedro.utils.flow_config import list_flows

        flows = list_flows()
        assert "grid_sma" in flows
        assert "baseline_sma" in flows

    def test_load_flow_config(self):
        """Test loading flow config."""
        from sf_kedro.utils.flow_config import load_flow_config

        config = load_flow_config("grid_sma")
        assert config["flow_id"] == "grid_sma"
        assert "detector" in config
        assert "strategy" in config

    def test_deep_merge(self):
        """Test deep merge utility."""
        from sf_kedro.utils.flow_config import deep_merge

        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 10, "d": 3}, "e": 5}
