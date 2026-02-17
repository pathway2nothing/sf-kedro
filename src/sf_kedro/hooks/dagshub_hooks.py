"""DagsHub integration hooks."""

import os
from pathlib import Path
from typing import Any

from kedro.framework.hooks import hook_impl


class DagsHubHook:
    """Hook for DagsHub MLflow integration."""

    def __init__(self):
        self.dagshub_repo = os.getenv("DAGSHUB_REPO")
        # Check if MLflow is enabled (default: True if DAGSHUB_REPO is set)
        mlflow_enabled = os.getenv("MLFLOW_ENABLED", "true").lower()
        self.mlflow_enabled = mlflow_enabled in ("true", "1", "yes")
        self.mlflow_available = False

    def _is_mlflow_enabled(self) -> bool:
        """Check if MLflow should be used."""
        if not self.mlflow_enabled:
            return False
        return self.dagshub_repo

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any], pipeline, catalog):
        """Initialize DagsHub and MLflow tracking."""

        # Skip if MLflow is disabled or repo not configured
        if not self._is_mlflow_enabled():
            if not self.dagshub_repo:
                print("[info] MLflow: DAGSHUB_REPO not set, skipping")
            else:
                print("[info] MLflow: Disabled via MLFLOW_ENABLED=false")
            return

        try:
            # Import here to make dependencies optional
            import dagshub
            import mlflow

            repo_owner, repo_name = self.dagshub_repo.split("/")

            dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            pipeline_name = run_params.get("pipeline_name", "default")

            experiment_name = f"signalflow_{pipeline_name}"
            mlflow.set_experiment(experiment_name)

            run_id = run_params.get("run_id", "manual")
            run_name = f"{pipeline_name}_{run_id}"

            active_run = mlflow.active_run()
            if active_run is None:
                mlflow.start_run(run_name=run_name)

            try:
                import git

                repo = git.Repo(Path.cwd())
                mlflow.set_tags(
                    {
                        "git.commit": repo.head.commit.hexsha[:7],
                        "git.branch": repo.active_branch.name,
                        "git.is_dirty": str(repo.is_dirty()),
                    }
                )
            except Exception as e:
                print(f"⚠️  MLflow: Could not log git info: {e}")

            mlflow.set_tags(
                {
                    "kedro.pipeline": pipeline_name,
                    "kedro.run_id": run_id,
                }
            )

            self.mlflow_available = True
            print(f"✓ MLflow tracking initialized: {experiment_name}")
            print(f"✓ Run: {run_name}")

        except ImportError as e:
            print(f"⚠️  MLflow: Dependencies not available ({e}), continuing without tracking")
            self.mlflow_available = False
        except Exception as e:
            print(f"⚠️  MLflow: Failed to initialize ({e}), continuing without tracking")
            self.mlflow_available = False

    @hook_impl
    def after_pipeline_run(self, run_params: dict[str, Any], pipeline, catalog):
        """Finalize MLflow run."""
        if not self.mlflow_available:
            return

        try:
            import mlflow

            if not hasattr(self, "_used_existing_run") or not self._used_existing_run:
                mlflow.set_tag("status", "success")
                mlflow.end_run()
                print("✓ MLflow run completed successfully")
        except Exception as e:
            print(f"⚠️  MLflow: Error ending run: {e}")

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: dict[str, Any], pipeline, catalog):
        """Log error and mark run as failed."""
        if not self.mlflow_available:
            return

        try:
            import mlflow

            if not hasattr(self, "_used_existing_run") or not self._used_existing_run:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(error))
                mlflow.end_run(status="FAILED")
                print(f"✗ MLflow run failed: {error}")
        except Exception as e:
            print(f"⚠️  MLflow: Error logging failure: {e}")
