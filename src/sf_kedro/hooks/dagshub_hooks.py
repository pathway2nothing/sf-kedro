"""DagsHub integration hooks."""

from kedro.framework.hooks import hook_impl
import dagshub
import mlflow
import os
from pathlib import Path
from typing import Any, Dict


class DagsHubHook:
    """Hook for DagsHub MLflow integration."""
    
    def __init__(self):
        self.dagshub_repo = os.getenv("DAGSHUB_REPO")
        
    @hook_impl
    def before_pipeline_run(
        self, 
        run_params: Dict[str, Any], 
        pipeline, 
        catalog
    ):
        """Initialize DagsHub and MLflow tracking."""
        
        if not self.dagshub_repo:
            print("Warning: DAGSHUB_REPO not set, skipping DagsHub initialization")
            return
        
        repo_owner, repo_name = self.dagshub_repo.split("/")
        
        dagshub.init(
            repo_name=repo_name,
            repo_owner=repo_owner,
            mlflow=True
        )
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        pipeline_name = run_params.get("pipeline_name", "default")
        
        experiment_name = f"signalflow_{pipeline_name}"
        mlflow.set_experiment(experiment_name)
        
        run_id = run_params.get("run_id", "manual")
        run_name = f"{pipeline_name}_{run_id}"
        
        mlflow.start_run(run_name=run_name)
        
        try:
            import git
            repo = git.Repo(Path.cwd())
            mlflow.set_tags({
                "git.commit": repo.head.commit.hexsha[:7],
                "git.branch": repo.active_branch.name,
                "git.is_dirty": str(repo.is_dirty()),
            })
        except Exception as e:
            print(f"Could not log git info: {e}")
        
        mlflow.set_tags({
            "kedro.pipeline": pipeline_name,
            "kedro.run_id": run_id,
        })
        
        print(f"✓ MLflow tracking initialized: {experiment_name}")
        print(f"✓ Run: {run_name}")
    
    @hook_impl
    def after_pipeline_run(
        self, 
        run_params: Dict[str, Any], 
        pipeline, 
        catalog
    ):
        """Finalize MLflow run."""
        try:
            mlflow.set_tag("status", "success")
            mlflow.end_run()
            print("✓ MLflow run completed successfully")
        except Exception as e:
            print(f"Error ending MLflow run: {e}")
    
    @hook_impl
    def on_pipeline_error(
        self, 
        error: Exception, 
        run_params: Dict[str, Any], 
        pipeline, 
        catalog
    ):
        """Log error and mark run as failed."""
        try:
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(error))
            mlflow.end_run(status="FAILED")
            print(f"✗ MLflow run failed: {error}")
        except Exception as e:
            print(f"Error logging failure: {e}")