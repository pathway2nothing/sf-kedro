"""Debug script for running Kedro pipelines locally with full error details.

Usage:
    python debug.py
    python debug.py --pipeline ml_validated
    python debug.py --node load_data
"""

import sys
from pathlib import Path
import os


# Find the Kedro project root (where pyproject.toml is located)
def find_kedro_project_root():
    """Find the Kedro project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent

    # Check current directory first
    if (current / "pyproject.toml").exists():
        return current

    # Check if we're in a subdirectory of the project
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # If not found, assume we're running from project root
    return current


project_root = find_kedro_project_root()

# Verify pyproject.toml exists
if not (project_root / "pyproject.toml").exists():
    print(f"ERROR: Could not find pyproject.toml in {project_root}")
    print(f"Current directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    print("\nPlease run this script from the Kedro project directory.")
    sys.exit(1)

# Change to project root
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Working directory: {Path.cwd()}")

import logging
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Configure logging for detailed output
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_pipeline(
    pipeline_name: str = "baseline", node_names: str = None, env: str = "local"
):
    """Run Kedro pipeline with full debugging enabled.

    Args:
        pipeline_name: Name of the pipeline to run (default: "baseline")
        node_names: Comma-separated node names to run specific nodes only
        env: Environment to use (default: "local")
    """
    print(f"\n{'=' * 60}")
    print(f"Running pipeline: {pipeline_name}")
    print(f"Environment: {env}")
    if node_names:
        print(f"Specific nodes: {node_names}")
    print(f"{'=' * 60}\n")

    try:
        metadata = bootstrap_project(project_root)

        with KedroSession.create(project_path=project_root, env=env) as session:
            if node_names:
                nodes = [n.strip() for n in node_names.split(",")]
                session.run(pipeline_name=pipeline_name, node_names=nodes)
            else:
                session.run(pipeline_name=pipeline_name)

            print(f"\n{'=' * 60}")
            print("Pipeline completed successfully!")
            print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("PIPELINE FAILED WITH ERROR:")
        print(f"{'=' * 60}\n")

        import traceback

        traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("Error summary:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"{'=' * 60}\n")

        sys.exit(1)


if __name__ == "__main__":
    run_pipeline(pipeline_name="ml_validated")
