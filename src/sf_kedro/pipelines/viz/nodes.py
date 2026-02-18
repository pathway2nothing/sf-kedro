"""Visualization pipeline nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loguru import logger

from sf_kedro.utils.flow_config import config_to_flow, load_flow_config


def generate_flow_dag_visualization(
    flow_id: str,
    output_dir: str | Path = "data/08_reporting/viz",
    output_format: Literal["html", "mermaid"] = "mermaid",
) -> dict[str, Any]:
    """Generate DAG visualization for a flow using Flow API.

    This uses the Flow (DAG) representation for visualization.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma')
        output_dir: Directory to save visualization
        output_format: Output format ('html' or 'mermaid')

    Returns:
        Dict with output path and metadata
    """
    # Load flow config and convert to Flow
    config = load_flow_config(flow_id)
    flow = config_to_flow(config)

    logger.info(f"Generating DAG visualization for flow: {flow_id}")

    # Compile to get execution order
    flow.compile()
    plan = flow.plan()

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate Mermaid diagram
    mermaid_code = _generate_mermaid_from_flow(flow)

    if output_format == "mermaid":
        output_file = output_path / f"{flow_id}_dag.md"
        output_file.write_text(f"```mermaid\n{mermaid_code}\n```\n")
    else:
        # Generate HTML with embedded Mermaid
        output_file = output_path / f"{flow_id}_dag.html"
        html_content = _generate_html_with_mermaid(flow_id, mermaid_code)
        output_file.write_text(html_content)

    logger.success(f"DAG visualization saved to: {output_file}")

    return {
        "flow_id": flow_id,
        "output_path": str(output_file),
        "format": output_format,
        "node_count": len(flow.nodes),
        "edge_count": len(flow.dependencies),
        "execution_order": [p["id"] for p in plan],
    }


def _generate_mermaid_from_flow(flow) -> str:
    """Generate Mermaid diagram from Flow."""
    lines = ["graph TD"]

    # Add nodes with styling
    for node_id, node in flow.nodes.items():
        label = f"{node_id}\\n{node.type}"
        if node.name:
            label = f"{node_id}\\n{node.name}"

        # Style based on node type
        if node.type == "data/loader":
            lines.append(f'    {node_id}[("{label}")]')
        elif node.type == "signals/detector":
            lines.append(f"    {node_id}[/{label}/]")
        elif node.type == "strategy":
            lines.append(f"    {node_id}[[{label}]]")
        else:
            lines.append(f"    {node_id}[{label}]")

    # Add edges
    for dep in flow.dependencies:
        label = dep.artifact or ""
        if label:
            lines.append(f"    {dep.source} -->|{label}| {dep.target}")
        else:
            lines.append(f"    {dep.source} --> {dep.target}")

    # Add styling
    lines.extend(
        [
            "",
            "    classDef loader fill:#e1f5fe,stroke:#01579b",
            "    classDef detector fill:#fff3e0,stroke:#e65100",
            "    classDef strategy fill:#e8f5e9,stroke:#1b5e20",
        ]
    )

    # Apply classes
    for node_id, node in flow.nodes.items():
        if node.type == "data/loader":
            lines.append(f"    class {node_id} loader")
        elif node.type == "signals/detector":
            lines.append(f"    class {node_id} detector")
        elif node.type == "strategy":
            lines.append(f"    class {node_id} strategy")

    return "\n".join(lines)


def _generate_html_with_mermaid(flow_id: str, mermaid_code: str) -> str:
    """Generate HTML page with embedded Mermaid diagram."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Flow: {flow_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>Flow: {flow_id}</h1>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
