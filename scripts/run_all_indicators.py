#!/usr/bin/env python3
"""
Script to run feature_analysis pipeline with each indicator from signalflow-ta.

This script iterates through all available technical indicators from signalflow-ta
and runs the feature analysis pipeline with each one using default parameters.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
import time
from datetime import datetime

# Import signalflow to get all indicators
import signalflow as sf
import signalflow.ta


def get_all_indicator_classes() -> List[tuple]:
    """
    Get all indicator classes from signalflow registry.

    Returns:
        List of (registry_name, class_obj) tuples
    """
    indicators = []

    # Get all registered features from the registry
    feature_names = sf.default_registry.list(component_type=sf.SfComponentType.FEATURE)

    for registry_name in feature_names:
        try:
            # Skip example features (they are just for testing)
            if registry_name.startswith('example/'):
                continue

            # Skip indicators without category (prefer categorized versions)
            # e.g., prefer "volatility/atr" over "atr"
            if '/' not in registry_name:
                continue

            # Get class from registry
            class_obj = sf.default_registry.get(
                component_type=sf.SfComponentType.FEATURE,
                name=registry_name
            )

            indicators.append((registry_name, class_obj))
        except Exception as e:
            print(f"❌ Error getting {registry_name}: {e}")
            continue

    return indicators


def get_test_params(class_obj) -> Dict[str, Any]:
    """
    Extract first test parameters from class test_params field.

    Args:
        class_obj: Indicator class

    Returns:
        Dictionary with parameter names and values for first test
    """
    # Try to get test_params field
    if hasattr(class_obj, '__dataclass_fields__'):
        test_params_field = class_obj.__dataclass_fields__.get('test_params')
        if test_params_field and test_params_field.default:
            test_params_list = test_params_field.default
            if test_params_list and len(test_params_list) > 0:
                # Return last test params
                return test_params_list[-1]

    # Fallback: try to get minimal default params
    params = {}
    if hasattr(class_obj, '__dataclass_fields__'):
        for field_name, field in class_obj.__dataclass_fields__.items():
            # Skip special fields
            if field_name in ['requires', 'outputs', 'test_params', 'component_type',
                            'group_col', 'ts_col', 'normalized', 'norm_period']:
                continue

            # Get default value
            if field.default is not field.default_factory:
                params[field_name] = field.default

    return params


def create_feature_config(registry_name: str, params: Dict[str, Any]) -> Dict:
    """
    Create feature configuration for kedro parameters.

    Args:
        registry_name: Registry name of the indicator (e.g., "momentum/rsi")
        params: Dictionary of parameters

    Returns:
        Feature configuration dict
    """
    config = {
        "type": registry_name,
        **params
    }
    return config


def update_parameters_file(registry_name: str, params: Dict[str, Any],
                          output_feature_name: str) -> None:
    """
    Update the feature_analysis.yml parameters file with new indicator config.

    Args:
        registry_name: Registry name of the indicator
        params: Parameters for the indicator
        output_feature_name: Name of the output feature for analysis
    """
    params_file = Path("conf/base/parameters/feature_analysis.yml")

    # Read current parameters
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update feature extractors
    config['feature_analysis']['features']['extractors'] = [
        create_feature_config(registry_name, params)
    ]

    # Update analysis config with the output feature name and indicator type
    config['feature_analysis']['analysis']['feature_name'] = output_feature_name
    config['feature_analysis']['analysis']['indicator_type'] = registry_name

    # Write back
    with open(params_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_feature_analysis_pipeline() -> bool:
    """
    Run the feature_analysis pipeline using kedro.

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["kedro", "run", "--pipeline", "feature_analysis"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            print(f"❌ Pipeline failed with exit code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("❌ Pipeline timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False


def get_output_feature_name(class_obj, params: Dict[str, Any]) -> str:
    """
    Generate the expected output feature name from the indicator class.
    For indicators with multiple outputs, returns the primary feature name.

    Args:
        class_obj: Indicator class
        params: Parameters used

    Returns:
        Expected output column name
    """
    # Try to instantiate and get output name
    try:
        instance = class_obj(**params)
        if hasattr(instance, '_get_output_name'):
            return instance._get_output_name()
        elif hasattr(instance, 'outputs') and instance.outputs:
            # For divergence indicators, prefer the strength column if available
            if 'divergence' in class_obj.__name__.lower():
                for output in instance.outputs:
                    if 'strength' in output.lower():
                        # Format with parameters
                        output_name = output
                        for key, value in params.items():
                            output_name = output_name.replace(f"{{{key}}}", str(value))
                        return output_name

            # Default: use first output and format with parameters
            output_template = instance.outputs[0]
            for key, value in params.items():
                output_template = output_template.replace(f"{{{key}}}", str(value))
            return output_template
    except:
        pass

    # Fallback: use class name in lowercase
    return class_obj.__name__.lower().replace('mom', '').replace('smooth', '').replace('vol', '').replace('volume', '').replace('trend', '').replace('stat', '').replace('price', '')


def read_latest_stats() -> Dict[str, Any]:
    """
    Read the latest statistics file generated by the pipeline.

    Returns:
        Dictionary with comprehensive statistics or empty dict if file not found
    """
    stats_file = Path("data/08_reporting/feature_analysis/latest_stats.json")

    if not stats_file.exists():
        return {}

    try:
        with open(stats_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to read stats file: {e}")
        return {}


def generate_consolidated_report(all_stats: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Generate a consolidated text report for all indicators.

    Args:
        all_stats: List of statistics dictionaries from all indicators
        output_file: Path to save the report
    """
    if not all_stats:
        print("No statistics to report")
        return

    lines = []

    # Header
    lines.append("=" * 120)
    lines.append("📊 INDICATOR ANALYSIS CONSOLIDATED REPORT")
    lines.append("=" * 120)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Indicators: {len(all_stats)}")
    lines.append("")

    # Get period info from first indicator (assuming all use same period)
    if all_stats[0].get('global_stats'):
        first_stats = all_stats[0]
        lines.append(f"Period: {first_stats.get('timestamp', 'N/A')}")
        lines.append("")

    # Table header
    lines.append("=" * 120)
    header = f"{'INDICATOR':<35} | {'MEAN':>10} | {'STD':>10} | {'SKEW':>7} | {'KURT':>7} | {'CORR_AVG':>10} | {'VALID%':>8} | {'TRANSFORM?':<12}"
    lines.append(header)
    lines.append("=" * 120)

    # Sort by indicator type
    sorted_stats = sorted(all_stats, key=lambda x: x.get('indicator_type', ''))

    # Table rows
    for stat in sorted_stats:
        indicator = stat.get('indicator_type', 'Unknown')[:35]
        gs = stat.get('global_stats', {})
        mean = gs.get('mean', 0.0)
        std = gs.get('std', 0.0)
        skew = gs.get('skewness', 0.0)
        kurt = gs.get('kurtosis', 0.0)
        corr_avg = stat.get('avg_correlation')
        corr_str = f"{corr_avg:+.4f}" if corr_avg is not None else "N/A"
        valid_pct = stat.get('data_quality', {}).get('avg_valid_pct', 0.0)
        transform = "Yes" if stat.get('transform_needed', False) else "No"

        # Add transform reason if applicable
        if stat.get('transform_needed'):
            reasons = stat.get('transform_reasons', [])
            if reasons:
                transform = f"Yes ({reasons[0][:20]}...)" if len(reasons[0]) > 20 else f"Yes ({reasons[0]})"

        row = f"{indicator:<35} | {mean:>10.6f} | {std:>10.6f} | {skew:>7.3f} | {kurt:>7.3f} | {corr_str:>10} | {valid_pct:>7.2f}% | {transform:<12}"
        lines.append(row)

    lines.append("=" * 120)
    lines.append("")

    # Summary statistics
    lines.append("📈 SUMMARY STATISTICS")
    lines.append("-" * 120)

    # Count indicators needing transformation
    transform_needed = sum(1 for s in all_stats if s.get('transform_needed', False))
    lines.append(f"Indicators needing transformation: {transform_needed} / {len(all_stats)} ({transform_needed/len(all_stats)*100:.1f}%)")

    # Average correlation
    all_corrs = [s.get('avg_correlation') for s in all_stats if s.get('avg_correlation') is not None]
    if all_corrs:
        avg_all_corr = sum(all_corrs) / len(all_corrs)
        lines.append(f"Average correlation across all indicators: {avg_all_corr:+.4f}")

    # Average data quality
    all_valid_pcts = [s.get('data_quality', {}).get('avg_valid_pct', 0) for s in all_stats]
    if all_valid_pcts:
        avg_valid = sum(all_valid_pcts) / len(all_valid_pcts)
        lines.append(f"Average data quality: {avg_valid:.2f}%")

    lines.append("")
    lines.append("=" * 120)

    # Write to file
    report_text = "\n".join(lines)
    with open(output_file, 'w') as f:
        f.write(report_text)

    # Also print to console
    print("\n" + report_text)


def main():
    """Main execution function."""
    print("=" * 80)
    print("🚀 Running Feature Analysis Pipeline with All Indicators")
    print("=" * 80)
    print()

    # Get all indicators
    print("📊 Discovering indicators from signalflow-ta...")
    indicators = get_all_indicator_classes()
    print(f"✅ Found {len(indicators)} indicators\n")

    # Track results and statistics
    results = {
        "successful": [],
        "failed": [],
        "skipped": []
    }
    all_stats = []  # Collect statistics for consolidated report

    # Process each indicator
    for i, (registry_name, class_obj) in enumerate(indicators, 1):
        print(f"\n[{i}/{len(indicators)}] Processing: {registry_name}")
        print("-" * 80)

        try:
            # Get test parameters (first test from test_params)
            params = get_test_params(class_obj)
            print(f"  Parameters: {params}")

            # Get expected output feature name
            output_name = get_output_feature_name(class_obj, params)
            print(f"  Output feature: {output_name}")

            # Update parameters file
            print("  📝 Updating configuration...")
            update_parameters_file(registry_name, params, output_name)

            # Run pipeline
            print("  ⚙️  Running pipeline...")
            start_time = time.time()
            success = run_feature_analysis_pipeline()
            elapsed = time.time() - start_time

            if success:
                print(f"  ✅ Success! (took {elapsed:.1f}s)")
                results["successful"].append({
                    "registry_name": registry_name,
                    "params": params,
                    "time": elapsed
                })

                # Read and collect statistics
                stats = read_latest_stats()
                if stats:
                    all_stats.append(stats)
                    print(f"  📊 Statistics collected")
            else:
                print(f"  ❌ Failed!")
                results["failed"].append({
                    "registry_name": registry_name,
                    "params": params
                })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["failed"].append({
                "registry_name": registry_name,
                "error": str(e)
            })

    # Print summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    print()

    if results['successful']:
        print("Successful indicators:")
        for item in results['successful']:
            print(f"  • {item['registry_name']} - {item['time']:.1f}s")

    if results['failed']:
        print("\nFailed indicators:")
        for item in results['failed']:
            print(f"  • {item['registry_name']}")
            if 'error' in item:
                print(f"    Error: {item['error']}")

    # Save results to file
    results_file = Path(f"indicator_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\n💾 Results saved to: {results_file}")

    # Generate consolidated statistics report
    if all_stats:
        report_file = Path(f"indicator_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        print(f"\n📊 Generating consolidated report...")
        generate_consolidated_report(all_stats, report_file)
        print(f"💾 Consolidated report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("🏁 Done!")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if not results['failed'] else 1)


if __name__ == "__main__":
    main()
