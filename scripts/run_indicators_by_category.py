#!/usr/bin/env python3
"""
Script to run feature_analysis pipeline with indicators from specific category.

Categories: momentum, volatility, volume, trend, overlap, stat, divergence
"""

import sys
from pathlib import Path
from typing import List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main script functions
from run_all_indicators import (
    get_all_indicator_classes,
    get_test_params,
    get_output_feature_name,
    update_parameters_file,
    run_feature_analysis_pipeline,
    read_latest_stats,
    generate_consolidated_report,
)
import time
from datetime import datetime
import yaml


def list_categories():
    """List all available categories with indicator counts."""
    indicators = get_all_indicator_classes()

    categories = {}
    for registry_name, _ in indicators:
        if "/" in registry_name:
            category = registry_name.split("/")[0]
        else:
            category = "other"

        if category not in categories:
            categories[category] = []
        categories[category].append(registry_name)

    print("📊 Available Categories:")
    print("=" * 80)
    for category, names in sorted(categories.items()):
        print(f"  {category:20s} - {len(names):3d} indicators")
    print("=" * 80)
    print(f"\nTotal: {len(indicators)} indicators")

    return categories


def filter_by_category(category: str) -> List[tuple]:
    """
    Filter indicators by category.

    Args:
        category: Category name (e.g., 'momentum', 'volatility')

    Returns:
        List of (registry_name, class_obj) tuples
    """
    all_indicators = get_all_indicator_classes()

    filtered = []
    for registry_name, class_obj in all_indicators:
        if registry_name.startswith(f"{category}/"):
            filtered.append((registry_name, class_obj))

    return filtered


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python3 run_indicators_by_category.py <category>")
        print("       python3 run_indicators_by_category.py --list")
        print()
        print(
            "Categories: momentum, volatility, volume, trend, overlap, stat, divergence"
        )
        sys.exit(1)

    if sys.argv[1] == "--list":
        list_categories()
        sys.exit(0)

    category = sys.argv[1]

    print("=" * 80)
    print(f"🚀 Running Feature Analysis Pipeline - {category.upper()} Indicators")
    print("=" * 80)
    print()

    # Get filtered indicators
    print(f"📊 Filtering {category} indicators...")
    indicators = filter_by_category(category)

    if not indicators:
        print(f"❌ No indicators found for category: {category}")
        print("\nUse --list to see all available categories")
        sys.exit(1)

    print(f"✅ Found {len(indicators)} indicators\n")
    print("Indicators to process:")
    for registry_name, _ in indicators:
        print(f"  • {registry_name}")
    print()

    # Track results and statistics
    results = {"category": category, "successful": [], "failed": [], "skipped": []}
    all_stats = []  # Collect statistics for consolidated report

    # Process each indicator
    for i, (registry_name, class_obj) in enumerate(indicators, 1):
        print(f"\n[{i}/{len(indicators)}] Processing: {registry_name}")
        print("-" * 80)

        try:
            # Get test parameters
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
                results["successful"].append(
                    {"registry_name": registry_name, "params": params, "time": elapsed}
                )

                # Read and collect statistics
                stats = read_latest_stats()
                if stats:
                    all_stats.append(stats)
                    print("  📊 Statistics collected")
            else:
                print("  ❌ Failed!")
                results["failed"].append(
                    {"registry_name": registry_name, "params": params}
                )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["failed"].append({"registry_name": registry_name, "error": str(e)})

    # Print summary
    print("\n" + "=" * 80)
    print(f"📈 SUMMARY - {category.upper()}")
    print("=" * 80)
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    print()

    if results["successful"]:
        print("Successful indicators:")
        for item in results["successful"]:
            print(f"  • {item['registry_name']} - {item['time']:.1f}s")

    if results["failed"]:
        print("\nFailed indicators:")
        for item in results["failed"]:
            print(f"  • {item['registry_name']}")
            if "error" in item:
                print(f"    Error: {item['error']}")

    # Save results to file
    results_file = Path(
        f"indicator_analysis_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    with open(results_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\n💾 Results saved to: {results_file}")

    # Generate consolidated statistics report
    if all_stats:
        report_file = Path(
            f"indicator_analysis_{category}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        print(f"\n📊 Generating consolidated report for {category}...")
        generate_consolidated_report(all_stats, report_file)
        print(f"💾 Consolidated report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("🏁 Done!")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
