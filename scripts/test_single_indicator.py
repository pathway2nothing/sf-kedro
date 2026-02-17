#!/usr/bin/env python3
"""
Test script to verify the indicator analysis pipeline works with a single indicator.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main script functions
from run_all_indicators import (
    get_all_indicator_classes,
    get_output_feature_name,
    get_test_params,
    run_feature_analysis_pipeline,
    update_parameters_file,
)


def main():
    print("=" * 80)
    print("🧪 Testing Feature Analysis Pipeline with Single Indicator")
    print("=" * 80)
    print()

    # Get all indicators
    print("📊 Discovering indicators...")
    indicators = get_all_indicator_classes()
    print(f"✅ Found {len(indicators)} indicators\n")

    if not indicators:
        print("❌ No indicators found!")
        sys.exit(1)

    # Test with first indicator (usually RsiMom)
    registry_name, class_obj = indicators[0]

    print(f"🎯 Testing with: {registry_name}")
    print("-" * 80)

    # Get test parameters
    params = get_test_params(class_obj)
    print(f"Parameters: {params}")

    # Get expected output feature name
    output_name = get_output_feature_name(class_obj, params)
    print(f"Output feature: {output_name}")

    # Update parameters file
    print("\n📝 Updating configuration...")
    update_parameters_file(registry_name, params, output_name)
    print("✅ Configuration updated")

    # Show what was written
    params_file = Path("conf/base/parameters/feature_analysis.yml")
    print(f"\n📄 Updated config file ({params_file}):")
    print("-" * 80)
    with open(params_file) as f:
        print(f.read())
    print("-" * 80)

    # Run pipeline
    print("\n⚙️  Running pipeline...")
    success = run_feature_analysis_pipeline()

    if success:
        print("\n✅ Pipeline completed successfully!")

        # Check output
        output_dir = Path("data/08_reporting/feature_analysis/feature_analysis")
        if output_dir.exists():
            plots = list(output_dir.glob("*.png"))
            print(f"\n📊 Generated {len(plots)} plots in {output_dir}")
            for plot in plots:
                print(f"  • {plot.name}")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🏁 Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
