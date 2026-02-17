#!/usr/bin/env python3
"""Simple script to run Luchsinger power model and plot results.

Usage:
    python scripts/run_luchsinger.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.power.luchsinger_power import LuchsingerPowerModel


def main():
    """Run Luchsinger model and plot."""
    # Define config paths
    config_dir = PROJECT_ROOT / "config" / "meridional_case_1"
    system_path = config_dir / "soft_kite_pumping_ground_gen_system.yml"
    settings_path = config_dir / "Lucsinger_simulation_settings_config.yml"
    
    # Define wind resource and results paths
    results_dir = PROJECT_ROOT / "results"
    wind_resource_path = results_dir / "shear_gaming" / "wind_resource.yml"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = results_dir / "luchsinger_power_curves.yml"
    
    # Initialize and load model
    model = LuchsingerPowerModel()
    
    model.load_configuration(
        systemPath=system_path,
        simulationSettingsPath=settings_path,
        operationalConstraintsPath=None,
        windResourcePath=wind_resource_path
    )

    # Test single wind speed calculation
    print("Testing single wind speed calculation:")
    model.calculate_power_at_wind_speed(windSpeed=12.0)
    
    # Compute power curves and export
    print("\nComputing power curves and exporting to YAML...")
    model.compute_power_curves(outputPath=output_yaml)
    
    # Generate plots using wrapper method
    print("\nGenerating plots...")
    try:
        model.plot_power_curves(powerCurvePath=output_yaml, outputDir=results_dir)
        print("\n" + "="*60)
        print("Analysis complete! Plots are displayed and saved.")
        print("="*60)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
