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
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = results_dir / "luchsinger_power_curves.yml"
    
    # Initialize and load model
    model = LuchsingerPowerModel()
    model.load_configuration(system_path, settings_path, operational_constraints_path=None)
    
    # Test single wind speed calculation
    print("Testing single wind speed calculation:")
    model.calculate_power_at_wind_speed(wind_speed=12.0, plot=True)
    
    # Compute power curves, export, and plot
    print("\nComputing power curves, exporting to YAML, and plotting...")
    model.compute_power_curves(output_path=output_yaml, plot=True)


if __name__ == "__main__":
    main()
