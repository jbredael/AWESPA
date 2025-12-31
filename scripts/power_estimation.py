#!/usr/bin/env python3
"""
AWE Power Estimation Test Script

This script tests the AWE power estimation wrapper by computing
power curves for wind resource clusters.

Usage:
    python scripts/test_power_estimation.py

Author: AWESPA Development Team
Date: December 2025
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.power.awe_power import AWEPowerEstimationModel


def test_power_estimation():
    """Test AWE power estimation and generate power curves."""
    
    # Define paths
    config_dir = PROJECT_ROOT / "config"
    results_path = PROJECT_ROOT / "results"
    output_file = results_path / "power_curves.yml"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    print("AWE Power Estimation Test")
    print("=" * 40)
    
    # Initialize power estimation model
    print("Initializing power estimation model...")
    power_model = AWEPowerEstimationModel()
    
    # Load configuration
    print("Loading configuration files...")
    try:
        power_model.load_configuration(config_dir)
        print(f"✓ Configuration loaded from {config_dir}")
        print(f"  - Kite mass: {power_model.airborne_config['mass']} kg")
        print(f"  - Kite area: {power_model.airborne_config['projected_area']} m²")
        print(f"  - Tether length: {power_model.tether_config['length']} m")
        print(f"  - Wind clusters: {len(power_model.wind_resource['clusters'])}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False
    
    # Compute power curves
    print("\\nComputing power curves...")
    try:
        power_model.compute_power_curves(output_file)
        print(f"✓ Power curves computed successfully")
        print(f"✓ Results saved to {output_file}")
    except Exception as e:
        print(f"✗ Error computing power curves: {e}")
        print("Note: This requires the AWE_production_estimation vendor to be properly installed")
        return False
    
    # Load and display results
    print("\\nAnalyzing power curve results...")
    try:
        display_power_curve_results(output_file)
        plot_power_curves(output_file, save_path=results_path / "power_curves.png")
    except Exception as e:
        print(f"✗ Error analyzing results: {e}")
        return False
    
    print("\\n" + "=" * 40)
    print("Power estimation test completed!")
    return True


def display_power_curve_results(yaml_file_path: Path):
    """Display power curve analysis results."""
    
    with open(yaml_file_path, 'r') as f:
        power_data = yaml.safe_load(f)
    
    metadata = power_data['metadata']
    power_curves = power_data['power_curves']
    
    print("Power Curve Analysis Results:")
    print("-" * 30)
    print(f"System: {metadata['system_configuration']['kite_mass']} kg kite, {metadata['system_configuration']['kite_area']} m² area")
    print(f"Clusters processed: {metadata['clusters_processed']}")
    print(f"Wind data source: {metadata['wind_profile_source']}")
    
    print("\\nPower curves by cluster:")
    for cluster_name, curve_data in power_curves.items():
        cluster_id = curve_data['cluster_id'] 
        frequency = curve_data['frequency']
        wind_speeds = curve_data['wind_speeds']
        powers = curve_data['power_outputs']
        
        if powers:
            max_power = max(powers)
            rated_wind = wind_speeds[powers.index(max_power)]
            print(f"  {cluster_name} (ID: {cluster_id}, {frequency:.1f}%): "
                  f"Max {max_power/1000:.1f} kW at {rated_wind:.1f} m/s")
        else:
            print(f"  {cluster_name} (ID: {cluster_id}, {frequency:.1f}%): No successful computations")


def plot_power_curves(yaml_file_path: Path, save_path: Path = None):
    """Generate power curve visualizations."""
    
    with open(yaml_file_path, 'r') as f:
        power_data = yaml.safe_load(f)
    
    power_curves = power_data['power_curves']
    
    # Create power curve plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('AWE Power Curves by Wind Profile Cluster', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(power_curves)))
    
    # Plot 1: Individual power curves
    for i, (cluster_name, curve_data) in enumerate(power_curves.items()):
        cluster_id = curve_data['cluster_id']
        frequency = curve_data['frequency']
        wind_speeds = curve_data['wind_speeds']
        powers = np.array(curve_data['power_outputs']) / 1000  # Convert to kW
        
        if len(wind_speeds) > 0:
            ax1.plot(wind_speeds, powers, 'o-', color=colors[i], linewidth=2,
                    label=f'Cluster {cluster_id} ({frequency:.1f}%)')
    
    ax1.set_xlabel('Wind Speed [m/s]')
    ax1.set_ylabel('Power Output [kW]')
    ax1.set_title('Power Curves by Cluster')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Frequency-weighted power curve
    # Combine all curves weighted by frequency
    combined_wind_speeds = np.linspace(4, 20, 50)
    combined_power = np.zeros_like(combined_wind_speeds)
    
    for cluster_name, curve_data in power_curves.items():
        frequency = curve_data['frequency'] / 100  # Convert percentage to fraction
        wind_speeds = curve_data['wind_speeds']
        powers = np.array(curve_data['power_outputs']) / 1000  # Convert to kW
        
        if len(wind_speeds) > 1:
            # Interpolate power curve to common wind speed grid
            interpolated_power = np.interp(combined_wind_speeds, wind_speeds, powers, left=0, right=0)
            combined_power += frequency * interpolated_power
    
    ax2.plot(combined_wind_speeds, combined_power, 'k-', linewidth=3, label='Frequency-weighted')
    ax2.set_xlabel('Wind Speed [m/s]')
    ax2.set_ylabel('Expected Power Output [kW]')
    ax2.set_title('Frequency-Weighted Power Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Power curve plots saved to {save_path}")
    
    plt.show()


def main():
    """Main function."""
    
    try:
        success = test_power_estimation()
        if not success:
            print("Test failed.")
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()