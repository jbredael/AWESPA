#!/usr/bin/env python3
"""
Luchsinger AEP Calculation Script

This script uses the Luchsinger power model to:
1. Load wind resource data from a YAML file
2. Generate power curves for each wind profile cluster
3. Calculate Annual Energy Production (AEP) using the AWESPA pipeline

Usage:
    python scripts/run_luchsinger_aep.py

Author: AWESPA Development Team
Date: December 2025
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.power.luchsinger_power import LuchsingerPowerModel


def calculate_aep_from_power_curves(power_curve_data: dict, 
                                     wind_resource_data: dict) -> dict:
    """Calculate AEP from power curves and wind resource probability data.
    
    Args:
        power_curve_data (dict): Power curve data from Luchsinger model.
        wind_resource_data (dict): Wind resource with probability matrix.
        
    Returns:
        dict: AEP calculation results.
    """
    # Hours per year
    HOURS_PER_YEAR = 8760
    
    # Get probability matrix (clusters x wind_speed_bins)
    probability_matrix = np.array(wind_resource_data['probability_matrix']['data'])
    
    # Get wind speed bin centers
    bin_centers = np.array(power_curve_data['wind_speed_bins']['bin_centers_m_s'])
    
    # Calculate AEP for each cluster
    cluster_aep = []
    total_aep_wh = 0.0
    
    for i, curve in enumerate(power_curve_data['cluster_power_curves']):
        cluster_id = curve['cluster_id']
        powers = np.array(curve['power_values_w'])
        
        # Get probability distribution for this cluster
        probabilities = probability_matrix[i, :] / 100.0  # Convert from % to fraction
        
        # Ensure arrays have same length
        min_len = min(len(powers), len(probabilities))
        powers = powers[:min_len]
        probabilities = probabilities[:min_len]
        
        # Calculate expected power for this cluster
        expected_power = np.sum(powers * probabilities)
        
        # Calculate AEP contribution from this cluster
        cluster_aep_wh = expected_power * HOURS_PER_YEAR
        cluster_aep.append({
            'cluster_id': cluster_id,
            'expected_power_w': float(expected_power),
            'aep_wh': float(cluster_aep_wh),
            'aep_kwh': float(cluster_aep_wh / 1000),
            'aep_mwh': float(cluster_aep_wh / 1e6),
        })
        
        total_aep_wh += cluster_aep_wh
    
    # Calculate capacity factor
    rated_power = power_curve_data['aggregate_power_curve']['max_power_w']
    capacity_factor = (total_aep_wh / HOURS_PER_YEAR) / rated_power if rated_power > 0 else 0
    
    return {
        'total_aep': {
            'wh': float(total_aep_wh),
            'kwh': float(total_aep_wh / 1000),
            'mwh': float(total_aep_wh / 1e6),
            'gwh': float(total_aep_wh / 1e9),
        },
        'rated_power_kw': float(rated_power / 1000),
        'capacity_factor': float(capacity_factor),
        'cluster_contributions': cluster_aep,
        'calculation_timestamp': datetime.now().isoformat(),
    }


def run_luchsinger_aep():
    """Run AEP calculation using Luchsinger power model."""
    
    # Define paths
    config_dir = PROJECT_ROOT / "config"
    results_path = PROJECT_ROOT / "results"
    wind_resource_path = results_path / "wind_resource.yml"
    power_curves_output = results_path / "luchsinger_power_curves.yml"
    aep_output = results_path / "luchsinger_aep_results.yml"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    print("AWESPA - Luchsinger AEP Calculation")
    print("=" * 50)
    
    # -------------------------------------------------------------------------
    # Step 1: Check wind resource file exists
    # -------------------------------------------------------------------------
    print("\nStep 1: Loading Wind Resource Data")
    print("-" * 40)
    
    if not wind_resource_path.exists():
        print(f"✗ Wind resource file not found: {wind_resource_path}")
        print("  Please run wind clustering first:")
        print("  python scripts/run_wind_clustering.py")
        return False
    
    with open(wind_resource_path, 'r') as f:
        wind_resource_data = yaml.load(f, Loader=yaml.FullLoader)
    
    metadata = wind_resource_data['metadata']
    print(f"✓ Wind resource loaded from {wind_resource_path}")
    print(f"  - Number of clusters: {metadata['n_clusters']}")
    print(f"  - Number of wind speed bins: {metadata['n_wind_speed_bins']}")
    print(f"  - Reference height: {metadata['reference_height_m']} m")
    print(f"  - Location: ({metadata['location']['latitude']}°N, {metadata['location']['longitude']}°E)")
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize and configure Luchsinger model
    # -------------------------------------------------------------------------
    print("\nStep 2: Initializing Luchsinger Power Model")
    print("-" * 40)
    
    try:
        power_model = LuchsingerPowerModel()
        power_model.load_configuration(config_dir)
        print("✓ Luchsinger model initialized successfully")
        print(f"  - Wing area: {power_model.airborne_config['projected_area']} m²")
        print(f"  - Tether length: {power_model.tether_config['length']} m")
        print(f"  - Cut-in speed: {power_model.power_model.cutInWindSpeed:.1f} m/s")
        print(f"  - Cut-out speed: {power_model.power_model.cutOutWindSpeed:.1f} m/s")
    except Exception as e:
        print(f"✗ Error initializing power model: {e}")
        return False
    
    # -------------------------------------------------------------------------
    # Step 3: Compute power curves for all clusters
    # -------------------------------------------------------------------------
    print("\nStep 3: Computing Power Curves")
    print("-" * 40)
    
    try:
        power_model.compute_power_curves(power_curves_output)
        print(f"✓ Power curves computed and saved to {power_curves_output}")
        
        # Load the computed power curves
        with open(power_curves_output, 'r') as f:
            power_curve_data = yaml.safe_load(f)
        
        # Print summary
        aggregate = power_curve_data['aggregate_power_curve']
        print(f"  - Maximum power: {aggregate['max_power_w']/1000:.1f} kW")
        print(f"  - Rated wind speed: {aggregate['rated_wind_speed_m_s']:.1f} m/s")
        
    except Exception as e:
        print(f"✗ Error computing power curves: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate AEP
    # -------------------------------------------------------------------------
    print("\nStep 4: Calculating Annual Energy Production (AEP)")
    print("-" * 40)
    
    try:
        aep_results = calculate_aep_from_power_curves(power_curve_data, wind_resource_data)
        
        print(f"✓ AEP calculation complete")
        print(f"\n  === AEP Results ===")
        print(f"  Total AEP: {aep_results['total_aep']['mwh']:.2f} MWh/year")
        print(f"  Rated Power: {aep_results['rated_power_kw']:.1f} kW")
        print(f"  Capacity Factor: {aep_results['capacity_factor']*100:.1f}%")
        
        # Save AEP results
        with open(aep_output, 'w') as f:
            yaml.dump(aep_results, f, default_flow_style=False)
        print(f"\n✓ AEP results saved to {aep_output}")
        
    except Exception as e:
        print(f"✗ Error calculating AEP: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # -------------------------------------------------------------------------
    # Step 5: Generate plots
    # -------------------------------------------------------------------------
    print("\nStep 5: Generating Plots")
    print("-" * 40)
    
    try:
        generate_plots(power_curve_data, aep_results, results_path)
        print("✓ Plots generated successfully")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("AWESPA Luchsinger AEP Calculation Complete!")
    print("=" * 50)
    print(f"\nOutput files:")
    print(f"  - Power curves: {power_curves_output}")
    print(f"  - AEP results: {aep_output}")
    print(f"  - Plots: {results_path}/luchsinger_*.png")
    
    return True


def generate_plots(power_curve_data: dict, 
                   aep_results: dict, 
                   results_path: Path):
    """Generate visualization plots.
    
    Args:
        power_curve_data (dict): Power curve data.
        aep_results (dict): AEP calculation results.
        results_path (Path): Directory to save plots.
    """
    # Plot 1: Aggregate Power Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Power curve
    ax1 = axes[0]
    aggregate = power_curve_data['aggregate_power_curve']
    wind_speeds = aggregate['wind_speed_m_s']
    powers = np.array(aggregate['power_w']) / 1000  # Convert to kW
    
    ax1.plot(wind_speeds, powers, 'b-', linewidth=2)
    ax1.fill_between(wind_speeds, 0, powers, alpha=0.3)
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Power (kW)', fontsize=12)
    ax1.set_title('Luchsinger Model - Aggregate Power Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 30])
    ax1.set_ylim(bottom=0)
    
    # Add rated point
    max_power = aggregate['max_power_w'] / 1000
    rated_ws = aggregate['rated_wind_speed_m_s']
    ax1.axhline(y=max_power, color='r', linestyle='--', alpha=0.7, 
                label=f'Rated: {max_power:.1f} kW')
    ax1.axvline(x=rated_ws, color='g', linestyle='--', alpha=0.7,
                label=f'Rated WS: {rated_ws:.1f} m/s')
    ax1.legend()
    
    # Plot 2: Cluster AEP contributions
    ax2 = axes[1]
    cluster_ids = [c['cluster_id'] for c in aep_results['cluster_contributions']]
    cluster_aeps = [c['aep_mwh'] for c in aep_results['cluster_contributions']]
    
    bars = ax2.bar(cluster_ids, cluster_aeps, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('AEP (MWh/year)', fontsize=12)
    ax2.set_title('AEP Contribution by Wind Cluster', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, aep in zip(bars, cluster_aeps):
        height = bar.get_height()
        ax2.annotate(f'{aep:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(results_path / "luchsinger_power_and_aep.png", dpi=150, bbox_inches='tight')
    
    # Plot 3: Individual cluster power curves
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(power_curve_data['cluster_power_curves'])))
    
    for curve, color in zip(power_curve_data['cluster_power_curves'], colors):
        cluster_id = curve['cluster_id']
        frequency = curve['frequency']
        powers = np.array(curve['power_values_w']) / 1000
        
        ax3.plot(wind_speeds, powers, 
                color=color, linewidth=1.5, alpha=0.8,
                label=f'Cluster {cluster_id} (f={frequency:.1%})')
    
    ax3.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax3.set_ylabel('Power (kW)', fontsize=12)
    ax3.set_title('Luchsinger Model - Power Curves by Cluster', fontsize=14)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 30])
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(results_path / "luchsinger_cluster_power_curves.png", dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    """Main entry point."""
    success = run_luchsinger_aep()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
