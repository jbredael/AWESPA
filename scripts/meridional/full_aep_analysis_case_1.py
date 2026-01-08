#!/usr/bin/env python3
"""
Full AEP Analysis for Meridional Case 1

This script performs a comprehensive analysis of Annual Energy Production
for an AWE system using the AWESPA modular architecture.

The analysis includes three main sections:
1. Wind clustering - Process ERA5 data and identify representative wind profiles
2. Power curve generation - Compute power curves using the Luchsinger model
3. AEP calculation and visualization - Calculate AEP, capacity factor, and generate plots

Usage:
    python scripts/meridional/full_aep_analysis_case_1.py

Author: Joren Bredael
Date: January 2026
"""

import sys
import yaml
from pathlib import Path

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.wind.clustering import WindProfileClusteringModel
from awespa.power.luchsinger_power import LuchsingerPowerModel
from awespa.pipeline.aep import calculate_aep


def main():
    """Execute full AEP analysis pipeline."""
    
    print("=" * 80)
    print("AWESPA FULL AEP ANALYSIS - MERIDIONAL CASE 1")
    print("=" * 80)
    
    # Define paths
    config_dir = PROJECT_ROOT / "config" / "meridional_case_1"
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results" / "meridional_case_1"
    
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    wind_resource_path = results_dir / "wind_resource.yml"
    power_curves_path = results_dir / "luchsinger_power_curves.yml"
    aep_results_path = results_dir / "luchsinger_aep_results.yml"
    
    # =========================================================================
    # SECTION 1: WIND CLUSTERING
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: WIND CLUSTERING")
    print("=" * 80)
    
    run_wind_clustering = False  # Set to False to skip if already run
    
    if run_wind_clustering:
        print("\n[1/3] Initializing wind clustering model...")
        wind_model = WindProfileClusteringModel()
        
        # Load wind clustering configuration
        wind_config_path = config_dir / "wind_clustering_config.yml"
        
        if not wind_config_path.exists():
            print(f"ERROR: Wind clustering config not found: {wind_config_path}")
            print("Looking for alternative configuration...")
            # Try the main config directory
            alt_config_path = PROJECT_ROOT / "config" / "wind_clustering_config.yml"
            if alt_config_path.exists():
                wind_config_path = alt_config_path
                print(f"Using alternative config: {wind_config_path}")
            else:
                print("ERROR: No wind clustering configuration found!")
                return False
        
        print(f"Loading configuration from: {wind_config_path}")
        try:
            wind_model.load_from_yaml(wind_config_path)
            print("✓ Wind clustering configuration loaded successfully")
        except Exception as e:
            print(f"✗ Error loading wind configuration: {e}")
            return False
        
        # Perform clustering
        print("\n[2/3] Performing wind profile clustering...")
        print("This may take several minutes depending on data size...")
        try:
            wind_model.cluster(data_dir, wind_resource_path)
            print(f"✓ Wind clustering complete")
            print(f"✓ Results saved to: {wind_resource_path}")
        except Exception as e:
            print(f"✗ Error during wind clustering: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Display clustering summary
        print("\n[3/3] Wind clustering summary:")
        try:
            with open(wind_resource_path, 'r') as f:
                wind_data = yaml.safe_load(f)
            print(f"  - Number of clusters: {wind_data['metadata']['n_clusters']}")
            print(f"  - Total samples: {wind_data['metadata']['total_samples']}")
            print(f"  - Time range: {wind_data['metadata']['time_range']['start_year']}-{wind_data['metadata']['time_range']['end_year']}")
            print(f"  - Reference height: {wind_data['metadata']['reference_height_m']} m")
        except Exception as e:
            print(f"Warning: Could not display summary: {e}")
    else:
        print("\nSkipping wind clustering (using existing results)")
        if not wind_resource_path.exists():
            print(f"ERROR: Wind resource file not found: {wind_resource_path}")
            print("Please run wind clustering first or set run_wind_clustering=True")
            return False
    
    # =========================================================================
    # SECTION 2: POWER CURVE GENERATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: POWER CURVE GENERATION (LUCHSINGER MODEL)")
    print("=" * 80)
    
    run_power_curves = True  # Set to False to skip if already run
    
    if run_power_curves:
        print("\n[1/3] Initializing Luchsinger power model...")
        power_model = LuchsingerPowerModel()
        
        # Specify configuration file paths
        # These can be customized to use different versions or variants
        # e.g., airborne_path = config_dir / "airborne_v2.yml"
        airborne_path = config_dir / "airborne.yml"
        tether_path = config_dir / "tether.yml"
        operational_constraints_path = config_dir / "operational_constraints.yml"
        ground_station_path = config_dir / "ground_station.yml"
        # Wind resource is in the results directory for this case
        wind_resource_config_path = results_dir / "wind_resource.yml"
        
        print(f"Loading configuration files:")
        print(f"  - Airborne: {airborne_path.name}")
        print(f"  - Tether: {tether_path.name}")
        print(f"  - Operational constraints: {operational_constraints_path.name}")
        print(f"  - Wind resource: {wind_resource_config_path}")
        
        try:
            power_model.load_configuration(
                airborne_path=airborne_path,
                tether_path=tether_path,
                operational_constraints_path=operational_constraints_path,
                ground_station_path=ground_station_path,
                wind_resource_path=wind_resource_config_path,
            )
            print("✓ Power model configuration loaded successfully")
            print(f"  - Kite area: {power_model.airborne_config['kite']['projected_area_m2']} m²")
            print(f"  - Kite mass: {power_model.airborne_config['kite']['mass_kg']} kg")
            print(f"  - Tether length: {power_model.tether_config['tether']['length_m']} m")
        except Exception as e:
            print(f"✗ Error loading power model configuration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Compute power curves
        print("\n[2/3] Computing power curves for all wind clusters...")
        try:
            power_model.compute_power_curves(power_curves_path)
            print(f"✓ Power curves computed successfully")
            print(f"✓ Results saved to: {power_curves_path}")
        except Exception as e:
            print(f"✗ Error computing power curves: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Display power curve summary
        print("\n[3/3] Power curve summary:")
        try:
            with open(power_curves_path, 'r') as f:
                power_data = yaml.safe_load(f)
            agg_curve = power_data['aggregate_power_curve']
            print(f"  - Number of cluster curves: {power_data['metadata']['n_clusters']}")
            print(f"  - Rated power: {agg_curve['max_power_w']/1000:.2f} kW")
            print(f"  - Mean power: {agg_curve['mean_power_w']/1000:.2f} kW")
            print(f"  - Cut-in wind speed: {agg_curve['cut_in_wind_speed_m_s']:.2f} m/s")
            print(f"  - Cut-out wind speed: {agg_curve['cut_out_wind_speed_m_s']:.2f} m/s")
        except Exception as e:
            print(f"Warning: Could not display summary: {e}")
    else:
        print("\nSkipping power curve generation (using existing results)")
        if not power_curves_path.exists():
            print(f"ERROR: Power curves file not found: {power_curves_path}")
            print("Please run power curve generation first or set run_power_curves=True")
            return False
    
    # =========================================================================
    # SECTION 3: AEP CALCULATION AND VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: AEP CALCULATION AND VISUALIZATION")
    print("=" * 80)
    
    print("\n[1/2] Calculating Annual Energy Production...")
    try:
        aep_results = calculate_aep(
            power_curve_path=power_curves_path,
            wind_resource_path=wind_resource_path,
            output_path=aep_results_path,
            plot=True,  # Generate plots
            plot_output_dir=results_dir
        )
        print(f"✓ AEP calculation complete")
        print(f"✓ Results saved to: {aep_results_path}")
    except Exception as e:
        print(f"✗ Error calculating AEP: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display AEP summary
    print("\n[2/2] AEP Analysis Summary:")
    print("-" * 80)
    print(f"Annual Energy Production:")
    print(f"  - Total AEP: {aep_results['total_aep']['mwh']:.2f} MWh/year")
    print(f"            = {aep_results['total_aep']['gwh']:.4f} GWh/year")
    print(f"  - Rated power: {aep_results['rated_power_kw']:.2f} kW")
    print(f"  - Mean power: {aep_results['mean_power_kw']:.2f} kW")
    print(f"  - Capacity factor: {aep_results['capacity_factor']*100:.2f}%")
    
    print(f"\nCluster Contributions:")
    for contrib in aep_results['cluster_contributions']:
        cluster_id = contrib['cluster_id']
        aep_mwh = contrib['aep_mwh']
        frequency = contrib['frequency'] * 100
        print(f"  - Cluster {cluster_id}: {aep_mwh:.2f} MWh/year ({frequency:.1f}% occurrence)")
    
    print("\n" + "=" * 80)
    print("FULL AEP ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {results_dir}")
    print(f"  - Wind resource: {wind_resource_path.name}")
    print(f"  - Power curves: {power_curves_path.name}")
    print(f"  - AEP results: {aep_results_path.name}")
    print(f"  - Plots: aep_analysis_complete.png")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
