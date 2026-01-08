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

# Add awesIO vendor path for validation
AWESIO_PATH = PROJECT_ROOT / "src" / "awespa" / "vendor" / "awesIO" / "src"
sys.path.insert(0, str(AWESIO_PATH))

from awespa.wind.clustering import WindProfileClusteringModel
from awespa.power.luchsinger_power import LuchsingerPowerModel
from awespa.pipeline.aep import calculate_aep
from awespa.vendor.awesIO.src.awesio.validator import validate as awesio_validate



def validate_config_file(config_path: Path, schema_type: str) -> bool:
    """Validate a configuration file using AWESIO validator.
    
    Args:
        config_path: Path to the configuration YAML file.
        schema_type: Schema type name (without _schema suffix).
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    try:
        awesio_validate(
            input=config_path,
            schema_type=f"{schema_type}_schema",
            restrictive=True,  # No extra parameters allowed
            defaults=False,    # No default values added
        )
        print(f"  ✓ {config_path.name} validated successfully")
        return True
    except Exception as e:
        print(f"  ✗ {config_path.name} validation failed: {e}")
        return False


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
    
    run_wind_clustering = True  # Set to False to skip if already run
    
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
        print("\n[1/4] Initializing Luchsinger power model...")
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
        
        # Validate configuration files using AWESIO
        print("\n[2/4] Validating configuration files with AWESIO...")
        validation_passed = True
        validation_passed &= validate_config_file(airborne_path, "airborne")
        validation_passed &= validate_config_file(tether_path, "tether")
        validation_passed &= validate_config_file(operational_constraints_path, "operational_constraints")
        validation_passed &= validate_config_file(ground_station_path, "ground_station")
        
        if not validation_passed:
            print("\n✗ Configuration validation failed. Please fix the errors above.")
            return False
        
        print(f"\nLoading configuration files:")
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
        print("\n[3/4] Computing power curves...")
        try:
            power_curve_data = power_model.compute_power_curves()
            print(f"✓ Power curves computed successfully")
        except Exception as e:
            print(f"✗ Error computing power curves: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Export power curve to YAML
        print("\n[4/4] Exporting power curve to YAML...")
        try:
            power_model.export_to_yaml(power_curves_path)
            print(f"✓ Results saved to: {power_curves_path}")
        except Exception as e:
            print(f"✗ Error exporting power curve: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Display power curve summary
        print("\nPower curve summary:")
        try:
            model_cfg = power_curve_data['metadata']['model_config']
            pc = power_curve_data['power_curves'][0]
            max_power = max(pc['cycle_power_w'])
            mean_power = sum(p for p in pc['cycle_power_w'] if p > 0) / max(1, sum(1 for p in pc['cycle_power_w'] if p > 0))
            rated_idx = pc['cycle_power_w'].index(max_power)
            rated_wind_speed = power_curve_data['reference_wind_speeds_m_s'][rated_idx]
            print(f"  - Rated power: {max_power/1000:.2f} kW")
            print(f"  - Mean power: {mean_power/1000:.2f} kW")
            print(f"  - Rated wind speed: {rated_wind_speed:.2f} m/s")
            print(f"  - Cut-in wind speed: {model_cfg['cut_in_wind_speed_m_s']:.2f} m/s")
            print(f"  - Cut-out wind speed: {model_cfg['cut_out_wind_speed_m_s']:.2f} m/s")
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
    
    run_aep_analysis = True  # Set to False to skip AEP analysis
    
    if run_aep_analysis:
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
    else:
        print("\nSkipping AEP analysis (set run_aep_analysis=True to enable)")
    
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
