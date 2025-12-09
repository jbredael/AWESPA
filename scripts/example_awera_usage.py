#!/usr/bin/env python3
"""
Example script demonstrating the AWERA toolchain usage.

This script shows how to:
1. Set up the modular AWERA components
2. Configure each component using YAML files
3. Execute the complete AEP calculation pipeline
4. Handle the YAML-based data exchange between components

Author: AWERA Development Team
Date: December 2025
"""

from pathlib import Path
import sys

# Add the src directory to the path to import awera
sys.path.insert(0, str(Path(__file__).parent / "src"))

from awera import (
    AEPCalculator,
    WindProfileClusteringModel
)


def main():
    """Main function demonstrating AWERA toolchain usage."""
    
    # Define paths
    project_root = Path(__file__).parent
    config_path = project_root / "config" / "main_config.yml"
    data_path = project_root / "data"
    results_path = project_root / "results"
    
    print("AWERA Toolchain Demonstration")
    print("=" * 50)
    
    # Step 1: Initialize the AEP calculator with main configuration
    print("Step 1: Initializing AEP Calculator...")
    try:
        aep_calculator = AEPCalculator(config_path)
        print(f"✓ Configuration loaded from {config_path}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return
    
    # Step 2: Set up the modular components
    print("\nStep 2: Setting up modular components...")
    try:
        # For this example, we'll set up the wind clustering model
        # In a complete implementation, you would also set up:
        # - OptimiserImplementation (inheriting from Optimiser)
        # - PowerEstimationImplementation (inheriting from PowerEstimationModel)
        # - KiteModelImplementation (inheriting from KiteModel)
        # - ControlModelImplementation (inheriting from ControlModel)
        # - GroundStationImplementation (inheriting from GroundStationModel)
        
        aep_calculator.setup_models(
            wind_model_class=WindProfileClusteringModel,
            optimiser_class=None,  # Would be your OptimiserImplementation
            power_model_class=None,  # Would be your PowerEstimationImplementation
            kite_model_class=None,  # Optional
            control_model_class=None,  # Optional
            groundstation_model_class=None  # Optional
        )
        print("✓ Component classes configured")
        
    except Exception as e:
        print(f"✗ Error setting up components: {e}")
        return
    
    # Step 3: Execute the AEP calculation pipeline
    print("\nStep 3: Executing AEP calculation pipeline...")
    print("Note: This demonstration shows the structure.")
    print("For a complete run, implement all component classes.")
    
    # Demonstrate individual wind clustering (if you want to test just that part)
    print("\nDemo: Wind Profile Clustering (standalone)")
    try:
        wind_model = WindProfileClusteringModel()
        
        # Load configuration
        wind_config_path = project_root / "config" / "wind_clustering_config.yml"
        if wind_config_path.exists():
            wind_model.load_from_yaml(wind_config_path)
            print(f"✓ Wind clustering configuration loaded")
            
            # Note: This would fail without proper ERA5 data and vendor dependencies
            # wind_model.cluster(data_path, results_path / "wind_profiles.yml")
            print("  (Clustering execution skipped - requires ERA5 data and dependencies)")
        else:
            print(f"✗ Wind configuration not found at {wind_config_path}")
            
    except Exception as e:
        print(f"✗ Error in wind clustering demo: {e}")
    
    # Step 4: Show the expected YAML-based data flow
    print("\nStep 4: YAML-based Data Exchange Flow")
    print("-" * 40)
    print("1. Wind Model → wind_profiles.yml")
    print("   Contains: cluster parameters, frequencies, representative profiles")
    print("2. Optimiser → optimal_controls.yml")
    print("   Contains: optimal control parameters for each wind cluster")
    print("3. Power Model → power_curve.yml")
    print("   Contains: power curve data based on wind profiles and controls")
    print("4. AEP Calculator → aep_results.yml")
    print("   Contains: final AEP calculation results")
    
    print("\n" + "=" * 50)
    print("AWERA Toolchain Demonstration Complete")
    print("\nNext steps:")
    print("1. Implement concrete classes for each Abstract Base Class")
    print("2. Ensure vendor dependencies are properly installed")
    print("3. Prepare ERA5 or other wind data")
    print("4. Run the complete pipeline with: aep_calculator.calculate_aep()")


def demonstrate_modular_architecture():
    """Demonstrate the modular plug-and-play architecture."""
    
    print("\n" + "=" * 50)
    print("MODULAR ARCHITECTURE DEMONSTRATION")
    print("=" * 50)
    
    print("\nAbstract Base Classes (interfaces):")
    print("├── WindProfileModel")
    print("├── PowerEstimationModel") 
    print("├── Optimiser")
    print("├── KiteModel")
    print("├── ControlModel")
    print("└── GroundStationModel")
    
    print("\nImplementation Classes (concrete):")
    print("├── WindProfileClusteringModel (wraps vendor wind-profile-clustering)")
    print("├── QSMPowerModel (would wrap vendor QSM)")
    print("├── GeneticAlgorithmOptimiser (would implement optimization)")
    print("├── SimpleKiteModel (would implement kite dynamics)")
    print("├── OptimalControlModel (would implement control strategies)")
    print("└── ElectricWinchGroundStation (would implement ground operations)")
    
    print("\nYAML Configuration Files:")
    print("├── main_config.yml (orchestrates entire pipeline)")
    print("├── wind_clustering_config.yml")
    print("├── optimisation_config.yml")
    print("├── power_estimation_config.yml")
    print("├── kite_config.yml")
    print("├── control_config.yml")
    print("└── groundstation_config.yml")
    
    print("\nData Exchange (YAML files):")
    print("├── wind_profiles.yml (wind → optimiser)")
    print("├── optimal_controls.yml (optimiser → power)")
    print("├── power_curve.yml (power → aep)")
    print("└── aep_results.yml (final output)")


if __name__ == "__main__":
    main()
    demonstrate_modular_architecture()