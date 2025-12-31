#!/usr/bin/env python3
"""
Power Model Comparison Script

This script demonstrates how to use both power estimation models:
- LuchsingerPowerModel: Simplified analytical model based on Luchsinger (2013)
- AWEPowerEstimationModel: Detailed QSM-based model (NOTE: currently not functional)

Usage:
    python scripts/compare_power_models.py

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

from awespa.power.luchsinger_power import LuchsingerPowerModel
from awespa.power.awe_power import AWEPowerEstimationModel


def compare_power_models():
    """Compare power output from different models.
    
    Note: AWEPowerEstimationModel is currently not functional. This script
    demonstrates the interface and how both models would be used when
    AWEPowerEstimationModel is operational.
    """
    # Define paths
    config_dir = PROJECT_ROOT / "config"
    results_path = PROJECT_ROOT / "results"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    print("AWESPA Power Model Comparison")
    print("=" * 50)
    
    # -------------------------------------------------------------------------
    # Initialize and use Luchsinger Model
    # -------------------------------------------------------------------------
    print("\n1. Luchsinger Power Model")
    print("-" * 30)
    
    try:
        luchsinger_model = LuchsingerPowerModel()
        luchsinger_model.load_configuration(config_dir)
        print("✓ Luchsinger model loaded successfully")
        
        # Generate power curve
        wind_speeds = np.linspace(4.0, 25.0, 50)
        luchsinger_powers = []
        
        for ws in wind_speeds:
            power = luchsinger_model.get_power_at_wind_speed(ws)
            luchsinger_powers.append(power)
        
        luchsinger_powers = np.array(luchsinger_powers)
        
        # Print some statistics
        max_power_idx = np.argmax(luchsinger_powers)
        print(f"  - Cut-in wind speed: {luchsinger_model.power_model.cutInWindSpeed:.1f} m/s")
        print(f"  - Cut-out wind speed: {luchsinger_model.power_model.cutOutWindSpeed:.1f} m/s")
        print(f"  - Maximum power: {luchsinger_powers[max_power_idx]/1000:.1f} kW")
        print(f"  - Wind speed at max power: {wind_speeds[max_power_idx]:.1f} m/s")
        
        luchsinger_available = True
        
    except Exception as e:
        print(f"✗ Error loading Luchsinger model: {e}")
        luchsinger_available = False
        luchsinger_powers = None
    
    # -------------------------------------------------------------------------
    # Initialize AWE Production Estimation Model (currently not functional)
    # -------------------------------------------------------------------------
    print("\n2. AWE Production Estimation Model")
    print("-" * 30)
    print("  NOTE: This model is currently not functional.")
    print("  Skipping AWE Production Estimation Model...")
    
    # The following code demonstrates how the model would be used:
    # try:
    #     awe_model = AWEPowerEstimationModel()
    #     awe_model.load_configuration(config_dir)
    #     print("✓ AWE Production Estimation model loaded successfully")
    #     
    #     # Compute power curves
    #     output_file = results_path / "awe_power_curves.yml"
    #     awe_model.compute_power_curves(output_file)
    #     
    #     awe_available = True
    # except Exception as e:
    #     print(f"✗ Error loading AWE model: {e}")
    #     awe_available = False
    
    awe_available = False
    
    # -------------------------------------------------------------------------
    # Plot comparison (only Luchsinger for now)
    # -------------------------------------------------------------------------
    if luchsinger_available:
        print("\n3. Generating Power Curve Plot")
        print("-" * 30)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Luchsinger model
        ax.plot(wind_speeds, luchsinger_powers / 1000, 
                'b-', linewidth=2, label='Luchsinger Model')
        
        # If AWE model were available, add it to the plot
        # if awe_available:
        #     ax.plot(wind_speeds, awe_powers / 1000, 
        #             'r--', linewidth=2, label='AWE Production Model')
        
        ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
        ax.set_ylabel('Power (kW)', fontsize=12)
        ax.set_title('AWESPA Power Model Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 30])
        ax.set_ylim(bottom=0)
        
        # Save plot
        plot_path = results_path / "power_model_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_path}")
        
        plt.show()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Luchsinger Model: {'Available' if luchsinger_available else 'Not Available'}")
    print(f"AWE Production Model: {'Available' if awe_available else 'Not Functional'}")
    
    if luchsinger_available:
        print("\nLuchsinger Model Key Parameters:")
        print(f"  - Wing area: {luchsinger_model.airborne_config['projected_area']} m²")
        print(f"  - Tether length: {luchsinger_model.tether_config['length']} m")


def main():
    """Main entry point."""
    compare_power_models()


if __name__ == "__main__":
    main()
