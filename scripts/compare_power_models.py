#!/usr/bin/env python3
"""
Power Model Demonstration Script

This script demonstrates how to use the Luchsinger power estimation model.
The model uses awesIO format configuration files.

Usage:
    python scripts/compare_power_models.py

Author: AWESPA Development Team
Date: January 2026
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.power.luchsinger_power import LuchsingerPowerModel


def compare_power_models():
    """Demonstrate the Luchsinger power model.
    
    This script demonstrates the interface and generates a power curve
    using the Luchsinger model with awesIO format configuration files.
    """
    # Define paths
    config_dir = PROJECT_ROOT / "config" / "meridional_case_1"
    results_path = PROJECT_ROOT / "results"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    print("AWESPA Power Model Demonstration")
    print("=" * 50)
    
    # -------------------------------------------------------------------------
    # Initialize and use Luchsinger Model
    # -------------------------------------------------------------------------
    print("\n1. Luchsinger Power Model")
    print("-" * 30)
    
    try:
        luchsinger_model = LuchsingerPowerModel()
        
        # Load configuration using the new awesIO format
        system_path = config_dir / "soft_kite_pumping_ground_gen_system.yml"
        simulation_settings_path = config_dir / "Lucsinger_simulation_settings_config.yml"
        
        luchsinger_model.load_configuration(
            system_path=system_path,
            simulation_settings_path=simulation_settings_path
        )
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
        import traceback
        traceback.print_exc()
        luchsinger_available = False
        luchsinger_powers = None
    
    # -------------------------------------------------------------------------
    # Plot power curve
    # -------------------------------------------------------------------------
    if luchsinger_available:
        print("\n2. Generating Power Curve Plot")
        print("-" * 30)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Luchsinger model
        ax.plot(wind_speeds, luchsinger_powers / 1000, 
                'b-', linewidth=2, label='Luchsinger Model')
        
        ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
        ax.set_ylabel('Power (kW)', fontsize=12)
        ax.set_title('AWESPA Power Model - Luchsinger Model', fontsize=14)
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
    
    if luchsinger_available:
        print("\nLuchsinger Model Key Parameters:")
        print(f"  - Wing area: {luchsinger_model.power_model.wingArea} m²")
        print(f"  - Tether length: {luchsinger_model.power_model.tetherMaxLength} m")


def main():
    """Main entry point."""
    compare_power_models()


if __name__ == "__main__":
    main()
