"""Luchsinger power estimation wrapper for the vendored LuchsingerModel repository.

This wrapper adapts the vendored Luchsinger power model to the AWESPA modular
architecture. The underlying model accepts awesIO format configuration files directly.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base import PowerEstimationModel

# Add vendor path to import the Luchsinger model code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "LuchsingerPowerModel"
sys.path.insert(0, str(VENDOR_PATH))

# Import vendor LuchsingerPowerModel
try:
    from src.power_luchsinger.power_model import PowerModel # type: ignore
    from src.power_luchsinger.plotting import plot_comprehensive_analysis, extract_model_params # type: ignore
except ImportError as e:
    print(f"Import error for LuchsingerPowerModel: {e}")
    PowerModel = None


class LuchsingerPowerModel(PowerEstimationModel):
    """Wrapper for the Luchsinger pumping kite power model.
    
    This wrapper adapts the vendored Luchsinger power model to the AWESPA
    modular architecture. The model accepts awesIO format configuration files.
    """
    
    def __init__(self):
        """Initialize the Luchsinger power estimation model."""
        self.power_model: Optional[PowerModel] = None # type: ignore
        
    def load_configuration(
        self, 
        system_path: Path,
        simulation_settings_path: Path,
        operational_constraints_path: Path = None
    ) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            system_path: Path to the combined system configuration YAML file
                (awesIO format containing wing, tether, ground_station components).
            simulation_settings_path: Path to simulation settings YAML file
                containing operational and atmosphere parameters.
            operational_constraints_path: Not used by Luchsinger model. Defaults to None.
        """
        if PowerModel is None:
            raise ImportError("Luchsinger PowerModel could not be imported")
        
        self.power_model = PowerModel.from_yaml(
            yamlPath=system_path,
            simulationSettingsPath=simulation_settings_path,
            validate=False
        )
    
    def compute_power_curves(
        self,
        output_path: Path = None,
        plot: bool = False
    ) -> None:
        """Compute power curves and optionally export/plot.
        
        Args:
            output_path: Path where power curve YAML will be written. If None,
                no export is performed.
            plot: Whether to generate and display plots.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Compute power curves with 500 points
        num_points = 500
        data = self.power_model.generate_power_curve(numPoints=num_points)
        
        # Export to YAML if output path provided
        if output_path is not None:
            self.power_model.export_power_curves_awesio(
                data=data,
                output_path=output_path,
                validate=False
            )
            print(f"Power curve exported to: {output_path}")
        
        # Generate plots if requested
        if plot:
            params = extract_model_params(self.power_model)
            plot_comprehensive_analysis(
                data=data,
                model_params=params,
                save_path=None,
                show=True
            )
    
    def calculate_power_at_wind_speed(
        self,
        wind_speed: float,
        output_path: Path = None,
        plot: bool = False
    ) -> float:
        """Calculate power output at a single wind speed.
        
        Args:
            wind_speed: Wind speed in m/s.
            output_path: Path where results will be written. If None,
                no export is performed.
            plot: Whether to generate visualization for this wind speed.
            
        Returns:
            Power output in W.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Calculate power at single wind speed
        result = self.power_model.calculate_power(windSpeed=wind_speed)
        power = result['cyclePower']
        
        # Print the result
        print(f"\nPower at {wind_speed:.1f} m/s: {power:.2f} W ({power/1000:.2f} kW)")
        
        # Export if requested (not yet implemented)
        if output_path is not None:
            print("Note: Functionality to export results from single wind speed does not exist yet.")
        
        # Plot if requested (not yet implemented in vendored model)
        if plot:
            print("Note: Plotting for single wind speed not yet implemented in Luchsinger model.")
        
        return power

