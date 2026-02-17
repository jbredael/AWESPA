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
        self.powerModel: Optional[PowerModel] = None # type: ignore
        
    def load_configuration(
        self, 
        systemPath: Path,
        simulationSettingsPath: Path,
        operationalConstraintsPath: Path = None
    ) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            systemPath (Path): Path to the combined system configuration YAML file
                (awesIO format containing wing, tether, ground_station components).
            simulationSettingsPath (Path): Path to simulation settings YAML file
                containing operational and atmosphere parameters.
            operationalConstraintsPath (Path): Not used by Luchsinger model. Defaults to None.
        """
        if PowerModel is None:
            raise ImportError("Luchsinger PowerModel could not be imported")
        
        self.powerModel = PowerModel.from_yaml(
            yamlPath=systemPath,
            simulationSettingsPath=simulationSettingsPath,
            validate=False
        )
    
    def compute_power_curves(
        self,
        outputPath: Path = None,
        plot: bool = False
    ) -> None:
        """Compute power curves and optionally export/plot.
        
        Args:
            outputPath (Path): Path where power curve YAML will be written. If None,
                no export is performed. Defaults to None.
            plot (bool): Whether to generate and display plots. Defaults to False.
        """
        if self.powerModel is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Compute power curves with 500 points
        numPoints = 500
        data = self.powerModel.generate_power_curve(numPoints=numPoints)
        
        # Export to YAML if output path provided
        if outputPath is not None:
            self.powerModel.export_power_curves_awesio(
                data=data,
                output_path=outputPath,
                validate=False
            )
            print(f"Power curve exported to: {outputPath}")
        
        # Generate plots if requested
        if plot:
            params = extract_model_params(self.powerModel)
            plot_comprehensive_analysis(
                data=data,
                model_params=params,
                save_path=None,
                show=True
            )
    
    def calculate_power_at_wind_speed(
        self,
        windSpeed: float,
        outputPath: Path = None,
        plot: bool = False
    ) -> float:
        """Calculate power output at a single wind speed.
        
        Args:
            windSpeed (float): Wind speed in m/s.
            outputPath (Path): Path where results will be written. If None,
                no export is performed. Defaults to None.
            plot (bool): Whether to generate visualization for this wind speed. Defaults to False.
            
        Returns:
            float: Power output in W.
        """
        if self.powerModel is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Calculate power at single wind speed
        result = self.powerModel.calculate_power(windSpeed=windSpeed)
        power = result['cyclePower']
        
        # Print the result
        print(f"\nPower at {windSpeed:.1f} m/s: {power:.2f} W ({power/1000:.2f} kW)")
        
        # Export if requested (not yet implemented)
        if outputPath is not None:
            print("Note: Functionality to export results from single wind speed does not exist yet.")
        
        # Plot if requested (not yet implemented in vendored model)
        if plot:
            print("Note: Plotting for single wind speed not yet implemented in Luchsinger model.")
        
        return power

