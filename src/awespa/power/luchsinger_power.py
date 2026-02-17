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
    from src.power_luchsinger.power_model import PowerModel, load_wind_shear_profiles # type: ignore
except ImportError as e:
    print(f"Import error for LuchsingerPowerModel: {e}")
    PowerModel = None
    load_wind_shear_profiles = None


class LuchsingerPowerModel(PowerEstimationModel):
    """Wrapper for the Luchsinger pumping kite power model.
    
    This wrapper adapts the vendored Luchsinger power model to the AWESPA
    modular architecture. The model accepts awesIO format configuration files.
    """
    
    def __init__(self):
        """Initialize the Luchsinger power estimation model."""
        self.powerModel: Optional[PowerModel] = None # type: ignore
        self.windShearData: Optional[Dict[str, Any]] = None
        
    def load_configuration(
        self, 
        systemPath: Path,
        simulationSettingsPath: Path,
        operationalConstraintsPath: Path = None,
        windResourcePath: Path = None
    ) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            systemPath (Path): Path to the combined system configuration YAML file
                (awesIO format containing wing, tether, ground_station components).
            simulationSettingsPath (Path): Path to simulation settings YAML file
                containing operational and atmosphere parameters.
            operationalConstraintsPath (Path): Not used by Luchsinger model. Defaults to None.
            windResourcePath (Path): Path to wind resource YAML file containing
                wind shear profiles and clusters. If None, uniform wind profile
                is used. Defaults to None.
        """
        if PowerModel is None:
            raise ImportError("Luchsinger PowerModel could not be imported")
        
        self.powerModel = PowerModel.from_yaml(
            yamlPath=systemPath,
            simulationSettingsPath=simulationSettingsPath,
            validate=False
        )
        
        # Load wind resource if provided
        if windResourcePath is not None:
            if load_wind_shear_profiles is None:
                raise ImportError("load_wind_shear_profiles could not be imported")
            
            windResourcePath = Path(windResourcePath)
            if not windResourcePath.exists():
                raise FileNotFoundError(f"Wind resource file not found: {windResourcePath}")
            
            self.windShearData = load_wind_shear_profiles(windResourcePath)
            print(f"Loaded wind resource with {self.windShearData['n_clusters']} clusters from: {windResourcePath.name}")
        else:
            self.windShearData = None
            print("No wind resource provided, using uniform wind profile")
    
    def compute_power_curves(
        self,
        outputPath: Path = None
    ) -> Dict[str, Any]:
        """Compute power curves and optionally export.
        
        If a wind resource was loaded, this computes power curves for each
        wind shear profile/cluster. Otherwise, computes a single power curve
        with uniform wind profile.
        
        Args:
            outputPath (Path): Path where power curve YAML will be written. If None,
                no export is performed. Defaults to None.
        
        Returns:
            dict: Power curve data (structure depends on whether wind shear is used).
        
        Note:
            For plotting, use the vendor's plotting.py script directly on the exported data.
        """
        if self.powerModel is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Compute power curves with 500 points
        numPoints = 500
        
        if self.windShearData is not None:
            # Compute power curves with wind shear profiles
            print(f"Computing power curves with wind shear ({self.windShearData['n_clusters']} profiles)...")
            data = self.powerModel.generate_power_curves_with_shear(
                wind_shear_data=self.windShearData,
                numPoints=numPoints
            )
        else:
            # Compute single power curve with uniform wind profile
            print("Computing power curve with uniform wind profile...")
            data = self.powerModel.generate_power_curve(numPoints=numPoints)
        
        # Export to YAML if output path provided
        if outputPath is not None:
            self.powerModel.export_power_curves_awesio(
                data=data,
                output_path=outputPath,
                validate=False
            )
            print(f"Power curve exported to: {outputPath}")
        
        return data
    
    def calculate_power_at_wind_speed(
        self,
        windSpeed: float,
        outputPath: Path = None
    ) -> float:
        """Calculate power output at a single wind speed.
        
        Args:
            windSpeed (float): Wind speed in m/s.
            outputPath (Path): Path where results will be written. If None,
                no export is performed. Defaults to None.
            
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
        
        return power
    
    def plot_power_curves(
        self,
        powerCurvePath: Path,
        outputDir: Path = None
    ) -> None:
        """Plot power curves using vendor plotting functions.
        
        This method uses the vendor's plotting scripts to visualize power curve
        data that has been exported to a YAML file.
        
        Args:
            powerCurvePath (Path): Path to the power curves YAML file to plot.
            outputDir (Path): Directory where plot images will be saved. If None,
                plots are saved in the same directory as powerCurvePath. Defaults to None.
        
        Raises:
            FileNotFoundError: If power curves file doesn't exist.
            ImportError: If vendor plotting functions cannot be imported.
        """
        # Import vendor plotting functions
        try:
            from scripts.plot_power_curves import plot_all_power_curves, plot_wind_shear_profiles, load_power_curves # type: ignore
        except ImportError as e:
            raise ImportError(f"Could not import vendor plotting functions: {e}")
        
        # Check if power curves file exists
        powerCurvePath = Path(powerCurvePath)
        if not powerCurvePath.exists():
            raise FileNotFoundError(f"Power curves file not found: {powerCurvePath}")
        
        # Set output directory
        if outputDir is None:
            outputDir = powerCurvePath.parent
        else:
            outputDir = Path(outputDir)
            outputDir.mkdir(parents=True, exist_ok=True)
        
        # Load power curve data
        print(f"Loading power curves from: {powerCurvePath.name}")
        data = load_power_curves(powerCurvePath)
        
        # Create power curves comparison plot
        print("Creating power curves comparison plot...")
        output_path_curves = outputDir / "all_power_curves.png"
        plot_all_power_curves(data, output_path_curves)
        
        # Create wind shear profiles plot (optional, may fail if metadata missing)
        try:
            print("Creating wind shear profiles plot...")
            output_path_shear = outputDir / "wind_shear_profiles.png"
            plot_wind_shear_profiles(data, output_path_shear)
        except KeyError:
            print("  Note: Wind shear profile plot requires additional metadata (skipped)")
        
        print(f"\nPlots generated and saved to: {outputDir}")

