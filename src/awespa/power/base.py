"""Abstract Base Class for power estimation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class PowerEstimationModel(ABC):
    """Abstract base class for power estimation models.
    
    All power estimation models must inherit from this class and implement
    the required methods for loading configuration, computing power curves,
    and exporting power data.
    """
    
    @abstractmethod
    def load_configuration(
        self, 
        system_path: Path,
        simulation_settings_path: Path = None,
        operational_constraints_path: Path = None
    ) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            system_path: Path to system configuration YAML file.
            simulation_settings_path: Path to simulation settings YAML file.
                May be None if model does not require simulation settings.
            operational_constraints_path: Path to operational constraints YAML file.
                May be None if model does not require operational constraints.
        """
        pass
    
    @abstractmethod
    def compute_power_curves(
        self,
        output_path: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None
    ) -> None:
        """Compute power curves and optionally export/plot.
        
        This method calculates the power curve, exports to YAML if output_path
        is provided, and generates plots if showplot or saveplot is True.
        
        Args:
            output_path: Path where power curve YAML will be written. If None,
                no export is performed.
            verbose: Whether to print verbose output.
            showplot: Whether to display plots.
            saveplot: Whether to save plots to file.
            plot_path: Path where plots will be saved if saveplot is True.
        
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def calculate_power_at_wind_speed(
        self,
        wind_speed: float,
        output_path: Path = None,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None
    ) -> float:
        """Calculate power output at a single wind speed.
        
        Args:
            wind_speed: Wind speed in m/s.
            output_path: Path where results will be written. If None,
                no export is performed.
            verbose: Whether to print verbose output.
            showplot: Whether to display plots for this wind speed.
            saveplot: Whether to save plots for this wind speed to file.
            plot_path: Path where plots will be saved if saveplot is True.
            
        Returns:
            Power output in W.
        """
        pass
    