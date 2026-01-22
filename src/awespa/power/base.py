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
        simulation_settings_path: Path,
        operational_constraints_path: Path = None
    ) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            system_path: Path to system configuration YAML file.
            simulation_settings_path: Path to simulation settings YAML file.
            operational_constraints_path: Path to operational constraints YAML file.
                May be None if model does not require operational constraints.
        """
        pass
    
    @abstractmethod
    def compute_power_curves(
        self,
        output_path: Path,
        plot: bool = False
    ) -> None:
        """Compute power curves and optionally export/plot.
        
        This method calculates the power curve, exports to YAML if output_path
        is provided, and generates plots if plot is True.
        
        Args:
            output_path: Path where power curve YAML will be written. If None,
                no export is performed.
            plot: Whether to generate and display plots.
        
        Returns:
            None
        """
        pass
    
    @abstractmethod
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
        pass
    