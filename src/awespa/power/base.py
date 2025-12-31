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
    def load_configuration(self, config_dir: Path) -> None:
        """Load power model configuration from YAML files.
        
        Expected configuration files in config_dir:
        - airborne.yml: Kite mass, area, and aerodynamic properties
        - ground_gen.yml: Generator and ground system properties
        - tether.yml: Tether properties and constraints
        - wind_resource.yml: Wind resource data and profiles
        - operational_constraints.yml: Operational limits and bounds
        
        :param config_dir: Directory containing configuration YAML files
        :type config_dir: Path
        """
        pass
    
    @abstractmethod
    def compute_power_curves(self, output_path: Path) -> None:
        """Compute power curves for all wind profiles in wind resource.
        
        This method should compute power output for each wind profile
        cluster defined in the wind resource and generate power curves.
        
        :param output_path: Path where power curve YAML will be written
        :type output_path: Path
        """
        pass
