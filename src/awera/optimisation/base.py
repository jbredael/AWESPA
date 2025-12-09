"""Abstract Base Class for optimisation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class Optimiser(ABC):
    """Abstract base class for optimisation models.
    
    All optimisation models must inherit from this class and implement
    the required methods for loading configuration, optimising system
    parameters, and exporting optimal control settings.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load optimisation configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def optimise(self, wind_profiles_path: Path, output_path: Path) -> None:
        """Perform system optimisation based on wind profiles.
        
        :param wind_profiles_path: Path to wind profiles YAML file
        :type wind_profiles_path: Path
        :param output_path: Path where optimal controls YAML will be written
        :type output_path: Path
        """
        pass
    
    @abstractmethod
    def export_to_yaml(self, output_path: Path) -> None:
        """Export optimal control settings to YAML format.
        
        The output YAML must contain optimal control parameters that
        can be consumed by the power estimation and other components.
        
        :param output_path: Path where YAML file will be written
        :type output_path: Path
        """
        pass
