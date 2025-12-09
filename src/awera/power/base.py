"""Abstract Base Class for power estimation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class PowerEstimationModel(ABC):
    """Abstract base class for power estimation models.
    
    All power estimation models must inherit from this class and implement
    the required methods for loading configuration, computing power output,
    and exporting power curves.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load power model configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def compute_power(self, wind_profiles_path: Path, optimal_controls_path: Path, output_path: Path) -> None:
        """Compute power output based on wind profiles and optimal controls.
        
        :param wind_profiles_path: Path to wind profiles YAML file
        :type wind_profiles_path: Path
        :param optimal_controls_path: Path to optimal controls YAML file
        :type optimal_controls_path: Path
        :param output_path: Path where power curve YAML will be written
        :type output_path: Path
        """
        pass
    
    @abstractmethod
    def export_to_yaml(self, output_path: Path) -> None:
        """Export power curve to YAML format.
        
        The output YAML must contain the power curve data that
        can be used for AEP calculations and analysis.
        
        :param output_path: Path where YAML file will be written
        :type output_path: Path
        """
        pass
