"""Abstract Base Class for kite models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class KiteModel(ABC):
    """Abstract base class for kite models.
    
    All kite models must inherit from this class and implement
    the required methods for loading configuration and computing
    kite dynamics and performance characteristics.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load kite configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def compute_dynamics(self, wind_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Compute kite dynamics for given wind conditions.
        
        :param wind_conditions: Wind condition parameters
        :type wind_conditions: Dict[str, Any]
        :return: Kite dynamics results
        :rtype: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def export_to_yaml(self, output_path: Path) -> None:
        """Export kite model parameters to YAML format.
        
        :param output_path: Path where YAML file will be written
        :type output_path: Path
        """
        pass
