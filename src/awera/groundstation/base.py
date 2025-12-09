"""Abstract Base Class for ground station models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class GroundStationModel(ABC):
    """Abstract base class for ground station models.
    
    All ground station models must inherit from this class and implement
    the required methods for loading configuration and computing
    ground station operations and constraints.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load ground station configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def compute_operations(self, system_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Compute ground station operations for given system conditions.
        
        :param system_conditions: Current system condition parameters
        :type system_conditions: Dict[str, Any]
        :return: Ground station operation results
        :rtype: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def export_to_yaml(self, output_path: Path) -> None:
        """Export ground station model parameters to YAML format.
        
        :param output_path: Path where YAML file will be written
        :type output_path: Path
        """
        pass
