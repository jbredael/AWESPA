"""Abstract Base Class for control models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class ControlModel(ABC):
    """Abstract base class for control system models.
    
    All control models must inherit from this class and implement
    the required methods for loading configuration and computing
    control strategies for the AWE system.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load control configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def compute_control_strategy(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute control strategy for given system state.
        
        :param system_state: Current system state parameters
        :type system_state: Dict[str, Any]
        :return: Control strategy results
        :rtype: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def export_to_yaml(self, output_path: Path) -> None:
        """Export control model parameters to YAML format.
        
        :param output_path: Path where YAML file will be written
        :type output_path: Path
        """
        pass
