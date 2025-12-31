"""Abstract Base Class for wind profile models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class WindProfileModel(ABC):
    """Abstract base class for wind profile clustering models.
    
    All wind profile models must inherit from this class and implement
    the required methods for loading configuration, clustering wind data,
    and exporting results to YAML format.
    """
    
    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        pass
    
    @abstractmethod
    def cluster(self, data_path: Path, output_path: Path) -> None:
        """Perform wind profile clustering on the input data.
        
        :param data_path: Path to the wind data directory
        :type data_path: Path
        :param output_path: Path where output YAML file will be written
        :type output_path: Path
        """
        pass
