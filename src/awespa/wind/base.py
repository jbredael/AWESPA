"""Abstract Base Class for wind profile models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class WindProfileModel(ABC):
    """Abstract base class for wind profile clustering models.

    All wind profile models must inherit from this class and implement
    the required methods for loading configuration, clustering wind data,
    and exporting results to YAML format.
    """

    @abstractmethod
    def load_from_yaml(self, config_path: Path) -> None:
        """Load configuration parameters from a YAML file.

        Args:
            config_path (Path): Path to the YAML configuration file.
        """
        pass

    @abstractmethod
    def cluster(
        self,
        data_path: Path,
        output_path: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        plotpath: Optional[Path] = None,) -> None:
        """Perform wind profile clustering on the input data.

        Args:
            data_path (Path): Path to the wind data directory.
            output_path (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots during clustering.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            plotpath (Optional[Path]): Directory path where plots are saved.
                Required if saveplot is True. Defaults to None.
        """
        pass
