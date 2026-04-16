"""Abstract Base Class for wind profile models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class WindProfileModel(ABC):
    """Abstract base class for wind profile clustering models.

    All wind profile models must inherit from this class and implement
    the required abstract methods. Non-abstract methods (``cluster``,
    ``fit_profile``, ``prescribe_profile``) define optional interfaces
    that implementations may override.
    """

    @abstractmethod
    def load_configuration(self, config_path: Path, validate: bool = True) -> None:
        """Load configuration parameters from a YAML file.

        Args:
            config_path (Path): Path to the YAML configuration file.
            validate (bool): If True, validate configuration files using
                the awesIO validator. Defaults to True.
        """
        pass

    def cluster(
        self,
        data_path: Path,
        output_path: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> None:
        """Perform wind profile clustering on the input data.

        Args:
            data_path (Path): Path to the wind data directory.
            output_path (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots during clustering.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.
        """
        pass

    def fit_profile(
        self,
        data_path: Path,
        output_path: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Fit an analytical wind profile to measured wind data.

        Args:
            data_path (Path): Path to the wind data directory.
            output_path (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots after fitting.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.

        Returns:
            Dict[str, Any]: Fitting results.
        """
        pass

    def prescribe_profile(
        self,
        output_path: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Build a prescribed analytical wind profile without measured data.

        Args:
            output_path (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots after prescribing.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.

        Returns:
            Dict[str, Any]: Prescribed profile results.
        """
        pass
