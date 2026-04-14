"""Luchsinger power estimation wrapper for the power-luchsinger package.

This wrapper adapts the Luchsinger power model to the AWESPA modular
architecture. The underlying model accepts awesIO format configuration files
directly and computes power curves using wind shear profiles from a wind resource."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base import PowerEstimationModel

try:
    from power_luchsinger.power_model import PowerModel  # type: ignore
except ImportError as e:
    print(f"Import error for LuchsingerPowerModel: {e}")
    PowerModel = None


class LuchsingerPowerModel(PowerEstimationModel):
    """Wrapper for the Luchsinger pumping kite power model.

    This wrapper adapts the vendored Luchsinger power model to the AWESPA
    modular architecture. The model requires three configuration files in
    awesIO format:

    - System configuration (kite, tether, ground station properties)
    - Wind resource (altitude profiles, clusters, probability matrix)
    - Simulation settings (operational envelope, atmosphere parameters)
    """

    def __init__(self):
        """Initialize the Luchsinger power estimation model."""
        self.model: Optional[Any] = None
        self.systemPath: Optional[Path] = None
        self.windResourceSettingsPath: Optional[Path] = None
        self.simulationSettingsPath: Optional[Path] = None

    def load_configuration(
        self,
        system_path: Path,
        simulation_settings_path: Path,
        operational_constraints_path: Path = None,
        wind_resource_settings_path: Path = None,
    ) -> None:
        """Load power model configuration from YAML files.

        Creates a ``PowerModel`` instance from the vendored Luchsinger package
        using the provided configuration files.

        Args:
            system_path (Path): Path to the system configuration YAML file
                (awesIO format with wing, tether, ground_station components).
            simulation_settings_path (Path): Path to simulation settings YAML
                file containing operational envelope and atmosphere parameters.
            operational_constraints_path (Path): Not used by this model.
                Defaults to None.
            wind_resource_settings_path (Path): Path to wind resource settings YAML file
                containing altitude profiles, clusters, and probability matrix.

        Raises:
            ImportError: If the vendored PowerModel cannot be imported.
            FileNotFoundError: If any required file does not exist.
        """
        if PowerModel is None:
            raise ImportError(
                "PowerModel could not be imported from LuchsingerPowerModel"
            )

        self.systemPath = Path(system_path)
        self.simulationSettingsPath = Path(simulation_settings_path)
        self.windResourceSettingsPath = Path(wind_resource_settings_path) if wind_resource_settings_path else None

        # Validate that required files exist
        for label, path in [
            ("System config", self.systemPath),
            ("Simulation settings", self.simulationSettingsPath),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        if self.windResourceSettingsPath is None:
            raise ValueError(
                "wind_resource_settings_path is required for the Luchsinger model."
            )
        if not self.windResourceSettingsPath.exists():
            raise FileNotFoundError(
                f"Wind resource settings file not found: {self.windResourceSettingsPath}"
            )

        # Create the PowerModel (loads all config internally)
        self.model = PowerModel(
            system_config_path=self.systemPath,
            wind_resource_path=self.windResourceSettingsPath,
            simulation_settings_path=self.simulationSettingsPath,
            validate_file=False,
        )

        print(f"Loaded Luchsinger configuration:")
        print(f"  Wing area:              {self.model.wingArea:.1f} m\u00b2")
        print(f"  Nominal tether force:   {self.model.nominalTetherForce:.0f} N")
        print(f"  Nominal generator power:{self.model.nominalGeneratorPower / 1000:.1f} kW")
        print(f"  Cut-in wind speed:      {self.model.cutInWindSpeed:.1f} m/s")
        print(f"  Cut-out wind speed:     {self.model.cutOutWindSpeed:.1f} m/s")
        print(
            f"  Number of wind clusters:"
            f" {self.model.wind_resource['n_clusters']}"
        )

    def compute_power_curves(
        self,
        wind_speeds: Optional[np.ndarray] = None,
        selected_profiles: list = None,
        output_path: Path = None,
        verbose: bool = True,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None,
    ) -> Dict[str, Any]:
        """Compute power curves for all wind shear profiles and optionally export.

        Args:
            output_path (Path): Path where power curve YAML will be written.
                If None, no export is performed. Defaults to None.
            wind_speeds (np.ndarray): Custom wind speeds to evaluate [m/s].
                If None, uses a linearly-spaced array between cut-in and
                cut-out from simulation settings. Defaults to None.
            verbose (bool): Whether to print a summary after generation.
                Defaults to True.
            showplot (bool): Whether to display plots after generation.
                Defaults to False.
            saveplot (bool): Whether to save plots to file alongside the
                YAML output. Requires ``output_path``. Defaults to False.
            plot_path (Path): Not used directly; plots are saved alongside
                ``output_path`` when ``saveplot`` is True. Defaults to None.

        Returns:
            dict: Power curve data with keys ``'reference_height_m'``,
                ``'operational_altitude_m'``, ``'altitudes'``, and
                ``'profiles'``.

        Raises:
            ValueError: If model is not initialized.
        """
        if self.model is None:
            raise ValueError(
                "Power model not initialized. Call load_configuration first."
            )

        nClusters = self.model.wind_resource["n_clusters"]
        print(f"Computing Luchsinger power curves ({nClusters} wind profile(s))...")

        data = self.model.generate_power_curves(
            wind_speeds=wind_speeds,
            selected_profiles=selected_profiles,
            output_path=output_path,
            verbose=verbose,
            show_plot=showplot,
            save_plot=saveplot,
            validate_file=False,
        )

        return data

    def calculate_power_at_wind_speed(
        self,
        wind_speed: float,
        selected_profiles: list = None,
        output_path: Path = None,
        verbose: bool = True,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None,
    ) -> Union[float, List[float]]:
        """Calculate power output at a single wind speed.

        Uses the first matching wind cluster profile to derive the wind shear
        and calls the Luchsinger model for a single operating point.

        Args:
            wind_speed (float): Wind speed at reference height [m/s].
            selected_profiles (list): Optional list of profile indices.
                If None, all profiles are simulated.
            output_path (Path): Not used; Luchsinger model does not support
                single-point export. Defaults to None.
            verbose (bool): Whether to print the result. Defaults to True.
            showplot (bool): Not used. Defaults to False.
            saveplot (bool): Not used. Defaults to False.
            plot_path (Path): Not used. Defaults to None.

        Returns:
            Union[float, List[float]]: Cycle power output [W]. Returns a
                list with one power value per profile when multiple profiles
                are simulated. Returns a scalar when only one profile is
                requested.

        Raises:
            ValueError: If model is not initialized.
            KeyError: If simulation output does not include a power key.
        """
        if self.model is None:
            raise ValueError(
                "Power model not initialized. Call load_configuration first."
            )

        simulation_data = self.model.simulate_cycle_at_one_wind_speed(
            wind_speed,
            selected_profiles=selected_profiles,
            verbose=verbose,
        )

        if isinstance(simulation_data, dict):
            simulation_data = [simulation_data]

        powers = []
        for profile_data in simulation_data:
            if "cyclePower" in profile_data:
                powers.append(profile_data["cyclePower"])
            elif "average_cycle_power_W" in profile_data:
                powers.append(profile_data["average_cycle_power_W"])
            else:
                raise KeyError(
                    "Simulation output is missing 'cyclePower' and "
                    "'average_cycle_power_W'."
                )

        single_profile_requested = (
            selected_profiles is not None and len(selected_profiles) == 1
        )

        if single_profile_requested or len(powers) == 1:
            return powers[0]

        return powers

    def plot_power_curves(
        self,
        power_curve_path: Path,
        output_dir: Path = None,
    ) -> None:
        """Plot power curves using the vendor's plotting functions.

        Loads the exported YAML and calls the vendor's
        ``plot_comprehensive_analysis`` visualisation.

        Args:
            power_curve_path (Path): Path to the exported power curves YAML.
            output_dir (Path): Directory where plot files will be saved.
                If None, plots are saved alongside ``power_curve_path``.
                Defaults to None.

        Raises:
            FileNotFoundError: If the power curves file does not exist.
            ImportError: If vendor plotting functions cannot be imported.
        """
        try:
            from src.power_luchsinger.plotting import (  # type: ignore
                plot_comprehensive_analysis,
                extract_model_params,
            )
        except ImportError as e:
            raise ImportError(
                f"Could not import vendor plotting functions: {e}"
            )

        powerCurvePath = Path(power_curve_path)
        if not powerCurvePath.exists():
            raise FileNotFoundError(
                f"Power curves file not found: {powerCurvePath}"
            )

        if output_dir is None:
            outputDir = powerCurvePath.parent
        else:
            outputDir = Path(output_dir)
            outputDir.mkdir(parents=True, exist_ok=True)

        savePath = str(outputDir / "power_curve_analysis.pdf")

        print(f"Generating power curve plot from: {powerCurvePath.name}")

        # Re-run power curves to get data object (vendor plotter requires dict,
        # not the YAML path) - use cached model
        data = self.compute_power_curves(verbose=False)

        plot_comprehensive_analysis(
            data,
            extract_model_params(self.model),
            save_path=savePath,
            show=True,
        )

        print(f"Plot saved to: {savePath}")

