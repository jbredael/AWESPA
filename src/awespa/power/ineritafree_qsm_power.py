"""Inertia-Free QSM power estimation wrapper for the vendored InertiaFree-QSM repository.

This wrapper adapts the vendored Inertia-Free Quasi-Steady Model to the AWESPA
modular architecture. The underlying model accepts awesIO format configuration
files directly and supports both direct simulation and optimization-based power
curve generation.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import PowerEstimationModel

# Add vendor path to import the InertiaFree-QSM code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "InertiaFree-QSM"
sys.path.insert(0, str(VENDOR_PATH / "src"))

try:
    from inertiafree_qsm import PowerCurveConstructor  # type: ignore
except ImportError as e:
    print(f"Import error for InertiaFree-QSM: {e}")
    PowerCurveConstructor = None


class InertiaFreeQSMPowerModel(PowerEstimationModel):
    """Wrapper for the Inertia-Free Quasi-Steady Model power curve constructor.

    This wrapper adapts the vendored InertiaFree-QSM to the AWESPA modular
    architecture. It supports two power curve generation methods:

    1. Direct simulation: Fast, uses pre-defined cycle parameters.
    2. Optimization: Slower, finds optimal cycle parameters per wind speed.

    The model requires three configuration files in awesIO format:
    - System configuration (kite, tether, ground station properties)
    - Wind resource (altitude profiles, clusters, probability matrix)
    - Simulation settings (cycle parameters, optimizer bounds, phase settings)
    """

    def __init__(self):
        """Initialize the Inertia-Free QSM power estimation model."""
        self.constructor: Optional[Any] = None
        self.systemPath: Optional[Path] = None
        self.windResourcePath: Optional[Path] = None
        self.simulationSettingsPath: Optional[Path] = None

    def load_configuration(
        self,
        system_path: Path,
        simulation_settings_path: Path,
        operational_constraints_path: Path = None,
        wind_resource_path: Path = None,
    ) -> None:
        """Load power model configuration from YAML files.

        Creates a ``PowerCurveConstructor`` instance from the vendored
        InertiaFree-QSM package using the provided configuration files.

        Args:
            system_path (Path): Path to the system configuration YAML file
                (awesIO format with wing, tether, ground_station components).
            simulation_settings_path (Path): Path to simulation settings YAML
                file containing cycle, phase, optimizer, and solver parameters.
            operational_constraints_path (Path): Not used by this model.
                Defaults to None.
            wind_resource_path (Path): Path to wind resource YAML file
                containing altitude profiles, clusters, and probability matrix.

        Raises:
            ImportError: If the vendored PowerCurveConstructor cannot be imported.
            FileNotFoundError: If any required file does not exist.
        """
        if PowerCurveConstructor is None:
            raise ImportError(
                "PowerCurveConstructor could not be imported from InertiaFree-QSM"
            )

        self.systemPath = Path(system_path)
        self.simulationSettingsPath = Path(simulation_settings_path)
        self.windResourcePath = Path(wind_resource_path) if wind_resource_path else None

        # Validate that required files exist
        for label, path in [
            ("System config", self.systemPath),
            ("Simulation settings", self.simulationSettingsPath),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        if self.windResourcePath is not None and not self.windResourcePath.exists():
            raise FileNotFoundError(
                f"Wind resource file not found: {self.windResourcePath}"
            )

        # Create the PowerCurveConstructor
        self.constructor = PowerCurveConstructor(
            system_config_path=self.systemPath,
            wind_resource_path=self.windResourcePath,
            simulation_settings_path=self.simulationSettingsPath,
        )

        print(f"Loaded InertiaFree-QSM configuration:")
        self.constructor.print_summary()

    def compute_power_curves(
        self,
        output_path: Path = None,
        method: str = "direct",
        wind_speeds: Optional[np.ndarray] = None,
        cluster_ids: Optional[List[int]] = None,
        verbose: bool = True,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None,
    ) -> Dict[str, Any]:
        """Compute power curves using direct simulation or optimization.

        Args:
            output_path (Path): Path where power curve YAML will be written.
                If None, no export is performed. Defaults to None.
            method (str): Simulation method, either ``'direct'`` or
                ``'optimization'``. Defaults to ``'direct'``.
            wind_speeds (np.ndarray): Custom wind speeds to evaluate [m/s].
                If None, uses wind speeds from simulation settings.
                Defaults to None.
            cluster_ids (list): Cluster IDs (1-indexed) to calculate. If None,
                calculates all clusters. Defaults to None.
            verbose (bool): Whether to print progress output.
                Defaults to True.
            showplot (bool): Whether to display plots after generation.
                Defaults to False.
            saveplot (bool): Whether to save plots to file. Requires
                ``output_path``. Defaults to False.
            plot_path (Path): Path where plots will be saved. If None and
                ``saveplot`` is True, plots are saved alongside the YAML.
                Defaults to None.

        Returns:
            dict: Power curve data in awesIO format.

        Raises:
            ValueError: If model is not initialized or method is unknown.
        """
        if self.constructor is None:
            raise ValueError(
                "Power model not initialized. Call load_configuration first."
            )

        if method == "direct":
            print(f"Computing power curves using direct simulation...")
            data = self.constructor.generate_power_curves_direct(
                wind_speeds=wind_speeds,
                cluster_ids=cluster_ids,
                output_path=output_path,
                verbose=verbose,
                show_plot=showplot,
                save_plot=saveplot,
            )
        elif method == "optimization":
            print(f"Computing power curves using optimization...")
            data = self.constructor.generate_power_curves_optimized(
                wind_speeds=wind_speeds,
                cluster_ids=cluster_ids,
                output_path=output_path,
                verbose=verbose,
                show_plot=showplot,
                save_plot=saveplot,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'direct' or 'optimization'."
            )

        if output_path is not None:
            print(f"Power curves exported to: {output_path}")

        return data

    def calculate_power_at_wind_speed(
        self,
        wind_speed: float,
        method: str = "direct",
        cluster_id: int = 1,
        output_path: Path = None,
        verbose: bool = True,
        showplot: bool = False,
        saveplot: bool = False,
        plot_path: Path = None,
    ) -> float:
        """Calculate power output at a single wind speed.

        Simulates a single pumping cycle at the given wind speed and returns
        the average cycle power.

        Args:
            wind_speed (float): Wind speed at reference height [m/s].
            method (str): Simulation method, either ``'direct'`` or
                ``'optimization'``. Defaults to ``'direct'``.
            cluster_id (int): Cluster ID (1-indexed) for wind profile
                selection. Defaults to 1.
            output_path (Path): Path where results YAML will be written.
                If None, no export is performed. Defaults to None.
            verbose (bool): Whether to print verbose output.
                Defaults to True.
            showplot (bool): Whether to display cycle detail plot.
                Defaults to False.
            saveplot (bool): Whether to save cycle detail plot. Requires
                ``output_path``. Defaults to False.
            plot_path (Path): Path where plots will be saved. Defaults to None.

        Returns:
            float: Average cycle power output [W].

        Raises:
            ValueError: If model is not initialized.
        """
        if self.constructor is None:
            raise ValueError(
                "Power model not initialized. Call load_configuration first."
            )

        result = self.constructor.simulate_single_wind_speed(
            wind_speed=wind_speed,
            cluster_id=cluster_id,
            method=method,
            output_path=output_path,
            verbose=verbose,
            show_plot=showplot,
            save_plot=saveplot,
        )

        # Extract average cycle power from result
        powerCurves = result.get("power_curves", [])
        if powerCurves:
            windSpeedData = powerCurves[0].get("wind_speed_data", [])
            if windSpeedData:
                power = windSpeedData[0]["performance"]["power"][
                    "average_cycle_power_w"
                ]
                if verbose:
                    print(
                        f"\nPower at {wind_speed:.1f} m/s ({method}): "
                        f"{power:.2f} W ({power / 1000:.2f} kW)"
                    )
                return float(power)

        print(f"Warning: Could not extract power from result at {wind_speed:.1f} m/s")
        return 0.0
