"""Luchsinger power estimation wrapper for the vendored LuchsingerModel repository."""

import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Any, Optional, List

from .base import PowerEstimationModel

# Add vendor path to import the Luchsinger model code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "LuchsingerModel"
sys.path.insert(0, str(VENDOR_PATH))

# Import vendor Luchsinger model functionality
try:
    from src.core.power_model import PowerModel
except ImportError as e:
    print(f"Import error for Luchsinger model: {e}")
    PowerModel = None


class LuchsingerPowerModel(PowerEstimationModel):
    """Wrapper for the Luchsinger pumping kite power model.
    
    This wrapper adapts the vendored Luchsinger power model to the AWESPA
    modular architecture, handling YAML-based configuration and power curve
    computation for wind profile clusters.
    
    The Luchsinger model is a simplified analytical model that computes
    power output for pumping kite systems based on:
    - Kite aerodynamic properties (lift/drag coefficients, area)
    - Tether properties (length, diameter, drag)
    - Ground station characteristics (force limits, generator power)
    - Operational constraints (cut-in/cut-out speeds, elevation angles)
    """
    
    def __init__(self):
        """Initialize the Luchsinger power estimation model."""
        self.airborne_config: Optional[Dict[str, Any]] = None
        self.tether_config: Optional[Dict[str, Any]] = None
        self.wind_resource: Optional[Dict[str, Any]] = None
        self.operational_constraints: Optional[Dict[str, Any]] = None
        
        # Luchsinger model instance (type set dynamically due to vendor import)
        self.power_model = None
        self.power_curve_results: Optional[Dict[str, Any]] = None
        
    def load_configuration(self, 
                          airborne_path: Path,
                          tether_path: Path,
                          operational_constraints_path: Path,
                          ground_station_path: Path,
                          wind_resource_path: Path) -> None:
        """Load power model configuration from YAML files.
        
        Args:
            airborne_path: Path to airborne configuration YAML file.
            tether_path: Path to tether configuration YAML file.
            operational_constraints_path: Path to operational constraints YAML file.
            ground_station_path: Path to ground station configuration YAML file.
            wind_resource_path: Path to wind resource YAML file.
        """
        config_files = {
            'airborne': airborne_path,
            'tether': tether_path,
            'operational_constraints': operational_constraints_path,
            'ground_station': ground_station_path,
        }
        
        # Load all configuration files
        for config_name, config_path in config_files.items():
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_name == 'airborne':
                self.airborne_config = config_data
            elif config_name == 'tether':
                self.tether_config = config_data
            elif config_name == 'operational_constraints':
                self.operational_constraints = config_data
            elif config_name == 'ground_station':
                self.ground_station_config = config_data
                
        # Load wind resource
        wind_resource_path = Path(wind_resource_path)
        if not wind_resource_path.exists():
            raise FileNotFoundError(f"Wind resource file not found: {wind_resource_path}")
            
        with open(wind_resource_path, 'r') as f:
            self.wind_resource = yaml.load(f, Loader=yaml.FullLoader)
        
        # Create Luchsinger model configuration and initialize
        self._create_luchsinger_config()
        
    def _create_luchsinger_config(self) -> None:
        """Create configuration compatible with the Luchsinger PowerModel."""
        
        # Get parameters from dedicated sections
        aero_props = self.airborne_config.get('aerodynamic_properties', {})
        generator = self.ground_station_config.get('generator', {})
        winch = self.ground_station_config.get('winch', {})
        storage = self.ground_station_config.get('storage', {})
        wind_limits = self.operational_constraints.get('wind_limits', {})
        atmosphere = self.operational_constraints.get('atmosphere', {})
        model_description = self.operational_constraints.get('metadata', {}).get('model_description', '')

        # Map AWESPA config to Luchsinger config format
        luchsinger_config = {
            'kite': {
                'wingArea': self.airborne_config['kite']['projected_area_m2'],
                'liftCoefficientOut': self.airborne_config['kite']['lift_coefficient']['powered'],
                'dragCoefficientOut': self.airborne_config['kite']['drag_coefficient']['powered'],
                'dragCoefficientIn': self.airborne_config['kite']['drag_coefficient']['depowered'],
                'flatteningFactor': aero_props.get('flattening_factor'),
                'areaDensity': aero_props.get('area_density_kg_m2'),
            },
            'tether': {
                'maxLength': self.tether_config['tether']['length_m'],
                'minLength': self.tether_config['tether'].get('min_length_m', self.tether_config['tether']['length_m'] * 0.5),
                'dragCoefficient': self.tether_config['tether']['drag_coefficient'],
                'diameter': self.tether_config['tether']['diameter_m'],
            },
            'atmosphere': {
                'airDensity': atmosphere.get('air_density_kg_m3'),
                'temperature': atmosphere.get('temperature_k'),
                'pressure': atmosphere.get('pressure_pa'),
                'viscosity': atmosphere.get('kinematic_viscosity_m2_s'),
            },
            'groundStation': {
                'nominalTetherForce': winch.get('nominal_tether_force_n'),
                'nominalGeneratorPower': generator.get('nominal_power_w'),
                'reelOutSpeedLimit': winch.get('reel_out_speed_limit_m_s'),
                'reelInSpeedLimit': winch.get('reel_in_speed_limit_m_s'),
                'generatorEfficiency': generator.get('efficiency'),
                'storageEfficiency': storage.get('efficiency'),
            },
            'operational': {
                'cutInWindSpeed': wind_limits.get('cut_in_wind_speed_m_s'),
                'cutOutWindSpeed': wind_limits.get('cut_out_wind_speed_m_s'),
                'elevationAngleOut': self.operational_constraints['bounds']['avg_elevation_deg']['min'],
                'elevationAngleIn': self.operational_constraints['bounds']['avg_elevation_deg']['max'],
            },
            'model': {
                'name': 'luchsinger_awespa_wrapper',
                'version': '1.0.0',
                'description': model_description,
            }
        }
        
        # Initialize the Luchsinger PowerModel
        if PowerModel is not None:
            self.power_model = PowerModel(luchsinger_config)
        else:
            raise ImportError("Luchsinger PowerModel could not be imported")
            
    def compute_power_curves(self, wind_speeds: np.ndarray = None) -> Dict[str, Any]:
        """Compute power curve for constant wind profile.
        
        The simplified Luchsinger model assumes constant wind speed (no wind shear).
        This method generates a power curve for a range of wind speeds.
        Output follows the AWESIO power_curves_schema format.
        
        Args:
            wind_speeds (np.ndarray): Array of wind speeds to evaluate. If None,
                uses range from cut-in to cut-out with 0.5 m/s steps.
                
        Returns:
            Dict: Power curve data matching AWESIO power_curves_schema format.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
        
        # Define wind speed range if not provided
        if wind_speeds is None:
            wind_speeds = np.arange(
                self.power_model.cutInWindSpeed,
                self.power_model.cutOutWindSpeed + 0.5,
                0.5
            )
        
        # Compute power for each wind speed, extracting all relevant outputs
        cycle_powers = []
        reel_out_powers = []
        reel_in_powers = []
        reel_out_times = []
        reel_in_times = []
        cycle_times = []
        
        for wind_speed in wind_speeds:
            result = self.power_model.calculate_power(windSpeed=wind_speed)
            cycle_powers.append(float(result['cyclePower']))
            reel_out_powers.append(float(result['reelOutPower']))
            reel_in_powers.append(float(result['reelInPower']))
            reel_out_times.append(float(result['reelOutTime']))
            reel_in_times.append(float(result['reelInTime']))
            cycle_times.append(float(result['reelOutTime'] + result['reelInTime']))
        
        # Get operating altitude from tether config
        operating_altitude = float(self.power_model.operationalLength)
        
        # Generate altitudes array (0 to max tether length)
        n_altitudes = 51  # Standard number of altitude points
        altitudes = np.linspace(0, self.power_model.tetherMaxLength, n_altitudes).tolist()
        
        # For simplified model: u_normalized = 1 at all altitudes, v_normalized = 0
        u_normalized = [1.0] * len(altitudes)
        v_normalized = [0.0] * len(altitudes)
        
        # Build power curve data structure matching AWESIO schema
        # Use OrderedDict to maintain field ordering: metadata, altitudes_m, reference_wind_speeds_m_s, power_curves
        power_curve_data = OrderedDict([
            ('metadata', OrderedDict([
                ('time_created', datetime.now().isoformat()),
                ('model_config', OrderedDict([
                    ('wing_area_m2', float(self.power_model.wingArea)),
                    ('nominal_power_w', float(self.power_model.nominalGeneratorPower)),
                    ('nominal_tether_force_n', float(self.power_model.nominalTetherForce)),
                    ('cut_in_wind_speed_m_s', float(self.power_model.cutInWindSpeed)),
                    ('cut_out_wind_speed_m_s', float(self.power_model.cutOutWindSpeed)),
                    ('operating_altitude_m', operating_altitude),
                    ('tether_length_operational_m', float(self.power_model.operationalLength)),
                ])),
            ])),
            ('altitudes_m', altitudes),
            ('reference_wind_speeds_m_s', wind_speeds.tolist()),
            ('power_curves', [
                OrderedDict([
                    ('profile_id', 1),
                    ('speed_ratio_at_operating_altitude', 1.0),
                    ('u_normalized', u_normalized),
                    ('v_normalized', v_normalized),
                    ('probability_weight', 1.0),
                    ('cycle_power_w', cycle_powers),
                    ('reel_out_power_w', reel_out_powers),
                    ('reel_in_power_w', reel_in_powers),
                    ('reel_out_time_s', reel_out_times),
                    ('reel_in_time_s', reel_in_times),
                    ('cycle_time_s', cycle_times),
                ])
            ]),
        ])
        
        # Store results
        self.power_curve_results = power_curve_data
        
        return power_curve_data
            
    def export_to_yaml(self, output_path: Path) -> None:
        """Export power curve results to YAML file.
        
        Args:
            output_path (Path): Path where power curve YAML will be written.
            
        Raises:
            ValueError: If no power curve results available.
        """
        if self.power_curve_results is None:
            raise ValueError("No power curve results to export. Run compute_power_curves first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use custom representer to preserve OrderedDict order in YAML output
        def represent_ordereddict(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
        
        yaml.add_representer(OrderedDict, represent_ordereddict)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.power_curve_results, f, default_flow_style=False, sort_keys=False)
        
        print(f"Power curve exported to: {output_path}")
    
    def _get_operating_altitude(self, 
                                 altitudes: np.ndarray, 
                                 speed_normalized: np.ndarray) -> float:
        """Determine optimal operating altitude based on wind profile.
        
        Args:
            altitudes (np.ndarray): Array of altitudes in m.
            speed_normalized (np.ndarray): Normalized wind speed at each altitude.
            
        Returns:
            float: Operating altitude in m.
        """
        # Get operational bounds from tether config
        min_altitude = self.tether_config['tether'].get('min_length_m', 100.0) * 0.5
        max_altitude = self.tether_config['tether']['length_m'] * 0.9
        
        # Find altitude with maximum wind within operational bounds
        valid_mask = (altitudes >= min_altitude) & (altitudes <= max_altitude)
        
        if not np.any(valid_mask):
            # If no altitudes in range, use mid-tether altitude
            return (min_altitude + max_altitude) / 2
        
        valid_altitudes = altitudes[valid_mask]
        valid_speeds = speed_normalized[valid_mask]
        
        max_idx = np.argmax(valid_speeds)
        return float(valid_altitudes[max_idx])
    

    def get_power_at_wind_speed(self, wind_speed: float) -> float:
        """Get power output for a given wind speed.
        
        Args:
            wind_speed (float): Wind speed in m/s.
            
        Returns:
            float: Power output in W.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized")
        
        return self.power_model.calculate_power(windSpeed=wind_speed)
    
    def get_power_curve(self, 
                        wind_speeds: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Get power curve as arrays.
        
        Args:
            wind_speeds (np.ndarray): Wind speeds to evaluate. If None, uses
                default range from cut-in to cut-out.
                
        Returns:
            Dict with 'wind_speed' and 'power' arrays.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized")
        
        if wind_speeds is None:
            wind_speeds = np.linspace(
                self.power_model.cutInWindSpeed,
                self.power_model.cutOutWindSpeed,
                100
            )
        
        powers = self.power_model.calculate_power(windSpeed=wind_speeds)
        
        return {
            'wind_speed': wind_speeds,
            'power': powers
        }
