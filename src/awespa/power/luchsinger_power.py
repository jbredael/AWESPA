"""Luchsinger power estimation wrapper for the vendored LuchsingerModel repository."""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import PowerEstimationModel

# Add vendor path to import the Luchsinger model code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "LuchsingerModel"
sys.path.insert(0, str(VENDOR_PATH))

# Import vendor Luchsinger model functionality
try:
    from src.core.power_model import PowerModel
    from src.core.utils import WindProfile, load_wind_resource
except ImportError as e:
    print(f"Import error for Luchsinger model: {e}")
    PowerModel = None
    WindProfile = None
    load_wind_resource = None


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
        
    def load_configuration(self, config_dir: Path) -> None:
        """Load power model configuration from YAML files.
        
        Expected configuration files in config_dir:
        - airborne.yml: Kite mass, area, and aerodynamic properties
        - tether.yml: Tether properties and constraints
        - operational_constraints.yml: Operational limits and bounds
        
        Wind resource is loaded from results/wind_resource.yml.
        
        Args:
            config_dir (Path): Directory containing configuration YAML files.
        """
        config_files = {
            'airborne': config_dir / 'airborne.yml',
            'tether': config_dir / 'tether.yml',
            'operational_constraints': config_dir / 'operational_constraints.yml'
        }
        
        # Wind resource is in results directory, not config
        wind_resource_path = config_dir.parent / 'results' / 'wind_resource.yml'
        
        # Load all configuration files
        for config_name, config_path in config_files.items():
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
                
        # Load wind resource from results directory
        if not wind_resource_path.exists():
            raise FileNotFoundError(f"Wind resource file not found: {wind_resource_path}")
            
        with open(wind_resource_path, 'r') as f:
            self.wind_resource = yaml.load(f, Loader=yaml.FullLoader)
        
        # Create Luchsinger model configuration and initialize
        self._create_luchsinger_config()
        
    def _create_luchsinger_config(self) -> None:
        """Create configuration compatible with the Luchsinger PowerModel."""
        # Extract bounds from operational constraints
        bounds = self.operational_constraints.get('bounds', {})
        op_limits = self.operational_constraints.get('operational_limits', {})
        
        # Get tether force bounds (vendor uses kgf, convert to N: max * 9.806)
        force_limits = bounds.get('force_limits', {})
        max_tether_force = force_limits.get('max', 500) * 9.806  # kgf -> N
        
        # Get speed limits
        speed_limits = bounds.get('speed_limits', {})
        max_reel_speed = speed_limits.get('max', 15.0)
        
        # Map AWESPA config to Luchsinger config format
        luchsinger_config = {
            'kite': {
                'wingArea': self.airborne_config['projected_area'],
                'liftCoefficientOut': self.airborne_config['lift_coefficient']['powered'],
                'dragCoefficientOut': self.airborne_config['drag_coefficient']['powered'],
                'dragCoefficientIn': self.airborne_config['drag_coefficient']['depowered'],
                'flatteningFactor': self.airborne_config.get('flattening_factor', 0.9),
                'areaDensity': self.airborne_config.get('area_density', 0.108),
            },
            'tether': {
                'maxLength': self.tether_config['length'],
                'minLength': self.tether_config.get('min_length', self.tether_config['length'] * 0.5),
                'dragCoefficient': self.tether_config['drag_coefficient'],
                'diameter': self.tether_config['diameter'],
            },
            'atmosphere': {
                'airDensity': self.operational_constraints.get('air_density', 1.225),
                'temperature': self.operational_constraints.get('temperature', 288.15),
                'pressure': self.operational_constraints.get('pressure', 101325.0),
            },
            'groundStation': {
                'nominalTetherForce': max_tether_force,
                'nominalGeneratorPower': self.operational_constraints.get(
                    'max_generator_power', 100000.0
                ),
                'drumOuterRadius': self.operational_constraints.get('drum_outer_radius', 0.5),
                'drumInnerRadius': self.operational_constraints.get('drum_inner_radius', 0.49),
                'reelOutSpeedLimit': max_reel_speed,
                'reelInSpeedLimit': max_reel_speed * 2,  # Typically reel-in is faster
            },
            'operational': {
                'cutInWindSpeed': op_limits.get('min_wind_speed', 4.0),
                'cutOutWindSpeed': op_limits.get('max_wind_speed', 25.0),
                'elevationAngleOut': bounds.get('avg_elevation', {}).get('min', 25.0),
                'elevationAngleIn': bounds.get('max_elevation', {}).get('max', 45.0) if isinstance(
                    bounds.get('max_elevation'), dict) else 45.0,
            },
            'model': {
                'name': 'luchsinger_awespa_wrapper',
                'version': '1.0.0',
            }
        }
        
        # Initialize the Luchsinger PowerModel
        if PowerModel is not None:
            self.power_model = PowerModel(luchsinger_config)
        else:
            raise ImportError("Luchsinger PowerModel could not be imported")
            
    def compute_power_curves(self, output_path: Path) -> None:
        """Compute power curves for all wind profiles in wind resource.
        
        This method computes power output for each wind profile cluster
        defined in the wind resource and generates power curves.
        
        Args:
            output_path (Path): Path where power curve YAML will be written.
        """
        if self.power_model is None:
            raise ValueError("Power model not initialized. Call load_configuration first.")
            
        if self.wind_resource is None:
            raise ValueError("Wind resource not loaded. Call load_configuration first.")
        
        # Extract wind speed bins
        wind_speed_bins = self.wind_resource['wind_speed_bins']
        bin_centers = np.array(wind_speed_bins['bin_centers_m_s'])
        
        # Get reference height and altitudes
        reference_height = self.wind_resource['metadata'].get('reference_height_m', 100.0)
        altitudes = np.array(self.wind_resource['altitudes'])
        
        # Get probability matrix and calculate cluster frequencies
        probability_matrix = np.array(self.wind_resource['probability_matrix']['data'])
        # Sum probabilities across wind speed bins for each cluster to get total frequency
        cluster_frequencies = np.sum(probability_matrix, axis=1) / 100.0  # Convert from % to fraction
        
        # Compute power curves for each cluster
        cluster_power_curves = []
        
        for i, cluster in enumerate(self.wind_resource['clusters']):
            cluster_id = cluster['id']
            frequency = float(cluster_frequencies[i])
            
            # Create WindProfile from cluster data
            u_normalized = np.array(cluster['u_normalized'])
            v_normalized = np.array(cluster['v_normalized'])
            
            # Calculate normalized wind speed at each altitude
            speed_normalized = np.sqrt(u_normalized**2 + v_normalized**2)
            
            # Determine operating altitude (typically where max wind is)
            operating_altitude = self._get_operating_altitude(altitudes, speed_normalized)
            
            # Get speed ratio at operating altitude vs reference height
            speed_ratio = np.interp(operating_altitude, altitudes, speed_normalized)
            
            # Compute power for each reference wind speed bin
            powers = []
            for ref_wind_speed in bin_centers:
                # Scale wind speed to operating altitude
                effective_wind_speed = ref_wind_speed * speed_ratio
                
                # Compute power using Luchsinger model
                power = self.power_model.calculate_power(
                    windSpeed=effective_wind_speed,
                    altitude=operating_altitude
                )
                powers.append(float(power))
            
            cluster_power_curves.append({
                'cluster_id': int(cluster_id),
                'frequency': float(frequency),
                'operating_altitude_m': float(operating_altitude),
                'speed_ratio': float(speed_ratio),
                'power_values_w': powers,
            })
        
        # Build output data structure
        power_curve_data = {
            'metadata': {
                'model': 'Luchsinger',
                'version': '1.0.0',
                'reference_height_m': reference_height,
                'n_clusters': len(cluster_power_curves),
                'n_wind_speed_bins': len(bin_centers),
            },
            'wind_speed_bins': {
                'bin_centers_m_s': bin_centers.tolist(),
                'bin_edges_m_s': wind_speed_bins['bin_edges_m_s'],
            },
            'cluster_power_curves': cluster_power_curves,
            'aggregate_power_curve': self._compute_aggregate_power_curve(
                cluster_power_curves, bin_centers
            ),
        }
        
        # Store results
        self.power_curve_results = power_curve_data
        
        # Save to YAML
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(power_curve_data, f, default_flow_style=False)
            
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
        min_altitude = self.tether_config.get('min_length', 100.0) * 0.5
        max_altitude = self.tether_config['length'] * 0.9
        
        # Find altitude with maximum wind within operational bounds
        valid_mask = (altitudes >= min_altitude) & (altitudes <= max_altitude)
        
        if not np.any(valid_mask):
            # If no altitudes in range, use mid-tether altitude
            return (min_altitude + max_altitude) / 2
        
        valid_altitudes = altitudes[valid_mask]
        valid_speeds = speed_normalized[valid_mask]
        
        max_idx = np.argmax(valid_speeds)
        return float(valid_altitudes[max_idx])
    
    def _compute_aggregate_power_curve(self,
                                        cluster_power_curves: List[Dict],
                                        bin_centers: np.ndarray) -> Dict[str, Any]:
        """Compute frequency-weighted aggregate power curve.
        
        Args:
            cluster_power_curves (List[Dict]): Power curves for each cluster.
            bin_centers (np.ndarray): Wind speed bin centers.
            
        Returns:
            Dict[str, Any]: Aggregate power curve data.
        """
        # Initialize aggregate power array
        aggregate_power = np.zeros(len(bin_centers))
        total_frequency = 0.0
        
        for curve in cluster_power_curves:
            frequency = curve['frequency']
            powers = np.array(curve['power_values_w'])
            aggregate_power += frequency * powers
            total_frequency += frequency
        
        # Normalize by total frequency (should be ~1.0)
        if total_frequency > 0:
            aggregate_power /= total_frequency
        
        return {
            'wind_speed_m_s': bin_centers.tolist(),
            'power_w': aggregate_power.tolist(),
            'max_power_w': float(np.max(aggregate_power)),
            'rated_wind_speed_m_s': float(bin_centers[np.argmax(aggregate_power)]),
        }
        
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
