"""AWE power estimation wrapper for the vendored AWE_production_estimation repository."""

import sys
import yaml
from yaml import UnsafeLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base import PowerEstimationModel

# Add vendor path to import the AWE production estimation code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "AWE_production_estimation"
sys.path.insert(0, str(VENDOR_PATH))

# Also add the awe_pe path specifically
AWE_PE_PATH = VENDOR_PATH / "awe_pe"
sys.path.insert(0, str(AWE_PE_PATH))

# Import vendor AWE production estimation functionality
try:
    import qsm
    import cycle_optimizer
    import utils
    from cycle_optimizer import OptimizerCycle
    from qsm import LogProfile, NormalisedWindTable1D, TractionPhasePattern, SystemProperties
    from utils import parse_system_properties_and_bounds, parse_opt_variables, parse_constraints
except ImportError as e:
    print(f"Import error: {e}")
    # Handle import errors gracefully during development
    OptimizerCycle = None
    LogProfile = None
    TractionPhasePattern = None
    SystemProperties = None
    parse_system_properties_and_bounds = None
    parse_opt_variables = None
    parse_constraints = None


class AWEPowerEstimationModel(PowerEstimationModel):
    """Wrapper for the AWE_production_estimation repository.
    
    This wrapper adapts the vendored AWE power estimation functionality
    to the AWESPA modular architecture, handling YAML-based configuration
    and power curve computation for wind profile clusters.
    """
    
    def __init__(self):
        """Initialize the AWE power estimation model."""
        self.config: Optional[Dict[str, Any]] = None
        self.airborne_config: Optional[Dict[str, Any]] = None
        self.ground_gen_config: Optional[Dict[str, Any]] = None
        self.tether_config: Optional[Dict[str, Any]] = None
        self.wind_resource: Optional[Dict[str, Any]] = None
        self.operational_constraints: Optional[Dict[str, Any]] = None
        
        # AWE PE components
        self.system_properties: Optional[SystemProperties] = None
        self.environment_profile: Optional[LogProfile] = None
        self.power_curve_results: Optional[Dict[str, Any]] = None
        
        # Cache for converted wind profiles
        self._wind_profiles = {}
        
    def load_configuration(self, config_dir: Path) -> None:
        """Load power model configuration from YAML files.
        
        :param config_dir: Directory containing configuration YAML files
        :type config_dir: Path
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
                config_data = yaml.load(f, Loader=UnsafeLoader)
            
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
            self.wind_resource = yaml.load(f, Loader=UnsafeLoader)
        
        # Create unified configuration for the vendor system
        self._create_vendor_configuration()
        
    def _create_vendor_configuration(self) -> None:
        """Create unified configuration compatible with AWE production estimation."""
        
        # Combine configurations into format expected by vendor code
        self.config = {
            'kite': {
                'mass': self.airborne_config['mass'],
                'projected_area': self.airborne_config['projected_area'],
                'drag_coefficient': {
                    'powered': self.airborne_config['drag_coefficient']['powered'],
                    'depowered': self.airborne_config['drag_coefficient']['depowered']
                },
                'lift_coefficient': {
                    'powered': self.airborne_config['lift_coefficient']['powered'],
                    'depowered': self.airborne_config['lift_coefficient']['depowered']
                }
            },
            'tether': {
                'length': self.tether_config['length'],
                'diameter': self.tether_config['diameter'],
                'density': self.tether_config['density'],
                'drag_coefficient': self.tether_config['drag_coefficient']
            },
            'environment': {
                'profile': self.wind_resource.get('profile_type', 'logarithmic'),
                'roughness_length': self.wind_resource.get('roughness_length', 0.07),
                'ref_height': self.wind_resource['metadata'].get('reference_height_m', 100),
            },
            'sim_settings': self.operational_constraints.get('sim_settings', {
                'force_or_speed_control': 'force',
                'time_step_RO': 0.25,
                'time_step_RI': 0.25,
                'time_step_RIRO': 0.25
            }),
            'bounds': self.operational_constraints['bounds'],
            'constraints': self.operational_constraints['constraints'],
            'opt_variables': self.operational_constraints['opt_variables'],
            'opt_settings': {
                'maxiter': self.operational_constraints.get('optimization', {}).get('max_iterations', 100),
                'iprint': 2,
                'ftol': self.operational_constraints.get('optimization', {}).get('tolerance', 1e-6),
                'eps': 1e-6
            }
        }
        
        # Initialize system properties
        if SystemProperties is not None:
            sys_props_dict = parse_system_properties_and_bounds(self.config)
            self.system_properties = SystemProperties(sys_props_dict)
            
        # Initialize environment profile
        if LogProfile is not None:
            self.environment_profile = LogProfile()
            self.environment_profile.set_reference_height(self.config['environment']['ref_height'])
            self.environment_profile.set_roughness_length(self.config['environment']['roughness_length'])
    
    def compute_power_curves(self, output_path: Path) -> None:
        """Compute power curves for all wind profiles in wind resource.
        
        :param output_path: Path where power curve YAML will be written
        :type output_path: Path
        """
        if self.wind_resource is None:
            raise ValueError("Wind resource not loaded. Call load_configuration first.")
            
        # Check if vendor functions are available
        if OptimizerCycle is None:
            raise ImportError("Vendor AWE production estimation functions not available")
        
        # Extract wind profiles and compute power curves for each cluster
        clusters = self.wind_resource['clusters']
        power_curves = {}
        
        print(f"Computing power curves for {len(clusters)} wind profile clusters...")
        
        for cluster in clusters:
            cluster_id = cluster['id']
            print(f"Processing cluster {cluster_id}...")
            
            # Extract reference wind speeds from cluster data
            ref_wind_speeds = self._extract_reference_wind_speeds(cluster)
            
            # Compute power curve for this cluster
            power_curve = self._compute_cluster_power_curve(cluster, ref_wind_speeds)
            power_curves[f'cluster_{cluster_id}'] = power_curve
        
        # Compile results
        self.power_curve_results = {
            'metadata': {
                'source': 'AWE_production_estimation',
                'clusters_processed': len(clusters),
                'wind_profile_source': self.wind_resource.get('metadata', {}).get('data_source', 'unknown'),
                'reference_height': self.config['environment']['ref_height'],
                'system_configuration': {
                    'kite_mass': self.config['kite']['mass'],
                    'kite_area': self.config['kite']['projected_area'],
                    'tether_length': self.config['tether']['length']
                }
            },
            'power_curves': power_curves
        }
        
        # Export to YAML
        with open(output_path, 'w') as f:
            yaml.dump(self.power_curve_results, f, default_flow_style=False)
        
        print(f"Power curves saved to {output_path}")
    
    def _extract_reference_wind_speeds(self, cluster: Dict[str, Any]) -> np.ndarray:
        """Extract representative wind speeds for power curve computation.
        
        :param cluster: Wind profile cluster data
        :type cluster: Dict[str, Any]
        :return: Array of reference wind speeds
        :rtype: np.ndarray
        """
        # Use the normalized wind profile to determine speed range
        u_profile = np.array(cluster['u_normalized'])
        v_profile = np.array(cluster['v_normalized'])
        magnitude = np.sqrt(u_profile**2 + v_profile**2)
        
        # Reference wind speeds based on typical AWE operating range
        # Scale by the cluster's wind magnitude characteristics
        max_magnitude = np.max(magnitude)
        base_speeds = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
        
        # Scale wind speeds based on cluster characteristics
        scaled_speeds = base_speeds * max_magnitude
        
        # Filter to reasonable range (3-25 m/s)
        valid_speeds = scaled_speeds[(scaled_speeds >= 3) & (scaled_speeds <= 25)]
        
        return valid_speeds
    
    def _compute_cluster_power_curve(self, cluster: Dict[str, Any], wind_speeds: np.ndarray) -> Dict[str, Any]:
        """Compute power curve for a specific wind profile cluster.
        
        :param cluster: Wind profile cluster data
        :type cluster: Dict[str, Any]
        :param wind_speeds: Reference wind speeds for power curve
        :type wind_speeds: np.ndarray
        :return: Power curve data for the cluster
        :rtype: Dict[str, Any]
        """
        # Setup cycle simulation settings
        cycle_sim_settings = {
            'cycle': {
                'traction_phase': TractionPhasePattern,
                'include_transition_energy': True,
            },
            'retraction': {'time_step': self.config['sim_settings'].get('time_step_RI', 0.5)},
            'transition': {'time_step': self.config['sim_settings'].get('time_step_RIRO', 0.5)},
            'traction': {'time_step': self.config['sim_settings'].get('time_step_RO', 0.25)},
        }
        
        # Convert cluster wind profile to vendor format
        environment_profile = self._convert_cluster_to_wind_profile(cluster)
        
        # Parse optimization variables and constraints
        opt_var_enabled_idx, init_vals = parse_opt_variables(self.config)
        cons_enabled_idx, cons_param_vals = parse_constraints(self.config)
        
        powers = []
        successful_speeds = []
        optimization_details = []
        
        for wind_speed in wind_speeds:
            try:
                # Set wind speed for this iteration
                environment_profile.set_reference_wind_speed(float(wind_speed))
                
                # Create optimizer for this wind speed
                optimizer = OptimizerCycle(
                    cycle_sim_settings, 
                    self.system_properties, 
                    environment_profile,
                    reduce_x=opt_var_enabled_idx,
                    reduce_ineq_cons=cons_enabled_idx,
                    parametric_cons_values=cons_param_vals,
                    force_or_speed_control=self.config['sim_settings'].get('force_or_speed_control', 'force')
                )
                
                # Set initial conditions and run optimization directly (bypass PowerCurveConstructor)
                optimizer.x0_real_scale = init_vals
                x_opt = optimizer.optimize(
                    maxiter=self.config['opt_settings'].get('maxiter', 30),
                    iprint=self.config['opt_settings'].get('iprint', 0),
                    eps=self.config['opt_settings'].get('eps', 1e-6),
                    ftol=self.config['opt_settings'].get('ftol', 1e-3)
                )
                
                # Extract power output - handle both successful and failed evaluations
                try:
                    cons, kpis = optimizer.eval_point()
                    eval_successful = True
                    sim_successful = True
                except Exception as e:
                    try:
                        # Try with relaxed errors
                        cons, kpis = optimizer.eval_point(relax_errors=True)
                        eval_successful = True
                        sim_successful = False  # Mark as not fully successful
                    except Exception as e2:
                        print(f"Failed to evaluate optimization result for {wind_speed} m/s: {e2}")
                        continue
                
                # Extract power from KPIs - handle dictionary structure
                power_output = 0.0
                if 'average_power' in kpis:
                    avg_power = kpis['average_power']
                    if isinstance(avg_power, dict):
                        # Extract cycle power (net power output)
                        power_output = float(avg_power.get('cycle', 0.0))
                    else:
                        power_output = float(avg_power)
                
                powers.append(power_output)
                successful_speeds.append(float(wind_speed))
                optimization_details.append({
                    'wind_speed': float(wind_speed),
                    'power': power_output,
                    'optimization_successful': sim_successful,
                    'evaluation_successful': eval_successful,
                    'kpis': {k: float(v) if isinstance(v, (np.ndarray, np.generic)) and np.isscalar(v) else str(v) 
                           for k, v in kpis.items() if not isinstance(v, (list, dict))},
                    'power_breakdown': avg_power if isinstance(avg_power, dict) else None
                })
                
            except Exception as e:
                print(f"Optimization failed for wind speed {wind_speed} m/s: {e}")
                optimization_details.append({
                    'wind_speed': float(wind_speed),
                    'power': 0.0,
                    'optimization_successful': False,
                    'error': str(e)
                })
        
        # Create power curve data structure
        power_curve_data = {
            'cluster_id': cluster['id'],
            'frequency': float(cluster.get('frequency', 0.0)),
            'wind_speeds': [float(ws) for ws in successful_speeds],
            'power_outputs': [float(p) for p in powers],
            'power_curve_data': [{
                'wind_speed': float(detail['wind_speed']),
                'power': float(detail['power']),
                'optimization_successful': detail['optimization_successful'],
                'error': detail.get('error')
            } for detail in optimization_details],
            'wind_profile': {
                'u_normalized': [float(u) for u in cluster['u_normalized']],
                'v_normalized': [float(v) for v in cluster['v_normalized']]
            }
        }
        
        return power_curve_data
    
    def _convert_cluster_to_wind_profile(self, cluster: Dict[str, Any]) -> NormalisedWindTable1D:
        """Convert wind cluster data to vendor wind profile format.
        
        :param cluster: Wind profile cluster from wind_resource.yml
        :type cluster: Dict[str, Any]
        :return: Vendor-compatible wind profile
        :rtype: NormalisedWindTable1D
        """
        cluster_id = cluster['id']
        
        # Check cache first
        if cluster_id in self._wind_profiles:
            return self._wind_profiles[cluster_id]
            
        # Create normalized wind profile from cluster data
        environment_profile = NormalisedWindTable1D()
        
        # Extract wind data from cluster
        if 'u_normalized' in cluster:
            # Use u-component as wind magnitude (assuming v is small for simplicity)
            u_normalized = np.array(cluster['u_normalized'])
            if 'v_normalized' in cluster:
                v_normalized = np.array(cluster['v_normalized'])
                # Calculate wind magnitude from components
                wind_magnitude_normalized = np.sqrt(u_normalized**2 + v_normalized**2)
            else:
                wind_magnitude_normalized = u_normalized
        else:
            # Fallback to normalized profile if available
            wind_magnitude_normalized = np.array(cluster.get('wind_magnitude_normalized', [1.0] * len(self.wind_resource['altitudes'])))
        
        # Get altitudes from wind resource
        altitudes = self.wind_resource['altitudes']
        
        # Ensure we have matching lengths
        if len(wind_magnitude_normalized) != len(altitudes):
            # Interpolate to match altitude levels if needed
            if len(wind_magnitude_normalized) > len(altitudes):
                wind_magnitude_normalized = wind_magnitude_normalized[:len(altitudes)]
            else:
                # Extend with last value if needed
                while len(wind_magnitude_normalized) < len(altitudes):
                    wind_magnitude_normalized = np.append(wind_magnitude_normalized, wind_magnitude_normalized[-1])
        
        # Set vendor wind profile parameters
        environment_profile.heights = list(altitudes)
        environment_profile.normalised_wind_speeds = list(wind_magnitude_normalized)
        
        # Set reference height from wind resource metadata
        ref_height = self.wind_resource.get('metadata', {}).get('reference_height_m', 100.0)
        environment_profile.set_reference_height(ref_height)
        
        # Cache the profile for reuse
        self._wind_profiles[cluster_id] = environment_profile
        
        return environment_profile