"""Annual Energy Production (AEP) calculation pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Type

from ..wind.base import WindProfileModel
from ..optimisation.base import Optimiser
from ..power.base import PowerEstimationModel
from ..kite.base import KiteModel
from ..control.base import ControlModel
from ..groundstation.base import GroundStationModel


class AEPCalculator:
    """Orchestrates the AWERA toolchain to compute Annual Energy Production.
    
    This class coordinates the execution of wind clustering, optimisation,
    and power estimation models to compute AEP for AWE systems.
    """
    
    def __init__(self, config_path: Path):
        """Initialize the AEP calculator with configuration.
        
        :param config_path: Path to the main configuration YAML file
        :type config_path: Path
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Model instances will be created based on config
        self.wind_model: WindProfileModel = None
        self.optimiser: Optimiser = None
        self.power_model: PowerEstimationModel = None
        self.kite_model: KiteModel = None
        self.control_model: ControlModel = None
        self.groundstation_model: GroundStationModel = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        :return: Configuration dictionary
        :rtype: Dict[str, Any]
        """
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_models(self, 
                     wind_model_class: Type[WindProfileModel],
                     optimiser_class: Type[Optimiser],
                     power_model_class: Type[PowerEstimationModel],
                     kite_model_class: Type[KiteModel] = None,
                     control_model_class: Type[ControlModel] = None,
                     groundstation_model_class: Type[GroundStationModel] = None) -> None:
        """Setup model instances based on provided classes.
        
        :param wind_model_class: Wind profile model class
        :type wind_model_class: Type[WindProfileModel]
        :param optimiser_class: Optimiser class
        :type optimiser_class: Type[Optimiser]
        :param power_model_class: Power estimation model class
        :type power_model_class: Type[PowerEstimationModel]
        :param kite_model_class: Kite model class, defaults to None
        :type kite_model_class: Type[KiteModel], optional
        :param control_model_class: Control model class, defaults to None
        :type control_model_class: Type[ControlModel], optional
        :param groundstation_model_class: Ground station model class, defaults to None
        :type groundstation_model_class: Type[GroundStationModel], optional
        """
        # Initialize required models
        self.wind_model = wind_model_class()
        self.optimiser = optimiser_class()
        self.power_model = power_model_class()
        
        # Initialize optional models if provided
        if kite_model_class:
            self.kite_model = kite_model_class()
        if control_model_class:
            self.control_model = control_model_class()
        if groundstation_model_class:
            self.groundstation_model = groundstation_model_class()
    
    def calculate_aep(self, data_path: Path, results_path: Path) -> Dict[str, Any]:
        """Execute the complete AEP calculation pipeline.
        
        :param data_path: Path to the input data directory
        :type data_path: Path
        :param results_path: Path to the results directory
        :type results_path: Path
        :return: AEP calculation results
        :rtype: Dict[str, Any]
        :raises ValueError: If required models are not initialized
        """
        if not all([self.wind_model, self.optimiser, self.power_model]):
            raise ValueError("Required models (wind, optimiser, power) must be initialized")
        
        # Define intermediate file paths
        wind_profiles_path = results_path / "wind_profiles.yml"
        optimal_controls_path = results_path / "optimal_controls.yml"
        power_curve_path = results_path / "power_curve.yml"
        
        # Ensure results directory exists
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Wind profile clustering
        print("Step 1: Performing wind profile clustering...")
        if 'wind' in self.config:
            wind_config_path = Path(self.config['wind']['config'])
            self.wind_model.load_from_yaml(wind_config_path)
        
        self.wind_model.cluster(data_path, wind_profiles_path)
        
        # Step 2: Optimisation
        print("Step 2: Running optimisation...")
        if 'optimisation' in self.config:
            opt_config_path = Path(self.config['optimisation']['config'])
            self.optimiser.load_from_yaml(opt_config_path)
        
        self.optimiser.optimise(wind_profiles_path, optimal_controls_path)
        
        # Step 3: Power computation
        print("Step 3: Computing power curve...")
        if 'power' in self.config:
            power_config_path = Path(self.config['power']['config'])
            self.power_model.load_from_yaml(power_config_path)
        
        self.power_model.compute_power(wind_profiles_path, optimal_controls_path, power_curve_path)
        
        # Step 4: Calculate AEP from power curve
        print("Step 4: Calculating AEP...")
        aep_results = self._calculate_aep_from_power_curve(power_curve_path, wind_profiles_path)
        
        # Save final results
        aep_results_path = results_path / "aep_results.yml"
        with open(aep_results_path, 'w') as f:
            yaml.dump(aep_results, f)
        
        print(f"AEP calculation complete. Results saved to {aep_results_path}")
        return aep_results
    
    def _calculate_aep_from_power_curve(self, power_curve_path: Path, wind_profiles_path: Path) -> Dict[str, Any]:
        """Calculate AEP from power curve and wind profile frequencies.
        
        :param power_curve_path: Path to power curve YAML file
        :type power_curve_path: Path
        :param wind_profiles_path: Path to wind profiles YAML file
        :type wind_profiles_path: Path
        :return: AEP calculation results
        :rtype: Dict[str, Any]
        """
        # Load power curve data
        with open(power_curve_path, 'r') as f:
            power_data = yaml.safe_load(f)
        
        # Load wind profile frequencies
        with open(wind_profiles_path, 'r') as f:
            wind_data = yaml.safe_load(f)
        
        # Calculate AEP (simplified implementation)
        # This would need to be expanded based on actual power curve and frequency data structure
        total_aep = 0.0
        hours_per_year = 8760
        
        # Placeholder calculation - to be implemented based on actual data structure
        if 'clusters' in wind_data and 'power_curve' in power_data:
            for cluster in wind_data['clusters']:
                frequency = cluster.get('frequency', 0)
                # Match power output for this cluster
                # This is a simplified example
                power_output = power_data['power_curve'].get('mean_power', 0)
                total_aep += power_output * frequency * hours_per_year
        
        return {
            'aep_mwh': total_aep / 1000,  # Convert to MWh
            'aep_gwh': total_aep / 1000000,  # Convert to GWh
            'calculation_timestamp': str(Path().resolve()),
            'input_files': {
                'power_curve': str(power_curve_path),
                'wind_profiles': str(wind_profiles_path)
            }
        }
