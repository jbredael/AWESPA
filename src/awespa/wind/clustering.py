"""Wind profile clustering wrapper for the vendored wind-profile-clustering repository."""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base import WindProfileModel

# Add vendor path to import the clustering code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "wind-profile-clustering"
sys.path.insert(0, str(VENDOR_PATH))

# Import vendor clustering functionality
try:
    from wind_profile_clustering import cluster_normalized_wind_profiles_pca
    from export_profiles_and_probabilities_yml import export_wind_profile_shapes_and_probabilities
except ImportError:
    # Handle import errors gracefully during development
    cluster_normalized_wind_profiles_pca = None
    export_wind_profile_shapes_and_probabilities = None


class WindProfileClusteringModel(WindProfileModel):
    """Wrapper for the wind-profile-clustering repository.
    
    This wrapper adapts the vendored wind profile clustering functionality
    to the AWESPA modular architecture, handling data paths redirection
    and YAML-based configuration.
    """
    
    def __init__(self):
        """Initialize the wind profile clustering model."""
        self.config: Optional[Dict[str, Any]] = None
        self.clustering_results: Optional[Dict[str, Any]] = None
        self.n_clusters: int = 6
        self.n_pcs: int = 5
        self.ref_height: float = 100.0
        self.n_wind_speed_bins: int = 50
        self.data_source: str = 'era5'
        self.location: Dict[str, float] = {'latitude': 52.0, 'longitude': 4.0}
        self.altitude_range: tuple = (0, 500)
        self.years: tuple = (2011, 2017)
        
    def load_from_yaml(self, config_path: Path) -> None:
        """Load configuration parameters from a YAML file.
        
        :param config_path: Path to the YAML configuration file
        :type config_path: Path
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract clustering parameters
        self.n_clusters = self.config.get('n_clusters', self.n_clusters)
        self.n_pcs = self.config.get('n_pcs', self.n_pcs)
        self.ref_height = self.config.get('ref_height', self.ref_height)
        self.n_wind_speed_bins = self.config.get('n_wind_speed_bins', self.n_wind_speed_bins)
        
        # Extract data source configuration
        if 'data_source' in self.config:
            data_config = self.config['data_source']
            self.data_source = data_config.get('type', self.data_source)
            self.location = data_config.get('location', self.location)
            self.altitude_range = tuple(data_config.get('altitude_range', self.altitude_range))
            self.years = tuple(data_config.get('years', self.years))
    
    def cluster(self, data_path: Path, output_path: Path) -> None:
        """Perform wind profile clustering on the input data.
        
        :param data_path: Path to the wind data directory
        :type data_path: Path
        :param output_path: Path where output YAML file will be written
        :type output_path: Path
        """
        # Check if vendor functions are available
        if cluster_normalized_wind_profiles_pca is None:
            raise ImportError("Vendor clustering functions not available")
            
        # Load and preprocess the wind data
        wind_data = self._load_wind_data(data_path)
        
        # Perform clustering
        print(f"Performing wind profile clustering with {self.n_clusters} clusters...")
        
        # Prepare training data by combining parallel and perpendicular components
        training_data = np.hstack([
            wind_data['parallel_profiles'],
            wind_data['perpendicular_profiles']
        ])
        
        # Perform PCA-based clustering
        self.clustering_results = cluster_normalized_wind_profiles_pca(
            training_data=training_data,
            n_clusters=self.n_clusters,
            n_pcs=self.n_pcs
        )
        
        # Extract results for export
        cluster_features = self.clustering_results['clusters_feature']
        heights = wind_data['heights']
        prl = cluster_features['parallel']
        prp = cluster_features['perpendicular']
        labels_full = self.clustering_results['sample_labels']
        normalisation_wind_speeds = wind_data['reference_wind_speeds']
        wind_directions = wind_data['wind_directions']
        n_samples = len(labels_full)
        
        # Prepare metadata
        metadata = {
            'data_source': self.data_source.upper(),
            'location': self.location,
            'time_range': {
                'start_year': self.years[0],
                'end_year': self.years[1],
                'years_included': list(range(self.years[0], self.years[1] + 1))
            },
            'altitude_range_m': self.altitude_range,
            'clustering_parameters': {
                'n_pcs': self.n_pcs,
                'explained_variance': self.clustering_results['pc_explained_variance'].tolist(),
                'fit_inertia': float(self.clustering_results['fit_inertia'])
            }
        }
        
        # Export to YAML using the vendor export function
        export_wind_profile_shapes_and_probabilities(
            heights=heights,
            prl=prl,
            prp=prp,
            labels_full=labels_full,
            normalisation_wind_speeds=normalisation_wind_speeds,
            wind_directions=wind_directions,
            n_samples=n_samples,
            n_clusters=self.n_clusters,
            output_file=str(output_path),
            ref_height=self.ref_height,
            n_wind_speed_bins=self.n_wind_speed_bins,
            metadata=metadata
        )
        
        print(f"Wind profile clustering results exported to {output_path}")
    
    def _load_wind_data(self, data_path: Path) -> Dict[str, Any]:
        """Load wind data from the specified data directory.
        
        This method redirects data loading to use the project-level data directory
        instead of the vendor repository's internal data directory.
        
        :param data_path: Path to the wind data directory
        :type data_path: Path
        :return: Processed wind data dictionary
        :rtype: Dict[str, Any]
        """
        if self.data_source.lower() == 'era5':
            return self._load_era5_data(data_path)
        else:
            raise NotImplementedError(f"Data source '{self.data_source}' not yet implemented")
    
    def _load_era5_data(self, data_path: Path) -> Dict[str, Any]:
        """Load ERA5 wind data and prepare for clustering.
        
        :param data_path: Path to the ERA5 data directory
        :type data_path: Path
        :return: Processed ERA5 wind data
        :rtype: Dict[str, Any]
        """
        # Import ERA5 reader and preprocessing functions
        sys.path.insert(0, str(VENDOR_PATH / "read_data"))
        sys.path.insert(0, str(VENDOR_PATH))
        
        try:
            from era5 import read_data
            from preprocess_data import preprocess_data
        except ImportError:
            raise ImportError("ERA5 data reader or preprocessing not available from vendor repository")
        
        # Configure ERA5 data reading with redirected path
        config = {
            'data_dir': str(data_path / "wind_data" / "era5"),
            'location': self.location,
            'altitude_range': self.altitude_range,
            'years': self.years
        }
        
        print(f"Loading ERA5 data from {config['data_dir']}...")
        
        # Load the raw data using the vendor reader
        raw_data = read_data(config)
        
        # Preprocess the data to get normalized profiles
        processed_data = preprocess_data(raw_data)
        
        # Extract relevant components for clustering
        wind_data = {
            'heights': processed_data['altitude'],
            'parallel_profiles': processed_data['training_data'][:, :len(processed_data['altitude'])],  # First half is parallel
            'perpendicular_profiles': processed_data['training_data'][:, len(processed_data['altitude']):],  # Second half is perpendicular
            'reference_wind_speeds': processed_data['normalisation_value'],  # Reference wind speeds for probability calculation
            'wind_directions': processed_data['reference_vector_direction'],  # Reference wind directions in radians (1D array)
            'timestamps': processed_data.get('datetime', None)
        }
        
        print(f"Loaded {len(wind_data['parallel_profiles'])} wind profiles")
        print(f"Altitude range: {wind_data['heights'][0]:.0f} - {wind_data['heights'][-1]:.0f} m")
        
        return wind_data
    
    def get_cluster_frequencies(self) -> np.ndarray:
        """Get the frequency of each cluster as percentage of total samples.
        
        :return: Array of cluster frequencies
        :rtype: np.ndarray
        :raises ValueError: If clustering has not been performed yet
        """
        if self.clustering_results is None:
            raise ValueError("No clustering results available. Run cluster() first.")
        
        return self.clustering_results['frequency_clusters']
    
    def get_cluster_profiles(self) -> Dict[str, np.ndarray]:
        """Get the representative wind profiles for each cluster.
        
        :return: Dictionary with 'parallel' and 'perpendicular' profile arrays
        :rtype: Dict[str, np.ndarray]
        :raises ValueError: If clustering has not been performed yet
        """
        if self.clustering_results is None:
            raise ValueError("No clustering results available. Run cluster() first.")
        
        return self.clustering_results['clusters_feature']