"""Wind profile clustering wrapper for the vendored wind-profile-clustering repository."""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base import WindProfileModel

# Add vendor path to import the clustering code
VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "wind-profile-clustering" / "src"
sys.path.insert(0, str(VENDOR_PATH))

# Import vendor clustering functionality
try:
    from wind_profile_clustering.clustering import perform_clustering_analysis
    from wind_profile_clustering.export_profiles_and_probabilities_yml import export_wind_profile_shapes_and_probabilities
    from wind_profile_clustering.read_data.era5 import read_data as read_era5_data
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Could not import vendor functions: {e}")
    perform_clustering_analysis = None
    export_wind_profile_shapes_and_probabilities = None
    read_era5_data = None


class WindProfileClusteringModel(WindProfileModel):
    """Wrapper for the wind-profile-clustering repository.
    
    This wrapper adapts the vendored wind profile clustering functionality
    to the AWESPA modular architecture, handling data paths redirection
    and YAML-based configuration.
    """
    
    def __init__(self):
        """Initialize the wind profile clustering model."""
        self.config: Optional[Dict[str, Any]] = None
        self.clusteringResults: Optional[Dict[str, Any]] = None
        self.nClusters: int = 6
        self.nPcs: int = 5
        self.refHeight: float = 100.0
        self.nWindSpeedBins: int = 50
        self.dataSource: str = 'era5'
        self.location: Dict[str, float] = {'latitude': 52.0, 'longitude': 4.0}
        self.altitudeRange: tuple = (0, 500)
        self.years: tuple = (2011, 2017)
        
    def load_from_yaml(self, configPath: Path) -> None:
        """Load configuration parameters from a YAML file.
        
        Args:
            configPath (Path): Path to the YAML configuration file.
        """
        with open(configPath, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract clustering parameters
        self.nClusters = self.config.get('n_clusters', self.nClusters)
        self.nPcs = self.config.get('n_pcs', self.nPcs)
        self.refHeight = self.config.get('ref_height', self.refHeight)
        self.nWindSpeedBins = self.config.get('n_wind_speed_bins', self.nWindSpeedBins)
        
        # Extract data source configuration
        if 'data_source' in self.config:
            dataConfig = self.config['data_source']
            self.dataSource = dataConfig.get('type', self.dataSource)
            self.location = dataConfig.get('location', self.location)
            self.altitudeRange = tuple(dataConfig.get('altitude_range', self.altitudeRange))
            self.years = tuple(dataConfig.get('years', self.years))
    
    def cluster(self, dataPath: Path, outputPath: Path) -> None:
        """Perform wind profile clustering on the input data.
        
        Args:
            dataPath (Path): Path to the wind data directory.
            outputPath (Path): Path where output YAML file will be written.
        """
        # Check if vendor functions are available
        if perform_clustering_analysis is None:
            raise ImportError("Vendor clustering functions not available")
            
        # Load raw wind data
        if self.dataSource.lower() == 'era5':
            config = {
                'data_dir': str(dataPath / "wind_data" / "era5"),
                'location': self.location,
                'altitude_range': self.altitudeRange,
                'years': self.years
            }
            print(f"Loading ERA5 data from {config['data_dir']}...")
            rawData = read_era5_data(config)
        else:
            raise NotImplementedError(f"Data source '{self.dataSource}' not yet implemented")
        
        # Perform clustering analysis using vendor function
        print(f"Performing wind profile clustering with {self.nClusters} clusters...")
        results = perform_clustering_analysis(rawData, self.nClusters)
        
        # Extract results
        processedDataFull = results['processedDataFull']
        self.clusteringResults = results['clusteringResults']
        labelsFull = results['labelsFull']
        frequencyClusters = results['frequencyClusters']
        
        # Extract cluster features for export
        clusterFeatures = self.clusteringResults['clusters_feature']
        heights = processedDataFull['altitude']
        prl = clusterFeatures['parallel']
        prp = clusterFeatures['perpendicular']
        normalisationWindSpeeds = processedDataFull['normalisation_value']
        windDirections = processedDataFull['reference_vector_direction']
        nSamples = processedDataFull['n_samples']
        
        # Prepare metadata
        metadata = {
            'data_source': self.dataSource.upper(),
            'location': self.location,
            'time_range': {
                'start_year': self.years[0],
                'end_year': self.years[1],
                'years_included': list(range(self.years[0], self.years[1] + 1))
            },
            'altitude_range_m': list(self.altitudeRange),  # Convert tuple to list
            'clustering_parameters': {
                'n_pcs': self.clusteringResults['pca'].n_components_,
                'explained_variance': self.clusteringResults['pc_explained_variance'].tolist(),
                'fit_inertia': float(self.clusteringResults['fit_inertia'])
            }
        }
        
        # Export to YAML using the vendor export function
        export_wind_profile_shapes_and_probabilities(
            heights=heights,
            prl=prl,
            prp=prp,
            labelsFull=labelsFull,
            normalisationWindSpeeds=normalisationWindSpeeds,
            windDirections=windDirections,
            nSamples=nSamples,
            nClusters=self.nClusters,
            outputFile=str(outputPath),
            refHeight=self.refHeight,
            nWindSpeedBins=self.nWindSpeedBins,
            metadata=metadata
        )
        
        print(f"Wind profile clustering results exported to {outputPath}")
    
    def get_cluster_frequencies(self) -> np.ndarray:
        """Get the frequency of each cluster as percentage of total samples.
        
        Returns:
            np.ndarray: Array of cluster frequencies.
            
        Raises:
            ValueError: If clustering has not been performed yet.
        """
        if self.clusteringResults is None:
            raise ValueError("No clustering results available. Run cluster() first.")
        
        return self.clusteringResults['frequency_clusters']
    
    def get_cluster_profiles(self) -> Dict[str, np.ndarray]:
        """Get the representative wind profiles for each cluster.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'parallel' and 'perpendicular' profile arrays.
            
        Raises:
            ValueError: If clustering has not been performed yet.
        """
        if self.clusteringResults is None:
            raise ValueError("No clustering results available. Run cluster() first.")
        
        return self.clusteringResults['clusters_feature']