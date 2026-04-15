"""Wind profile clustering wrapper for the wind-profile-clustering package."""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

from .base import WindProfileModel

# Import vendor clustering functionality
try:
    from wind_profile_clustering.clustering import perform_clustering_analysis # type: ignore
    from wind_profile_clustering.export_profiles_and_probabilities_yml import export_wind_profile_shapes_and_probabilities # type: ignore
    from wind_profile_clustering.plotting import plot_all_results # type: ignore
    from wind_profile_clustering.fitting_and_prescribing.fit_profile import fit_wind_profile # type: ignore
    from wind_profile_clustering.fitting_and_prescribing.prescribe_profile import prescribe_wind_profile # type: ignore
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Could not import vendor functions: {e}")
    perform_clustering_analysis = None
    export_wind_profile_shapes_and_probabilities = None
    plot_all_results = None
    fit_wind_profile = None
    prescribe_wind_profile = None


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
    
    def cluster(
        self,
        dataPath: Path,
        outputPath: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        plotpath: Optional[Path] = None,
    ) -> None:
        """Perform wind profile clustering on the input data.

        Args:
            dataPath (Path): Path to the wind data directory.
            outputPath (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots after clustering.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            plotpath (Optional[Path]): Directory where plots are saved.
                Required if saveplot is True. Defaults to None.
        """
        # Check if vendor functions are available
        if perform_clustering_analysis is None:
            raise ImportError("Vendor clustering functions not available")

        # Load raw wind data

        config = {
            'data_dir': str(dataPath),
            'location': self.location,
            'altitude_range': self.altitudeRange,
            'years': self.years
        }
        if self.dataSource.lower() == 'era5':
            from wind_profile_clustering.read_data.era5 import read_data # type: ignore
        elif self.dataSource.lower() == 'fgw_lidar':
            from wind_profile_clustering.read_data.fgw_lidar import read_data # type: ignore
        elif self.dataSource.lower() == 'dowa':
            from wind_profile_clustering.read_data.dowa import read_data # type: ignore
        rawData = read_data(config)
        
        # Perform clustering analysis using vendor function
        if verbose:
            print(f"Performing wind profile clustering with {self.nClusters} clusters...")
        results = perform_clustering_analysis(rawData, self.nClusters, ref_height=self.refHeight)
        
        # Extract results
        processedDataFull = results['processedDataFull']
        processedData = results['processedData']
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
            'name': f"Wind Profile Clustering - {self.nClusters} Clusters",
            'description': f"Wind profile clustering results with {self.nClusters} clusters.",
            'note': "N/A",
            'data_source': self.dataSource.upper(),
            'location': self.location,
            'time_range': {
                'start_date': str(rawData['datetime'][0].astype('datetime64[D]')),
                'end_date': str(rawData['datetime'][-1].astype('datetime64[D]')),
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

        if verbose:
            print(f"Wind profile clustering results exported to {outputPath}")

        # Plotting
        if showplot or saveplot:
            if plot_all_results is None:
                raise ImportError("Vendor plotting functions not available")

            fig_nums_before = set(plt.get_fignums())
            plot_all_results(
                processed_data=processedData,
                res=self.clusteringResults,
                processed_data_full=processedDataFull,
                labels_full=labelsFull,
                frequency_clusters_full=frequencyClusters,
                n_clusters=self.nClusters,
                savePlots=False,
            )
            new_figs = [plt.figure(n) for n in plt.get_fignums() if n not in fig_nums_before]

            if saveplot:
                if plotpath is None:
                    raise ValueError("plotpath must be provided when saveplot is True")
                plotpath = Path(plotpath)
                plotpath.mkdir(parents=True, exist_ok=True)
                plot_names = [
                    "wind_profile_shapes.pdf",
                    "cluster_patterns.pdf",
                    "pc_projection.pdf",
                    "cluster_frequencies_comparison.pdf",
                ]
                for fig, name in zip(new_figs, plot_names):
                    fig_file = plotpath / name
                    fig.savefig(fig_file, bbox_inches='tight')
                    if verbose:
                        print(f"Saved: {fig_file}")

            if showplot:
                plt.show()
            else:
                for fig in new_figs:
                    plt.close(fig)
        return {
            'processedDataFull': processedDataFull,
            'processedData': processedData,
            'clusteringResults': self.clusteringResults,
            'labelsFull': labelsFull,
            'frequencyClusters': frequencyClusters
        }

    def fit_profile(
        self,
        dataPath: Path,
        outputPath: Path,
        profileType: str = 'logarithmic',
        refHeight: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Fit a logarithmic or power law profile to wind data and export to YAML.

        Args:
            dataPath (Path): Path to the wind data directory.
            outputPath (Path): Path where output YAML file will be written.
            profileType (str): Profile type to fit. Either 'logarithmic' or
                'power_law'. Defaults to 'logarithmic'.
            refHeight (Optional[float]): Reference height for normalisation in
                metres. Defaults to the instance attribute refHeight if None.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing the fit results with keys
                'fitResults' and 'fitParams'.
        """
        if fit_wind_profile is None:
            raise ImportError("Vendor fitting functions not available")
        if export_wind_profile_shapes_and_probabilities is None:
            raise ImportError("Vendor export functions not available")

        usedRefHeight = refHeight if refHeight is not None else self.refHeight

        # Load raw wind data
        config = {
            'data_dir': str(dataPath),
            'location': self.location,
            'altitude_range': self.altitudeRange,
            'years': self.years
        }
        if self.dataSource.lower() == 'era5':
            from wind_profile_clustering.read_data.era5 import read_data # type: ignore
        elif self.dataSource.lower() == 'fgw_lidar':
            from wind_profile_clustering.read_data.fgw_lidar import read_data # type: ignore
        elif self.dataSource.lower() == 'dowa':
            from wind_profile_clustering.read_data.dowa import read_data # type: ignore
        else:
            raise ValueError(
                f"Unknown data source: {self.dataSource}. "
                "Choose from 'era5', 'fgw_lidar', or 'dowa'."
            )
        rawData = read_data(config)

        if verbose:
            print(f"Fitting {profileType} wind profile...")

        fitResults = fit_wind_profile(rawData, profileType=profileType, refHeight=usedRefHeight)

        if verbose:
            print(f"Fit parameters: {fitResults['fitParams']}")

        # Build metadata
        dataSourceLabel = self.dataSource.upper()
        profileLabels = {
            'logarithmic': 'logarithmic  U(z) = (u*/kappa) * ln(z/z0)',
            'power_law': 'power law  U(z) = U_ref * (z/z_ref)**alpha',
        }
        profileLabel = profileLabels.get(profileType, profileType)
        note = (
            f"Wind speed magnitude sqrt(u_east**2 + u_north**2) was computed at each altitude "
            f"and timestep. A {profileLabel} profile was fitted to the time-averaged wind speed "
            f"profile. u_normalized contains the fitted profile normalised to 1 at "
            f"{usedRefHeight:.0f} m; v_normalized is zero for all altitudes. "
            f"Fit parameters: {fitResults['fitParams']}."
        )
        nameLabel = profileType.replace('_', ' ').title()
        metadata = {
            'name': f'{dataSourceLabel} Wind Profile {nameLabel} Fit',
            'description': (
                f'Wind profile obtained by fitting a {profileType} profile '
                f'to {dataSourceLabel} data'
            ),
            'note': note,
            'data_source': dataSourceLabel,
            'location': self.location,
            'time_range': {
                'start_date': str(rawData['datetime'][0].astype('datetime64[D]')),
                'end_date': str(rawData['datetime'][-1].astype('datetime64[D]')),
            },
            'altitude_range': list(self.altitudeRange),
        }

        export_wind_profile_shapes_and_probabilities(
            fitResults['altitude'],
            fitResults['prl'],
            fitResults['prp'],
            fitResults['labelsFull'],
            fitResults['normalisationWindSpeeds'],
            fitResults['windDirections'],
            fitResults['nSamples'],
            1,
            str(outputPath),
            metadata=metadata,
            refHeight=usedRefHeight,
        )

        if verbose:
            print(f"Fitted wind profile exported to {outputPath}")

        return {
            'fitResults': fitResults,
            'fitParams': fitResults['fitParams'],
        }

    def prescribe_profile(
        self,
        outputPath: Path,
        altitudes: np.ndarray,
        profileType: str = 'logarithmic',
        refHeight: float = 200.0,
        meanWindSpeed: float = 10.0,
        weibullK: float = 2.0,
        nSamples: int = 100000,
        frictionVelocity: float = 0.4,
        roughnessLength: float = 0.03,
        alpha: float = 0.14,
        name: str = 'Prescribed Wind Profile',
        description: str = 'Wind resource file with a prescribed analytical wind profile',
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Build a prescribed analytical wind profile and export to YAML.

        No measured wind data is required. The wind speed probability
        distribution is a Weibull distribution defined by a mean wind speed
        and shape factor k, with a single omnidirectional wind-direction bin.

        Args:
            outputPath (Path): Path where output YAML file will be written.
            altitudes (np.ndarray): Array of altitudes in metres to evaluate
                the profile at.
            profileType (str): Profile type. Either 'logarithmic' or
                'power_law'. Defaults to 'logarithmic'.
            refHeight (float): Reference height for profile normalisation in
                metres. Defaults to 200.0.
            meanWindSpeed (float): Mean wind speed at reference height in m/s
                for the Weibull distribution. Defaults to 10.0.
            weibullK (float): Weibull shape factor k. Defaults to 2.0.
            nSamples (int): Number of synthetic samples for the wind speed
                distribution. Defaults to 100000.
            frictionVelocity (float): Friction velocity u* in m/s. Used when
                profileType is 'logarithmic'. Defaults to 0.4.
            roughnessLength (float): Roughness length z0 in metres. Used when
                profileType is 'logarithmic'. Defaults to 0.03.
            alpha (float): Power law exponent. Used when profileType is
                'power_law'. Defaults to 0.14.
            name (str): Name field for the output metadata. Defaults to
                'Prescribed Wind Profile'.
            description (str): Description field for the output metadata.
                Defaults to 'Wind resource file with a prescribed analytical
                wind profile'.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing 'profileParams',
                'weibullParams', and the full result dict from
                prescribe_wind_profile.
        """
        if prescribe_wind_profile is None:
            raise ImportError("Vendor prescribing functions not available")
        if export_wind_profile_shapes_and_probabilities is None:
            raise ImportError("Vendor export functions not available")

        if verbose:
            print(f"Building prescribed {profileType} profile...")

        if profileType == 'logarithmic':
            result = prescribe_wind_profile(
                altitudes,
                profileType='logarithmic',
                refHeight=refHeight,
                meanWindSpeed=meanWindSpeed,
                weibullK=weibullK,
                nSamples=nSamples,
                frictionVelocity=frictionVelocity,
                roughnessLength=roughnessLength,
            )
            paramStr = (
                f"friction velocity u* = {frictionVelocity} m/s, "
                f"roughness length z0 = {roughnessLength} m"
            )
            profileFormula = "U(z) = (u*/kappa) * ln(z/z0)"
        elif profileType == 'power_law':
            result = prescribe_wind_profile(
                altitudes,
                profileType='power_law',
                refHeight=refHeight,
                meanWindSpeed=meanWindSpeed,
                weibullK=weibullK,
                nSamples=nSamples,
                alpha=alpha,
            )
            paramStr = f"exponent alpha = {alpha}"
            profileFormula = "U(z) = U_ref * (z/z_ref)**alpha"
        else:
            raise ValueError(
                f"Unknown profile type: {profileType}. "
                "Choose 'logarithmic' or 'power_law'."
            )

        if verbose:
            print(f"Profile parameters: {result['profileParams']}")
            print(f"Weibull parameters: {result['weibullParams']}")

        note = (
            f"Profile shape prescribed analytically using a {profileType} profile "
            f"({profileFormula}) with {paramStr}. "
            f"No measured wind data was used. "
            f"The wind speed probability distribution is a Weibull distribution with "
            f"mean wind speed {meanWindSpeed} m/s and shape factor k = {weibullK} "
            f"(Weibull scale parameter lambda = {result['weibullParams']['lambda']:.4f} m/s). "
            f"u_normalized contains the prescribed profile normalised to 1 at "
            f"{refHeight:.0f} m; v_normalized is zero for all altitudes. "
            f"The probability matrix has a single wind-direction bin (omnidirectional)."
        )
        metadata = {
            'name': name,
            'description': description,
            'note': note,
            'data_source': 'prescribed_analytical',
            'altitude_range': [float(altitudes.min()), float(altitudes.max())],
        }

        export_wind_profile_shapes_and_probabilities(
            altitudes,
            result['prl'],
            result['prp'],
            result['labelsFull'],
            result['normalisationWindSpeeds'],
            result['windDirections'],
            result['nSamples'],
            1,
            str(outputPath),
            metadata=metadata,
            refHeight=refHeight,
            windDirectionBinWidth=360,
        )

        if verbose:
            print(f"Prescribed wind profile exported to {outputPath}")

        return {
            'profileParams': result['profileParams'],
            'weibullParams': result['weibullParams'],
            'result': result,
        }
    
