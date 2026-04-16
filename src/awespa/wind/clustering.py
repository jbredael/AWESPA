"""Wind profile clustering wrapper for the wind-profile-clustering package."""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

from .base import WindProfileModel

try:
    from awesio.validator import validate as awesio_validate  # type: ignore
except ImportError:
    awesio_validate = None

# Import vendor clustering functionality
try:
    from wind_profile_clustering.clustering import perform_clustering_analysis # type: ignore
    from wind_profile_clustering.export_profiles_and_probabilities_yml import export_wind_profile_shapes_and_probabilities # type: ignore
    from wind_profile_clustering.plotting import plot_all_results, plot_wind_profile_shapes # type: ignore
    from wind_profile_clustering.fitting_and_prescribing.fit_profile import fit_wind_profile # type: ignore
    from wind_profile_clustering.fitting_and_prescribing.prescribe_profile import prescribe_wind_profile # type: ignore
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Could not import vendor functions: {e}")
    perform_clustering_analysis = None
    export_wind_profile_shapes_and_probabilities = None
    plot_all_results = None
    plot_wind_profile_shapes = None
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
        self.refHeight: float = 100.0
        self.dataSource: str = 'era5'
        self.location: Dict[str, float] = {'latitude': 52.0, 'longitude': 4.0}
        self.altitudeRange: tuple = (0, 500)
        self.years: tuple = (2011, 2017)

        # Clustering defaults
        self.nClusters: int = 6
        self.nPcs: int = 5
        self.nWindSpeedBins: int = 50
        self.clusterName: str = 'Wind Profile Clustering'
        self.clusterDescription: str = 'Wind profile clustering results'

        # Fitting defaults
        self.fitProfileType: str = 'logarithmic'
        self.fitName: str = 'Wind Profile Fit'
        self.fitDescription: str = 'Wind profile obtained by fitting an analytical profile to data'

        # Prescribing defaults
        self.prescribeProfileType: str = 'logarithmic'
        self.prescribeAltitudeRange: tuple = (0, 500)  # Altitude range [m]
        self.prescribeMeanWindSpeed: float = 10.0
        self.prescribeWeibullK: float = 2.0
        self.prescribeNSamples: int = 100000
        self.prescribeFrictionVelocity: float = 0.4
        self.prescribeRoughnessLength: float = 0.03
        self.prescribeAlpha: float = 0.14
        self.prescribeName: str = 'Prescribed Wind Profile'
        self.prescribeDescription: str = 'Wind resource file with a prescribed analytical wind profile'
        
    def load_configuration(self, configPath: Path, validate: bool = True) -> None:
        """Load configuration parameters from a YAML file.
        
        Args:
            configPath (Path): Path to the YAML configuration file.
            validate (bool): If True, validate configuration files using
                the awesIO validator. Defaults to True.
        """
        with open(configPath, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract general settings
        self.refHeight = self.config.get('ref_height', self.refHeight)
        
        # Extract data source configuration
        if 'data_source' in self.config:
            dataConfig = self.config['data_source']
            self.dataSource = dataConfig.get('type', self.dataSource)
            self.location = dataConfig.get('location', self.location)
            self.altitudeRange = tuple(dataConfig.get('altitude_range', self.altitudeRange))
            self.years = tuple(dataConfig.get('years', self.years))

        # Extract clustering parameters
        if 'clustering' in self.config:
            clusterConfig = self.config['clustering']
            self.nClusters = clusterConfig.get('n_clusters', self.nClusters)
            self.nPcs = clusterConfig.get('n_pcs', self.nPcs)
            self.nWindSpeedBins = clusterConfig.get('n_wind_speed_bins', self.nWindSpeedBins)
            self.clusterName = clusterConfig.get('name', self.clusterName)
            self.clusterDescription = clusterConfig.get('description', self.clusterDescription)

        # Extract fitting parameters
        if 'fitting' in self.config:
            fitConfig = self.config['fitting']
            self.fitProfileType = fitConfig.get('profile_type', 'logarithmic')
            self.fitName = fitConfig.get('name', self.fitName)
            self.fitDescription = fitConfig.get('description', self.fitDescription)

        # Extract prescribing parameters
        if 'prescribing' in self.config:
            presConfig = self.config['prescribing']
            self.prescribeProfileType = presConfig.get('profile_type', 'logarithmic')
            self.prescribeAltitudeRange = tuple(presConfig.get('altitude_range', (0, 500)))
            self.prescribeMeanWindSpeed = presConfig.get('mean_wind_speed', 10.0)
            self.prescribeWeibullK = presConfig.get('weibull_k', 2.0)
            self.prescribeNSamples = presConfig.get('n_samples', 100000)
            self.prescribeFrictionVelocity = presConfig.get('friction_velocity', 0.4)
            self.prescribeRoughnessLength = presConfig.get('roughness_length', 0.03)
            self.prescribeAlpha = presConfig.get('alpha', 0.14)
            self.prescribeName = presConfig.get('name', 'Prescribed Wind Profile')
            self.prescribeDescription = presConfig.get('description', 'Wind resource file with a prescribed analytical wind profile')
    
    def cluster(
        self,
        dataPath: Path,
        outputPath: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
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
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.
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
            'name': self.clusterName,
            'description': self.clusterDescription,
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

        # Validate output file
        if validate:
            if awesio_validate is None:
                raise ImportError("awesIO validator not available")
            awesio_validate(input=outputPath)
            if verbose:
                print(f"Output validated: {outputPath}")

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
                plotpath = Path(outputPath).parent / "plots"
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
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Fit a logarithmic or power law profile to wind data and export to YAML.

        Args:
            dataPath (Path): Path to the wind data directory.
            outputPath (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots after fitting.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.

        Returns:
            Dict[str, Any]: Dictionary containing the fit results with keys
                'fitResults' and 'fitParams'.
        """
        if fit_wind_profile is None:
            raise ImportError("Vendor fitting functions not available")
        if export_wind_profile_shapes_and_probabilities is None:
            raise ImportError("Vendor export functions not available")

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

        # Filter out ground level (altitude=0) to avoid log(0) in logarithmic fitting
        altitudes = rawData['altitude']
        validAlt = altitudes > 0
        if not validAlt.all():
            nAlt = len(altitudes)
            rawData = dict(rawData)
            for key, val in rawData.items():
                if isinstance(val, np.ndarray):
                    if val.ndim == 1 and len(val) == nAlt:
                        rawData[key] = val[validAlt]
                    elif val.ndim == 2 and val.shape[1] == nAlt:
                        rawData[key] = val[:, validAlt]

        if verbose:
            print(f"Fitting {self.fitProfileType} wind profile...")

        fitResults = fit_wind_profile(rawData, profileType=self.fitProfileType, refHeight=self.refHeight)

        if verbose:
            print(f"Fit parameters: {fitResults['fitParams']}")

        # Build metadata
        dataSourceLabel = self.dataSource.upper()
        profileLabels = {
            'logarithmic': 'logarithmic  U(z) = (u*/kappa) * ln(z/z0)',
            'power_law': 'power law  U(z) = U_ref * (z/z_ref)**alpha',
        }
        profileLabel = profileLabels.get(self.fitProfileType, self.fitProfileType)
        note = (
            f"Wind speed magnitude sqrt(u_east**2 + u_north**2) was computed at each altitude "
            f"and timestep. A {profileLabel} profile was fitted to the time-averaged wind speed "
            f"profile. u_normalized contains the fitted profile normalised to 1 at "
            f"{self.refHeight:.0f} m; v_normalized is zero for all altitudes. "
            f"Fit parameters: {fitResults['fitParams']}."
        )
        metadata = {
            'name': self.fitName,
            'description': self.fitDescription,
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
            rawData['altitude'],
            fitResults['prl'],
            fitResults['prp'],
            fitResults['labelsFull'],
            fitResults['normalisationWindSpeeds'],
            fitResults['windDirections'],
            fitResults['nSamples'],
            1,
            str(outputPath),
            metadata=metadata,
            refHeight=self.refHeight,
        )

        if verbose:
            print(f"Fitted wind profile exported to {outputPath}")

        # Validate output file
        if validate:
            if awesio_validate is None:
                raise ImportError("awesIO validator not available")
            awesio_validate(input=outputPath)
            if verbose:
                print(f"Output validated: {outputPath}")

        # Plotting
        if showplot or saveplot:
            if plot_wind_profile_shapes is None:
                raise ImportError("Vendor plotting functions not available")

            altitudes = rawData['altitude']
            fig_nums_before = set(plt.get_fignums())
            plot_wind_profile_shapes(altitudes, fitResults['prl'], fitResults['prp'])
            new_figs = [plt.figure(n) for n in plt.get_fignums() if n not in fig_nums_before]

            # Wind speed distribution histogram
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(fitResults['normalisationWindSpeeds'], bins=50, density=True, alpha=0.7)
            ax_hist.set_xlabel('Wind speed at reference height [m/s]')
            ax_hist.set_ylabel('Probability density [-]')
            ax_hist.set_title('Wind Speed Distribution')
            ax_hist.grid(True)
            fig_hist.tight_layout()
            new_figs.append(fig_hist)

            if saveplot:
                plotpath = Path(outputPath).parent / "plots"
                plotpath.mkdir(parents=True, exist_ok=True)
                plot_names = ["fitted_profile_shape.pdf", "fitted_wind_speed_distribution.pdf"]
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
            'fitResults': fitResults,
            'fitParams': fitResults['fitParams'],
        }

    def prescribe_profile(
        self,
        outputPath: Path,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Build a prescribed analytical wind profile and export to YAML.

        No measured wind data is required. The wind speed probability
        distribution is a Weibull distribution defined by a mean wind speed
        and shape factor k, with a single omnidirectional wind-direction bin.

        When parameters are None they fall back to values loaded from the
        configuration file via ``load_configuration``.

        Args:
            outputPath (Path): Path where output YAML file will be written.
            verbose (bool): If True, print progress and diagnostic information.
                Defaults to False.
            showplot (bool): If True, display plots after prescribing.
                Defaults to False.
            saveplot (bool): If True, save plots to disk. Defaults to False.
            validate (bool): If True, validate the output YAML file using
                the awesIO validator. Defaults to True.

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
            print(f"Building prescribed {self.prescribeProfileType} profile...")
        altitudes = np.linspace(self.prescribeAltitudeRange[0], self.prescribeAltitudeRange[1], 100)

        if self.prescribeProfileType == 'logarithmic':
            result = prescribe_wind_profile(
                heights=altitudes,
                profileType='logarithmic',
                refHeight=self.refHeight,
                meanWindSpeed=self.prescribeMeanWindSpeed,
                weibullK=self.prescribeWeibullK,
                nSamples=self.prescribeNSamples,
                frictionVelocity=self.prescribeFrictionVelocity,
                roughnessLength=self.prescribeRoughnessLength,
            )
            paramStr = (
                f"friction velocity u* = {self.prescribeFrictionVelocity} m/s, "
                f"roughness length z0 = {self.prescribeRoughnessLength} m"
            )
            profileFormula = "U(z) = (u*/kappa) * ln(z/z0)"
        elif self.prescribeProfileType == 'power_law':
            result = prescribe_wind_profile(
                heights=altitudes,
                profileType='power_law',
                refHeight=self.refHeight,
                meanWindSpeed=self.prescribeMeanWindSpeed,
                weibullK=self.prescribeWeibullK,
                nSamples=self.prescribeNSamples,
                alpha=self.prescribeAlpha,
            )
            paramStr = f"exponent alpha = {self.prescribeAlpha}"
            profileFormula = "U(z) = U_ref * (z/z_ref)**alpha"
        else:
            raise ValueError(
                f"Unknown profile type: {self.prescribeProfileType}. "
                "Choose 'logarithmic' or 'power_law'."
            )

        if verbose:
            print(f"Profile parameters: {result['profileParams']}")
            print(f"Weibull parameters: {result['weibullParams']}")

        note = (
            f"Profile shape prescribed analytically using a {self.prescribeProfileType} profile "
            f"({profileFormula}) with {paramStr}. "
            f"No measured wind data was used. "
            f"The wind speed probability distribution is a Weibull distribution with "
            f"mean wind speed {self.prescribeMeanWindSpeed} m/s and shape factor k = {self.prescribeWeibullK} "
            f"(Weibull scale parameter lambda = {result['weibullParams']['lambda']:.4f} m/s). "
            f"u_normalized contains the prescribed profile normalised to 1 at "
            f"{self.refHeight:.0f} m; v_normalized is zero for all altitudes. "
            f"The probability matrix has a single wind-direction bin (omnidirectional)."
        )
        metadata = {
            'name': self.prescribeName,
            'description': self.prescribeDescription,
            'note': note,
            'data_source': 'prescribed_analytical',
            'altitude_range': [float(self.prescribeAltitudeRange[0]), float(self.prescribeAltitudeRange[1])],
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
            refHeight=self.refHeight,
            windDirectionBinWidth=360,
        )

        if verbose:
            print(f"Prescribed wind profile exported to {outputPath}")

        # Validate output file
        if validate:
            if awesio_validate is None:
                raise ImportError("awesIO validator not available")
            awesio_validate(input=outputPath)
            if verbose:
                print(f"Output validated: {outputPath}")

        # Plotting
        if showplot or saveplot:
            if plot_wind_profile_shapes is None:
                raise ImportError("Vendor plotting functions not available")

            fig_nums_before = set(plt.get_fignums())
            plot_wind_profile_shapes(altitudes, result['prl'], result['prp'])
            new_figs = [plt.figure(n) for n in plt.get_fignums() if n not in fig_nums_before]

            # Wind speed distribution histogram
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(result['normalisationWindSpeeds'], bins=50, density=True, alpha=0.7)
            ax_hist.set_xlabel('Wind speed at reference height [m/s]')
            ax_hist.set_ylabel('Probability density [-]')
            ax_hist.set_title('Wind Speed Distribution (Weibull)')
            ax_hist.grid(True)
            fig_hist.tight_layout()
            new_figs.append(fig_hist)

            if saveplot:
                plotpath = Path(outputPath).parent / "plots"
                plotpath.mkdir(parents=True, exist_ok=True)
                plot_names = ["prescribed_profile_shape.pdf", "prescribed_wind_speed_distribution.pdf"]
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
            'profileParams': result['profileParams'],
            'weibullParams': result['weibullParams'],
            'result': result,
        }
    
