WindProfileClusteringModel
==========================

The ``WindProfileClusteringModel`` is the standard AWESPA implementation of the
wind module. It wraps the
`wind-profile-clustering <https://github.com/jbredael/wind-profile-clustering-1>`_
package and provides three functionalities:

1. **Clustering** — identify representative wind profile shapes from
   reanalysis or measurement data using K-means on PCA-reduced profiles.
2. **Fitting** — fit a logarithmic or power-law profile to measured wind data.
3. **Prescribing** — build an analytical wind profile without measured data,
   using a prescribed profile shape and Weibull wind speed distribution.

All three functionalities write a wind resource file in awesIO YAML format,
which is the shared input for the power module and the AEP pipeline.

Supported data sources
----------------------

* ``era5`` — ERA5 reanalysis data (NetCDF files from Copernicus)
* ``fgw_lidar`` — FGW lidar measurement files
* ``dowa`` — Dutch Offshore Wind Atlas data

The wrapper automatically selects the correct data reader based on the
``data_source.type`` setting in the configuration file.

API reference
-------------

.. autoclass:: awespa.wind.clustering.WindProfileClusteringModel
   :members:
   :undoc-members:
   :show-inheritance:


Configuration file
------------------

All settings are provided in a single YAML file. The file is divided into
sections for general settings, data source, clustering, fitting, and
prescribing. An annotated example is shown below
(see ``config/example/wind_clustering_settings.yml``):

.. code-block:: yaml

   # ============================================================================
   # GENERAL SETTINGS
   # ============================================================================
   # Reference height used for wind speed normalisation across all functionalities
   ref_height: 200.0               # Reference height [m]

   # ============================================================================
   # DATA SOURCE CONFIGURATION
   # ============================================================================
   data_source:
     type: "era5"                  # 'era5' | 'fgw_lidar' | 'dowa'
     location:
       latitude: 54.13
       longitude: -9.78
     altitude_range: [0, 500]      # [m]
     years: [2011, 2011]           # inclusive

   # ============================================================================
   # CLUSTERING PARAMETERS
   # ============================================================================
   clustering:
     n_clusters: 8                 # Number of K-means clusters
     n_pcs: 5                      # Principal components retained
     n_wind_speed_bins: 50         # Bins for wind speed probability distribution

   # ============================================================================
   # FITTING PARAMETERS
   # ============================================================================
   fitting:
     profile_type: "logarithmic"   # 'logarithmic' or 'power_law'

   # ============================================================================
   # PRESCRIBING PARAMETERS
   # ============================================================================
   prescribing:
     profile_type: "logarithmic"   # 'logarithmic' or 'power_law'
     altitudes: [10, 50, 100, 150, 200, 250, 300, 400, 500]

     # Weibull wind speed distribution
     mean_wind_speed: 10.0         # Mean wind speed at ref_height [m/s]
     weibull_k: 2.0                # Weibull shape factor k [-]
     n_samples: 100000             # Synthetic samples for distribution

     # Logarithmic profile parameters
     friction_velocity: 0.4        # u* [m/s]
     roughness_length: 0.03        # z0 [m]

     # Power law profile parameters
     alpha: 0.14                   # Power law exponent [-]

     # Metadata
     name: "Prescribed Wind Profile"
     description: "Wind resource file with a prescribed analytical wind profile"


Usage examples
--------------

Clustering
~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from awespa.wind.clustering import WindProfileClusteringModel

   model = WindProfileClusteringModel()
   model.load_configuration(Path("config/example/wind_clustering_settings.yml"))

   model.cluster(
       dataPath=Path("data/wind_data/era5"),
       outputPath=Path("results/example/wind_resource.yml"),
       verbose=True,
       showplot=False,
       saveplot=True,
       plotpath=Path("results/example/plots"),
   )

Or use the ready-made script:

.. code-block:: bash

   python scripts/run_wind_clustering.py

Fitting a profile
~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = WindProfileClusteringModel()
   model.load_configuration(Path("config/example/wind_clustering_settings.yml"))

   model.fit_profile(
       dataPath=Path("data/wind_data/era5"),
       outputPath=Path("results/example/wind_resource_fit.yml"),
       verbose=True,
       showplot=False,
       saveplot=True,
       plotpath=Path("results/example/plots"),
   )

Prescribing a profile
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = WindProfileClusteringModel()
   model.load_configuration(Path("config/example/wind_clustering_settings.yml"))

   model.prescribe_profile(
       outputPath=Path("results/example/wind_resource_prescribed.yml"),
       verbose=True,
       showplot=False,
       saveplot=True,
       plotpath=Path("results/example/plots"),
   )


Output
------

All three methods write a single YAML file in awesIO format containing:

* Altitude vector
* Wind profile shapes (parallel and perpendicular components) per cluster
* Wind speed bins and probability matrix (clusters × wind speed bins)
* Cluster occurrence frequencies
* Metadata (location, time range, clustering/fitting/prescribing parameters)

This file is the input for the power module and the AEP pipeline.
