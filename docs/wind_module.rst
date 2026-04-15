Wind Module
===========

The wind module is responsible for processing raw wind data and producing a
**wind resource file** — a YAML file (in awesIO format) that contains the
representative wind profile shapes, their associated wind speed probability
distributions, and cluster occurrence frequencies. This file is the shared
input for all power module models and the AEP pipeline.

Architecture
------------

The module is built around an Abstract Base Class (ABC) that defines the
interface every wind model must implement. Concrete implementations are
wrappers around external packages, adapting them to the AWESPA interface.

.. code-block:: text

   WindProfileModel  (abstract base class)
   └── WindProfileClusteringModel  (ERA5 / lidar / DOWA wrapper)

Base Class
----------

.. autoclass:: awespa.wind.base.WindProfileModel
   :members:
   :undoc-members:
   :show-inheritance:

The base class enforces two methods on every implementation:

``load_from_yaml(config_path)``
    Load all model settings from a YAML configuration file so that the
    analysis is fully reproducible from a single file.

``cluster(dataPath, outputPath, ...)``
    Execute the clustering and write the wind resource YAML to
    ``outputPath``.

WindProfileClusteringModel
--------------------------

The ``WindProfileClusteringModel`` is the standard AWESPA implementation.
It wraps the
`wind-profile-clustering <https://github.com/jbredael/wind-profile-clustering-1>`_
package and supports three data sources:

* ``era5`` — ERA5 reanalysis data (NetCDF files downloaded from Copernicus)
* ``fgw_lidar`` — FGW lidar measurement files
* ``dowa`` — Dutch Offshore Wind Atlas data

The wrapper handles:

* Reading the YAML configuration and mapping settings to the underlying library.
* Routing the correct data-reader for the selected ``data_source.type``.
* Invoking the clustering algorithm (K-means on PCA-reduced wind profiles).
* Collecting and re-packaging results in awesIO format.
* Optionally plotting and saving cluster visualisations.

.. autoclass:: awespa.wind.clustering.WindProfileClusteringModel
   :members:
   :undoc-members:
   :show-inheritance:

Configuration file
~~~~~~~~~~~~~~~~~~

All settings are provided in a YAML file. An annotated example is shown
below (see ``config/example/wind_clustering_settings.yml``):

.. code-block:: yaml

   # Number of K-means clusters to identify
   n_clusters: 8

   # Number of principal components retained before clustering
   n_pcs: 5

   # Reference height [m] used for wind speed normalisation
   ref_height: 200.0

   # Number of bins for the wind speed probability distribution
   n_wind_speed_bins: 50

   data_source:
     type: "era5"              # 'era5' | 'fgw_lidar' | 'dowa'
     location:
       latitude: 54.13
       longitude: -9.78
     altitude_range: [0, 500]  # [m]
     years: [2011, 2017]       # inclusive

Usage example
~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from awespa.wind.clustering import WindProfileClusteringModel

   model = WindProfileClusteringModel()
   model.load_from_yaml(Path("config/example/wind_clustering_settings.yml"))

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

Output
~~~~~~

The ``cluster`` method writes a single YAML file (``wind_resource.yml``)
containing:

* Altitude vector
* Wind profile shapes (parallel and perpendicular components) per cluster
* Wind speed bins and probability matrix (clusters × wind speed bins)
* Cluster occurrence frequencies
* Metadata (location, time range, clustering parameters)

This file is the only output required by the power module and the AEP pipeline.
