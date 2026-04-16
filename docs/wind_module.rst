Wind Module
===========

The wind module is responsible for processing raw wind data and producing a
**wind resource file** — a YAML file (in awesIO format) that contains the
representative wind profile shapes, their associated wind speed probability
distributions, and cluster occurrence frequencies. This file is the shared
input for all power-module models and the AEP pipeline.

Architecture
------------

The module is built around an Abstract Base Class (ABC) that defines the
interface every wind model must implement. Concrete implementations are
wrappers around external packages, adapting them to the AWESPA interface.

.. code-block:: text

   WindProfileModel  (abstract base class)
   └── WindProfileClusteringModel  (ERA5 / lidar / DOWA wrapper)

Base Class — ``WindProfileModel``
---------------------------------

.. autoclass:: awespa.wind.base.WindProfileModel
   :members:
   :undoc-members:
   :show-inheritance:

The base class enforces the following interface on every implementation:

``load_configuration(config_path)``
    Load all model settings from a YAML configuration file so that the
    analysis is fully reproducible from a single file.

``cluster(data_path, output_path, verbose, showplot, saveplot, plotpath)``
    Execute the clustering and write the wind resource YAML to
    ``output_path``.

Implementations
---------------

.. toctree::
   :maxdepth: 1

   wind_clustering
