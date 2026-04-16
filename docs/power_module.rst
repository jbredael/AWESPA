Power Module
============

The power module computes **power curves** — the electrical power output of
an AWE system as a function of wind speed — for each wind profile cluster
produced by the wind module. The resulting power curve file (in awesIO YAML
format) is the direct input for the AEP pipeline.

Architecture
------------

The module is built around an Abstract Base Class that defines the interface
every power model must implement. Concrete implementations are thin wrappers
around external physics-based packages.

.. code-block:: text

   PowerEstimationModel  (abstract base class)
   ├── LuchsingerPowerModel       (wraps power-luchsinger)
   └── InertiaFreeQSMPowerModel   (wraps inertiafree-qsm)

Swapping models requires only changing the class that is instantiated — the
rest of the pipeline (configuration loading, output format, AEP calculation)
remains identical.

Base Class — ``PowerEstimationModel``
--------------------------------------

.. autoclass:: awespa.power.base.PowerEstimationModel
   :members:
   :undoc-members:
   :show-inheritance:

Every implementation must provide three methods:

``load_configuration(system_path, simulation_settings_path, ...)``
    Load all model settings from awesIO-format YAML files. Validates that
    required files exist and creates the underlying model object.

``compute_power_curves(output_path, verbose, showplot, saveplot, plot_path)``
    Calculate the full power curve for every wind profile cluster, optionally
    export the result to YAML, and optionally show / save plots.

``calculate_power_at_wind_speed(wind_speed, output_path, verbose, showplot, saveplot, plot_path)``
    Simulate a single operating point (one wind speed) and return the
    average cycle power in watts.

Output format
-------------

Both models write a YAML file in awesIO format with the following top-level
keys:

* ``reference_wind_speeds_m_s`` — wind speed vector used for evaluation
* ``power_curves`` — list of per-cluster power curve dictionaries, each
  containing the cluster ID, profile metadata, and power values [W]

This file is directly consumed by :func:`awespa.pipeline.aep.calculate_aep`.

Implementations
---------------

.. toctree::
   :maxdepth: 1

   power_luchsinger
   power_inertiafree_qsm
