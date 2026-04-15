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

Base Class
----------

.. autoclass:: awespa.power.base.PowerEstimationModel
   :members:
   :undoc-members:
   :show-inheritance:

Every implementation must provide three methods:

``load_configuration(system_path, simulation_settings_path, ...)``
    Load all model settings from awesIO-format YAML files. Validates that
    required files exist and creates the underlying model object.

``compute_power_curves(output_path, ...)``
    Calculate the full power curve for every wind profile cluster, optionally
    export the result to YAML, and optionally show / save plots.

``calculate_power_at_wind_speed(wind_speed, ...)``
    Simulate a single operating point (one wind speed) and return the
    average cycle power in watts.

LuchsingerPowerModel
--------------------

The ``LuchsingerPowerModel`` wraps the
`power-luchsinger <https://github.com/jbredael/LuchsingerPowerModel>`_
package. The underlying model is an energy-balance model for pumping-cycle
AWE systems originally proposed by Luchsinger (2013). It analytically
evaluates the cycle power for a given wind shear profile.

The wrapper maps the three awesIO configuration files to the underlying
``PowerModel`` constructor and exposes the AWESPA standard interface.

.. autoclass:: awespa.power.luchsinger_power.LuchsingerPowerModel
   :members:
   :undoc-members:
   :show-inheritance:

Configuration files
~~~~~~~~~~~~~~~~~~~

``LuchsingerPowerModel.load_configuration`` expects three YAML files:

``system_path``
    System configuration in awesIO format (wing area, nominal tether force,
    nominal generator power, cut-in / cut-out wind speeds, etc.).

``simulation_settings_path``
    Simulation settings (operational altitude, atmosphere parameters, etc.).

``wind_resource_path``
    Output of the wind module — altitude profiles, cluster shapes, and the
    wind speed probability matrix.

Usage example
~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from awespa.power.luchsinger_power import LuchsingerPowerModel

   model = LuchsingerPowerModel()
   model.load_configuration(
       system_path=Path("config/example/kitepower V3_20.yml"),
       simulation_settings_path=Path("config/example/luchsinger_settings.yml"),
       wind_resource_path=Path("results/example/wind_resource.yml"),
   )

   # Compute all power curves and save to YAML
   model.compute_power_curves(
       output_path=Path("results/example/power_curves.yml"),
       verbose=True,
       showplot=False,
       saveplot=True,
   )

   # Single operating point
   power_w = model.calculate_power_at_wind_speed(wind_speed=10.0, verbose=True)
   print(f"Power at 10 m/s: {power_w / 1000:.1f} kW")

Or use the ready-made script:

.. code-block:: bash

   python scripts/run_luchsinger.py

InertiaFreeQSMPowerModel
------------------------

The ``InertiaFreeQSMPowerModel`` wraps the
`inertiafree-qsm <https://github.com/jbredael/InertiaFree-QSM>`_
package. The underlying model is an Inertia-Free Quasi-Steady Model (QSM)
that simulates the full pumping cycle with six flight phases. It supports
two power curve generation methods:

``'direct'``
    Fast evaluation using pre-defined cycle parameters from the simulation
    settings file.

``'optimization'``
    Slower, numerically optimises the cycle parameters per wind speed to
    maximise output power.

.. autoclass:: awespa.power.inertiafree_qsm_power.InertiaFreeQSMPowerModel
   :members:
   :undoc-members:
   :show-inheritance:

Configuration files
~~~~~~~~~~~~~~~~~~~

``InertiaFreeQSMPowerModel.load_configuration`` expects three YAML files:

``system_path``
    System configuration in awesIO format (same format as Luchsinger).

``simulation_settings_path``
    QSM-specific settings — cycle parameters, phase settings, optimizer
    bounds, and solver tolerances.

``wind_resource_path``
    Output of the wind module (same file as used by Luchsinger).

Usage example
~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from awespa.power.inertiafree_qsm_power import InertiaFreeQSMPowerModel

   model = InertiaFreeQSMPowerModel()
   model.load_configuration(
       system_path=Path("config/example/kitepower V3_20.yml"),
       simulation_settings_path=Path("config/example/intertiafree-qsm_settings.yml"),
       wind_resource_path=Path("results/example/wind_resource.yml"),
   )

   # Direct simulation (fast)
   model.compute_power_curves(
       output_path=Path("results/example/power_curves_qsm.yml"),
       method="direct",
       verbose=True,
       showplot=False,
       saveplot=True,
   )

   # Optimisation-based (slower, higher fidelity)
   model.compute_power_curves(
       output_path=Path("results/example/power_curves_qsm_optim.yml"),
       method="optimization",
       verbose=True,
   )

Or use the ready-made script:

.. code-block:: bash

   python scripts/run_inertiafree_qsm.py

Output format
-------------

Both models write a YAML file in awesIO format with the following top-level
keys:

* ``reference_wind_speeds_m_s`` — wind speed vector used for evaluation
* ``power_curves`` — list of per-cluster power curve dictionaries, each
  containing the cluster ID, profile metadata, and power values [W]

This file is directly consumed by :func:`awespa.pipeline.aep.calculate_aep`.
