======
AWESPA
======

**Airborne Wind Energy System Performance Assessment Toolchain**

A modular Python toolchain for assessing Airborne Wind Energy (AWE) system 
performance using wind profile clustering, power estimation models, and 
Annual Energy Production (AEP) calculation.

Getting Started
===============

Overview
--------

AWESPA provides a complete pipeline for AWE system performance analysis:

* **Wind Profile Clustering**: Process ERA5 wind data to identify representative wind profiles
* **Power Estimation**: Compute power curves using physics-based models (e.g., Luchsinger model)
* **AEP Calculation**: Calculate Annual Energy Production, capacity factor, and cluster contributions

Project Structure
-----------------

.. code-block:: text

   AWESPA/
   ├── config/                    # Configuration files (YAML)
   │   ├── wind_clustering_config.yml
   │   └── meridional_case_1/     # Case-specific configurations
   ├── data/                      # Input data (ERA5 wind data)
   ├── processed_data/            # Intermediate processed data
   ├── results/                   # Output results and plots
   ├── scripts/                   # Runnable analysis scripts
   │   ├── run_wind_clustering.py
   │   ├── run_luchsinger.py
   │   ├── compare_power_models.py
   │   └── meridional/            # Case study scripts
   ├── src/awespa/                # Main package source code
   │   ├── wind/                  # Wind modeling components
   │   ├── power/                 # Power estimation models
   │   ├── pipeline/              # AEP calculation pipeline
   │   └── vendor/                # External dependencies
   ├── tests/                     # Test suite
   └── docs/                      # Documentation

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.8 or higher
* pip package manager

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/awegroup/AWESPA.git
      cd AWESPA

2. Create a virtual environment:
   
   **Linux or Mac:**

   .. code-block:: bash

      python3 -m venv venv

   **Windows:**

   .. code-block:: bash

      python -m venv venv

3. Activate the virtual environment:

   **Linux or Mac:**

   .. code-block:: bash

      source venv/bin/activate

   **Windows (PowerShell):**

   .. code-block:: bash

      .\venv\Scripts\Activate

4. Install the package:

   **For users:**

   .. code-block:: bash

      pip install .

   **For developers:**

   .. code-block:: bash

      pip install -e .[dev]

5. To deactivate the virtual environment:

   .. code-block:: bash

      deactivate

Usage
-----

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import awespa

   # Access main components
   from awespa import WindProfileClusteringModel, PowerEstimationModel, calculate_aep

   # Or access modules directly
   from awespa.wind import WindProfileClusteringModel
   from awespa.power import LuchsingerPowerModel
   from awespa.pipeline import calculate_aep

Running Analysis Scripts
~~~~~~~~~~~~~~~~~~~~~~~~

**Wind Profile Clustering:**

.. code-block:: bash

   python scripts/run_wind_clustering.py

**Power Curve Generation:**

.. code-block:: bash

   python scripts/run_luchsinger.py

**Full AEP Analysis (Case Study):**

.. code-block:: bash

   python scripts/meridional/full_aep_analysis_case_1.py

Complete Analysis Pipeline Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from awespa.wind.clustering import WindProfileClusteringModel
   from awespa.power.luchsinger_power import LuchsingerPowerModel
   from awespa.pipeline.aep import calculate_aep

   # 1. Wind Clustering
   wind_model = WindProfileClusteringModel()
   wind_model.load_from_yaml(Path("config/wind_clustering_config.yml"))
   wind_model.cluster(data_path=Path("data"), output_path=Path("results/wind_resource.yml"))

   # 2. Power Curve Generation
   power_model = LuchsingerPowerModel()
   power_model.load_configuration(
       system_path=Path("config/meridional_case_1/soft_kite_pumping_ground_gen_system.yml"),
       simulation_settings_path=Path("config/meridional_case_1/Lucsinger_simulation_settings_config.yml")
   )
   power_model.compute_power_curves(output_path=Path("results/power_curves.yml"), plot=True)

   # 3. AEP Calculation
   aep_results = calculate_aep(
       power_curve_path=Path("results/power_curves.yml"),
       wind_resource_path=Path("results/wind_resource.yml"),
       output_path=Path("results/aep_results.yml"),
       plot=True
   )
   print(f"Annual Energy Production: {aep_results['aep_kwh']:.2f} kWh")

Testing
-------

Run tests using pytest:

.. code-block:: bash

   pytest

Run with coverage:

.. code-block:: bash

   pytest --cov=awespa

Contributing
============

Pull requests are welcome. For major changes, please open an issue first 
to discuss what you would like to change.

Please make sure to update tests as appropriate.

See the `Developer Guide <https://github.com/awegroup/AWESPA/blob/main/README_dev.md>`_ 
for detailed development guidelines.

Resources
=========

* `GitHub Repository <https://github.com/awegroup/AWESPA>`_
* `AWE Group Developer Guide <https://awegroup.github.io/developer-guide/>`_

License
=======

MIT License

Copyright (c) 2024 Airborne Wind Energy Research Group, TU Delft

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
