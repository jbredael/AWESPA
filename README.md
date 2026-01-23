# AWESPA - Airborne Wind Energy System Performance Assessment

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jbredael.github.io/AWESPA/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A modular Python toolchain for assessing Airborne Wind Energy (AWE) system performance using wind profile clustering, power estimation models, and Annual Energy Production (AEP) calculation.

## Overview

AWESPA provides a complete pipeline for AWE system performance analysis:

- **Wind Profile Clustering**: Process ERA5 wind data to identify representative wind profiles
- **Power Estimation**: Compute power curves using physics-based models (e.g., Luchsinger model)
- **AEP Calculation**: Calculate Annual Energy Production, capacity factor, and cluster contributions

## Project Structure

```
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
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/awegroup/AWESPA.git
    cd AWESPA
    ```

2. Create a virtual environment:
   
    Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
3. Activate the virtual environment:

    Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows (PowerShell):
    ```bash
    .\venv\Scripts\Activate
    ```

4. Install the package:

    For users:
    ```bash
    pip install .
    ```
        
    For developers:
    ```bash
    pip install -e .[dev]
    ```

5. To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Usage

### Quick Start

```python
import awespa

# Access main components
from awespa import WindProfileClusteringModel, PowerEstimationModel, calculate_aep

# Or access modules directly
from awespa.wind import WindProfileClusteringModel
from awespa.power import LuchsingerPowerModel
from awespa.pipeline import calculate_aep
```

### Running Analysis Scripts

**Wind Profile Clustering:**
```bash
python scripts/run_wind_clustering.py
```

**Power Curve Generation:**
```bash
python scripts/run_luchsinger.py
```

**Full AEP Analysis (Case Study):**
```bash
python scripts/meridional/full_aep_analysis_case_1.py
```

### Example: Complete Analysis Pipeline

```python
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
```

## Testing

Run tests using pytest:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=awespa
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

See [README_dev.md](README_dev.md) for detailed development guidelines.

## License

MIT License

Copyright (c) 2024 Airborne Wind Energy Research Group, TU Delft

Technische Universiteit Delft hereby disclaims all copyright interest in the program "AWESPA" (Airborne Wind Energy System Performance Assessment Toolchain) written by the Author(s).

Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering

## Help and Documentation

- [**AWESPA Documentation**](https://awegroup.github.io/AWESPA/) - Getting started guide and API reference
- [AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)
- [AWESPA GitHub Repository](https://github.com/awegroup/AWESPA)


