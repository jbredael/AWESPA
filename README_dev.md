# AWESPA Developer Guide

Welcome to the AWESPA development guide. This document contains instructions for developers working on the AWESPA project.

## Introduction

AWESPA (Airborne Wind Energy System Performance Assessment) is a modular Python toolchain with three main components:
- **Wind Module**: Wind profile clustering from ERA5 data
- **Power Module**: Physics-based power estimation models
- **Pipeline Module**: AEP calculation orchestration

## Project Architecture

```
src/awespa/
├── __init__.py              # Package initialization and exports
├── wind/                    # Wind modeling components
│   ├── base.py              # WindProfileModel ABC
│   └── clustering.py        # WindProfileClusteringModel implementation
├── power/                   # Power estimation models
│   ├── base.py              # PowerEstimationModel ABC
│   ├── awe_power.py         # AWE power model
│   └── luchsinger_power.py  # Luchsinger power model
├── pipeline/                # Pipeline orchestration
│   └── aep.py               # AEP calculation functions
└── vendor/                  # External dependencies
    ├── AWE_production_estimation/
    ├── LuchsingerPowerModel/
    └── wind-profile-clustering/
```

## Development Guidelines

### Branch Management
- Work with `main` branch for stable releases
- Create feature branches for implementing new features
- Merge via Pull Request once features are complete and tested

### Configuration Files
- Write user settings in `.yaml` files in the `config/` directory
- Store case-specific configurations in subdirectories (e.g., `config/meridional_case_1/`)

### Code Organization
- All essential code resides in `src/awespa/`
- Install the package using `pip install -e .` for development
- Import using `import awespa` or `from awespa import ...`

### Data Management
- Raw data → `data/`
- Processed data → `processed_data/`
- Results → `results/`
- Results should include the settings used for reproducibility

### Git Ignore Policy
The folders `data/`, `processed_data/`, and `results/` are in `.gitignore` because they may contain:
- Large files unsuitable for version control
- Confidential data
- Generated data that can be recreated

## Setting Up Your Development Environment

1. Clone the repository:
    ```bash
    git clone git@github.com:awegroup/AWESPA.git
    cd AWESPA
    ```

2. Create and activate a virtual environment:
    ```bash
    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    
    # Windows (PowerShell)
    python -m venv venv
    .\venv\Scripts\Activate
    ```

3. Install in development mode:
    ```bash
    pip install -e .[dev]
    ```

4. Verify installation:
    ```bash
    python -c "import awespa; print(awespa.__version__)"
    ```

## Workflow for Implementing New Features

1. **Setup Environment**
    ```bash
    cd AWESPA
    # Remove old venv if necessary
    python -m venv venv
    source venv/bin/activate  # or .\venv\Scripts\Activate on Windows
    pip install -e .[dev]
    ```

2. **Create Issue on GitHub**
   - Describe the feature or bug fix
   - Assign labels and milestone

3. **Create Feature Branch**
    ```bash
    git checkout -b feature/<issue-number>-<short-description>
    ```

4. **Implement Your Feature**
   - Follow naming conventions (see below)
   - Add/update docstrings
   - Write tests

5. **Verify with Tests**
    ```bash
    pytest
    pytest --cov=awespa
    ```

6. **Commit and Push**
    ```bash
    git add .
    git commit -m "#<issue-number> <descriptive commit message>"
    git push -u origin feature/<issue-number>-<short-description>
    ```

7. **Create Pull Request**
   - Open PR on GitHub
   - Link to the issue
   - Request review

8. **After Merge**
    ```bash
    git fetch --prune
    git checkout main
    git pull
    git branch -d feature/<issue-number>-<short-description>
    ```

## Naming Conventions

| Element       | Convention         | Examples                            |
|---------------|-------------------|-------------------------------------|
| Functions     | snake_case        | `compute_force`, `calculate_power`  |
| Variables     | mixedCase         | `windSpeed`, `turbinePower`         |
| Classes       | CamelCase         | `WindModel`, `PowerCalculator`      |
| Methods       | snake_case        | `get_values`, `update_state`        |
| Constants     | UPPER_SNAKE_CASE  | `AIR_DENSITY`, `MAX_ITERATIONS`     |
| Modules       | snake_case        | `utilities.py`, `aero_model.py`     |
| Packages      | lowercase         | `simulation`, `windanalysis`        |
| Booleans      | is_ prefix or UPPER| `is_valid`, `IS_ACTIVE`            |

## Docstring Conventions

Use Google-style docstrings:

```python
def calculate_power(wind_speed: float, air_density: float = 1.225) -> float:
    """Calculate power output at given wind speed.

    Computes mechanical power using simplified physics model.

    Args:
        wind_speed: Wind speed in m/s.
        air_density: Air density in kg/m³. Defaults to 1.225.

    Raises:
        ValueError: If wind_speed is negative.

    Returns:
        Power output in Watts.
    """
    if wind_speed < 0:
        raise ValueError("Wind speed cannot be negative")
    return 0.5 * air_density * wind_speed ** 3
```

## Testing Guidelines

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_wind_clustering.py  # Wind module tests
├── test_power_models.py     # Power module tests
└── test_aep_pipeline.py     # Pipeline tests
```

### Test Naming
- Test files: `test_<module>.py`
- Test functions: `test_<function>_<scenario>`

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=awespa --cov-report=html

# Run specific test file
pytest tests/test_power_models.py

# Run specific test function
pytest tests/test_power_models.py::test_luchsinger_power_curve
```

### Test Requirements
- Target ~70% code coverage
- Tests should be independent
- Tests should run quickly
- Mock external dependencies when appropriate

## Packaging with pyproject.toml

### Key Configuration Fields

```toml
[project]
name = "awespa"
version = "1.0.0"
description = "Airborne Wind Energy System Performance Assessment Toolchain"
requires-python = ">=3.8"

dependencies = [
    "numpy",
    "pandas>=1.5.3",
    "matplotlib>=3.7.1",
    "pyyaml",
    "scipy",
    "scikit-learn",
    "xarray",
    "netCDF4"
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "black", "flake8", "isort", "mypy"]
docs = ["sphinx", "sphinx-rtd-theme", "myst-parser"]
```

### Using the Package

```python
# Import the main package
import awespa

# Import specific components
from awespa import WindProfileClusteringModel, calculate_aep
from awespa.power import LuchsingerPowerModel

# Access version
print(awespa.__version__)
```

## Adding New Power Models

To add a new power estimation model:

1. Create a new file in `src/awespa/power/` (e.g., `my_power_model.py`)

2. Inherit from the abstract base class:
    ```python
    from awespa.power.base import PowerEstimationModel
    
    class MyPowerModel(PowerEstimationModel):
        def load_configuration(self, system_path, simulation_settings_path, 
                               operational_constraints_path=None):
            # Implementation
            pass
        
        def compute_power_curves(self, output_path, plot=False):
            # Implementation
            pass
        
        def calculate_power_at_wind_speed(self, wind_speed, output_path=None, 
                                          plot=False):
            # Implementation
            pass
    ```

3. Export in `src/awespa/power/__init__.py`:
    ```python
    from .my_power_model import MyPowerModel
    __all__ = [..., 'MyPowerModel']
    ```

4. Add tests in `tests/test_power_models.py`

## Documentation

### Building Documentation

```bash
pip install -e .[docs]
cd docs
make html
```

Documentation is built using Sphinx with the ReadTheDocs theme.

## Continuous Integration

Tests run automatically on:
- Push to `main`
- Pull requests

The CI pipeline includes:
- Running pytest
- Coverage reporting
- Linting checks

## Resources

- [AWE Group Developer Guide](https://awegroup.github.io/developer-guide/)
- [AWESPA Repository](https://github.com/awegroup/AWESPA)
- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
