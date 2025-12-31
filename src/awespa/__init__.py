"""AWESPA: Airborne Wind Energy System Performance Assessment Toolchain.

A modular toolchain for assessing airborne wind energy system performance using
wind profile clustering, system optimization, and power estimation.
"""

from . import wind
from . import power
from . import pipeline

# Import key classes for easy access
from .wind import WindProfileModel, WindProfileClusteringModel  
from .power import PowerEstimationModel
from .pipeline import AEPCalculator

__version__ = "1.0.0"

__all__ = [
    # Modules
    'wind',
    'power',
    'pipeline',
    # Key classes
    'WindProfileModel',
    'WindProfileClusteringModel',
    'PowerEstimationModel',
    'AEPCalculator'
]