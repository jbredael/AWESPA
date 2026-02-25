#!/usr/bin/env python3
"""
Wind Profile Clustering Script

Performs wind profile clustering using the AWESPA wrapper and saves the
results to the results and config directories.

Usage:
    python scripts/run_wind_clustering.py

Author: Joren Bredael
Date: December 2025
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.wind.clustering import WindProfileClusteringModel


def main():
    """Run wind profile clustering."""

    config_path = PROJECT_ROOT / "config" / "wind_clustering_config.yml"
    data_path = PROJECT_ROOT / "data"
    results_path = PROJECT_ROOT / "results"
    output_file = results_path / "wind_resource.yml"

    results_path.mkdir(exist_ok=True)

    model = WindProfileClusteringModel()
    model.load_from_yaml(config_path)
    model.cluster(
        dataPath=data_path,
        outputPath=output_file,
        verbose=True,
        showplot=True,
        saveplot=True,
        plotpath=results_path / "plots",
    )

if __name__ == "__main__":
    main()