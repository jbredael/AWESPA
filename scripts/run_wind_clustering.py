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

    config_path = PROJECT_ROOT / "config" / "example" / "wind_clustering_settings.yml"
    # Assuming the wind data is stored in the data directory, adjust as needed
    data_path = PROJECT_ROOT / "data" / "wind_data" / "era5"
    results_path = PROJECT_ROOT / "results" / "example"
    output_file_cluster = results_path / "wind_resource.yml"
    output_file_fit = results_path / "wind_resource_profile_fit.yml"
    output_file_prescribe = results_path / "wind_resource_profile_prescribe.yml"

    results_path.mkdir(exist_ok=True)

    # Initialize the wind profile clustering model
    model = WindProfileClusteringModel()

    # Load configuration parameters from the YAML file
    model.load_configuration(config_path)

    # Below are three functionalities of the model that can be called independently. 
    # Depending on the use case and data availability,
    #  you can choose how to create a wind resource file with the model. For example, if you only have wind data and want to create a wind resource file based on clustering, you can call only the cluster method. If you already have a wind resource file and want to fit profiles to it, you can call only the fit_profile method. If you want to prescribe analytical profiles without using any data, you can call only the prescribe_profile method. In this example, we will call all three methods sequentially to demonstrate the full workflow of the model.
    
    # Perform clustering and export results
    model.cluster(
        dataPath=data_path,
        outputPath=output_file_cluster,
        verbose=True,
        showplot=True,
        saveplot=True,
    )
    # # Fit wind profiles and export results
    # model.fit_profile(
    #     dataPath=data_path,
    #     outputPath=output_file_fit,
    #     verbose=True,
    #     showplot=True,
    #     saveplot=True,
    # )
    # # Prescribe analytical wind profiles and export results
    # model.prescribe_profile(
    #     outputPath=output_file_prescribe,
    #     verbose=True,
    #     showplot=True,
    #     saveplot=True,
    # )
    


if __name__ == "__main__":
    main()