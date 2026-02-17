#!/usr/bin/env python3
"""
Wind Profile Clustering Script

This script performs wind profile clustering using the AWESPA wrapper
and displays the resulting clusters in their normalized form.

Usage:
    python scripts/run_wind_clustering.py

Author: Joren Bredael
Date: December 2025
"""

import sys
import yaml
from yaml import UnsafeLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.wind.clustering import WindProfileClusteringModel


def run_wind_clustering():
    """Run wind profile clustering and display results."""
    
    # Define paths
    config_path = PROJECT_ROOT / "config" / "wind_clustering_config.yml"
    data_path = PROJECT_ROOT / "data"
    results_path = PROJECT_ROOT / "results"
    output_file = results_path / "wind_resource.yml"
    
    # Also save a copy to the main config directory for other components to use
    config_output_file = PROJECT_ROOT / "config" / "wind_resource.yml"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    # Initialize and configure model
    model = WindProfileClusteringModel()
    try:
        model.load_from_yaml(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Perform clustering
    try:
        model.cluster(data_path, output_file)
        # Copy to config directory for other components
        import shutil
        shutil.copy2(output_file, config_output_file)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return False
    
    # Display results and generate plots
    try:
        display_cluster_results_from_yaml(output_file)
        plot_normalized_clusters(output_file, save_path=results_path / "cluster_profiles.pdf")
    except Exception as e:
        print(f"Error displaying results: {e}")
        return False
    
    return True


def display_cluster_results_from_yaml(yaml_file_path: Path):
    """Display detailed information about the clustering results from YAML file."""
    
    with open(yaml_file_path, 'r') as f:
        cluster_data = yaml.load(f, Loader=UnsafeLoader)
    
    # Extract metadata and cluster information
    metadata = cluster_data['metadata']
    altitudes = np.array(cluster_data['altitudes'])
    clusters = cluster_data['clusters']
    prob_matrix = np.array(cluster_data['probability_matrix']['data'])
    
    # Calculate frequencies (sum over wind speed and direction bins for each cluster)
    frequencies = np.sum(prob_matrix, axis=(1, 2))
    
    print(f"Clusters: {metadata['n_clusters']}, Samples: {metadata['total_samples']}")
    print(f"Time range: {metadata['time_range']['start_year']}-{metadata['time_range']['end_year']}")
    
    print("\nCluster Frequencies:")
    for i, freq in enumerate(frequencies):
        print(f"  Cluster {i+1}: {freq:.1f}%")
    
    print("\nCluster Statistics:")
    # Display statistics for each cluster
    for i, cluster in enumerate(clusters):
        u_profile = np.array(cluster['u_normalized'])
        v_profile = np.array(cluster['v_normalized'])
        magnitude = np.sqrt(u_profile**2 + v_profile**2)
        max_magnitude = np.max(magnitude)
        max_altitude_idx = np.argmax(magnitude)
        max_altitude = altitudes[max_altitude_idx]
        
        print(f"  Cluster {cluster['id']} ({frequencies[i]:.1f}%): "
              f"Max {max_magnitude:.2f} at {max_altitude:.0f}m, "
              f"Surface ({u_profile[0]:.2f}, {v_profile[0]:.2f})")


def plot_normalized_clusters(yaml_file_path: Path, save_path: Path = None):
    """Generate visualization of the normalized wind profile clusters from YAML file."""
    
    # Load cluster data from YAML file
    with open(yaml_file_path, 'r') as f:
        cluster_data = yaml.load(f, Loader=UnsafeLoader)
    
    # Extract data from YAML
    altitudes = np.array(cluster_data['altitudes'])
    clusters = cluster_data['clusters']
    n_clusters = len(clusters)
    
    # Extract frequencies from probability matrix (sum over wind speed and direction bins)
    prob_matrix = np.array(cluster_data['probability_matrix']['data'])
    frequencies = np.sum(prob_matrix, axis=(1, 2))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Normalized Wind Profile Clusters', fontsize=16)
    
    # Plot individual cluster profiles
    for i in range(min(n_clusters, 6)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        cluster = clusters[i]
        u_profile = np.array(cluster['u_normalized'])
        v_profile = np.array(cluster['v_normalized'])
        magnitude = np.sqrt(u_profile**2 + v_profile**2)
        
        ax.plot(u_profile, altitudes, 'b-', linewidth=2, label='u (parallel)')
        ax.plot(v_profile, altitudes, 'r-', linewidth=2, label='v (perpendicular)')
        ax.plot(magnitude, altitudes, 'k--', linewidth=1.5, label='magnitude')
        
        ax.set_xlabel('Normalized Wind Speed [-]')
        ax.set_ylabel('Height [m]')
        ax.set_title(f'Cluster {cluster["id"]} ({frequencies[i]:.1f}%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([-1.2, 1.5])
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_clusters, 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Create combined plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig2.suptitle('All Wind Profile Clusters - Normalized Components', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot parallel components
    for i, cluster in enumerate(clusters):
        u_profile = np.array(cluster['u_normalized'])
        ax1.plot(u_profile, altitudes, color=colors[i], linewidth=2, 
                label=f'Cluster {cluster["id"]} ({frequencies[i]:.1f}%)')
    
    ax1.set_xlabel('Parallel Component (u) [-]')
    ax1.set_ylabel('Height [m]')
    ax1.set_title('Parallel Wind Components')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # Plot perpendicular components
    for i, cluster in enumerate(clusters):
        v_profile = np.array(cluster['v_normalized'])
        ax2.plot(v_profile, altitudes, color=colors[i], linewidth=2,
                label=f'Cluster {cluster["id"]} ({frequencies[i]:.1f}%)')
    
    ax2.set_xlabel('Perpendicular Component (v) [-]')
    ax2.set_ylabel('Height [m]')
    ax2.set_title('Perpendicular Wind Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        combined_path = save_path.parent / "cluster_components.pdf"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main function."""
    
    try:
        success = run_wind_clustering()
        if not success:
            print("Analysis failed.")
    except KeyboardInterrupt:
        print("Analysis interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()