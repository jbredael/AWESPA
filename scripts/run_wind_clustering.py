#!/usr/bin/env python3
"""
Wind Profile Clustering Script

This script performs wind profile clustering using the AWERA wrapper
and displays the resulting clusters in their normalized form.

Usage:
    python scripts/run_wind_clustering.py

Author: AWERA Development Team
Date: December 2025
"""

import sys
import yaml
from yaml import UnsafeLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path to import awera
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awera.wind.clustering import WindProfileClusteringModel


def run_wind_clustering():
    """Run wind profile clustering and display results."""
    
    print("Wind Profile Clustering Script")
    print("=" * 50)
    
    # Define paths
    config_path = PROJECT_ROOT / "config" / "wind_clustering_config.yml"
    data_path = PROJECT_ROOT / "data"
    results_path = PROJECT_ROOT / "results"
    output_file = results_path / "wind_profiles.yml"
    
    # Ensure results directory exists
    results_path.mkdir(exist_ok=True)
    
    # Initialize the wind clustering model
    print("Step 1: Initializing wind clustering model...")
    model = WindProfileClusteringModel()
    
    # Load configuration
    print("Step 2: Loading configuration...")
    try:
        model.load_from_yaml(config_path)
        print(f"✓ Configuration loaded from {config_path}")
        print(f"  - Clusters: {model.n_clusters}")
        print(f"  - Data source: {model.data_source}")
        print(f"  - Location: {model.location}")
        print(f"  - Years: {model.years[0]}-{model.years[1]}")
        print(f"  - Altitude range: {model.altitude_range[0]}-{model.altitude_range[1]} m")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False
    
    # Perform clustering
    print("\\nStep 3: Performing wind profile clustering...")
    try:
        model.cluster(data_path, output_file)
        print(f"✓ Clustering completed successfully")
        print(f"✓ Results saved to {output_file}")
    except Exception as e:
        print(f"✗ Error during clustering: {e}")
        print("\\nPossible causes:")
        print("- ERA5 data files not found in data/wind_data/era5/")
        print("- Missing dependencies (xarray, netcdf4, etc.)")
        print("- Invalid configuration parameters")
        return False
    
    # Display cluster information from YAML
    print("\nStep 4: Displaying cluster results from YAML...")
    try:
        display_cluster_results_from_yaml(output_file)
    except Exception as e:
        print(f"✗ Error displaying results: {e}")
        return False
    
    # Generate visualization from YAML
    print("\nStep 5: Generating cluster visualization from YAML...")
    try:
        plot_normalized_clusters(output_file, save_path=results_path / "cluster_profiles.png")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        print("Continuing without visualization...")
    
    print("\\n" + "=" * 50)
    print("Wind Profile Clustering Complete!")
    return True


def display_cluster_results_from_yaml(yaml_file_path: Path):
    """Display detailed information about the clustering results from YAML file."""
    
    try:
        with open(yaml_file_path, 'r') as f:
            cluster_data = yaml.load(f, Loader=UnsafeLoader)
    except Exception as e:
        print(f"✗ Error loading YAML file: {e}")
        return
    
    # Extract metadata and cluster information
    metadata = cluster_data['metadata']
    altitudes = np.array(cluster_data['altitudes'])
    clusters = cluster_data['clusters']
    prob_matrix = np.array(cluster_data['probability_matrix']['data'])
    
    # Calculate frequencies
    frequencies = np.sum(prob_matrix, axis=1)
    
    print("Cluster Analysis Summary (from YAML):")
    print("-" * 40)
    print(f"Number of clusters: {metadata['n_clusters']}")
    if 'clustering_parameters' in metadata:
        print(f"Total explained variance: {sum(metadata['clustering_parameters']['explained_variance']):.3f}")
        print(f"Fit inertia: {metadata['clustering_parameters']['fit_inertia']:.0f}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Data source: {metadata['data_source']}")
    print(f"Time range: {metadata['time_range']['start_year']}-{metadata['time_range']['end_year']}")
    
    print("\\nCluster Frequencies:")
    print("-" * 20)
    for i, freq in enumerate(frequencies):
        print(f"Cluster {i+1}: {freq:.2f}% of samples")
    
    print("\\nCluster Profile Statistics:")
    print("-" * 30)
    
    # Display statistics for each cluster
    for i, cluster in enumerate(clusters):
        u_profile = np.array(cluster['u_normalized'])
        v_profile = np.array(cluster['v_normalized'])
        
        # Calculate profile characteristics
        magnitude = np.sqrt(u_profile**2 + v_profile**2)
        max_magnitude = np.max(magnitude)
        max_altitude_idx = np.argmax(magnitude)
        max_altitude = altitudes[max_altitude_idx]
        
        print(f"\\nCluster {cluster['id']} (Frequency: {frequencies[i]:.2f}%):")
        print(f"  Max magnitude: {max_magnitude:.3f} (normalized)")
        print(f"  Max magnitude altitude: {max_altitude:.0f} m")
        print(f"  Surface wind (u,v): ({u_profile[0]:.3f}, {v_profile[0]:.3f})")
        print(f"  Top wind (u,v): ({u_profile[-1]:.3f}, {v_profile[-1]:.3f})")
        
        # Wind direction at different levels
        surface_direction = np.degrees(np.arctan2(v_profile[0], u_profile[0]))
        top_direction = np.degrees(np.arctan2(v_profile[-1], u_profile[-1]))
        print(f"  Surface wind direction: {surface_direction:.1f}°")
        print(f"  Top wind direction: {top_direction:.1f}°")


def plot_normalized_clusters(yaml_file_path: Path, save_path: Path = None):
    """Generate visualization of the normalized wind profile clusters from YAML file."""
    
    # Load cluster data from YAML file
    try:
        with open(yaml_file_path, 'r') as f:
            cluster_data = yaml.load(f, Loader=UnsafeLoader)
        print(f"✓ Loaded cluster data from {yaml_file_path}")
    except Exception as e:
        print(f"✗ Error loading YAML file: {e}")
        return
    
    # Extract data from YAML
    altitudes = np.array(cluster_data['altitudes'])
    clusters = cluster_data['clusters']
    n_clusters = len(clusters)
    
    # Extract frequencies from probability matrix
    prob_matrix = np.array(cluster_data['probability_matrix']['data'])
    frequencies = np.sum(prob_matrix, axis=1)  # Sum probabilities for each cluster
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Normalized Wind Profile Clusters (from YAML)', fontsize=16)
    
    # Plot individual cluster profiles
    for i in range(min(n_clusters, 6)):  # Plot up to 6 clusters
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        cluster = clusters[i]
        u_profile = np.array(cluster['u_normalized'])
        v_profile = np.array(cluster['v_normalized'])
        magnitude = np.sqrt(u_profile**2 + v_profile**2)
        
        # Plot parallel and perpendicular components
        ax.plot(u_profile, altitudes, 'b-', linewidth=2, label='u (parallel)')
        ax.plot(v_profile, altitudes, 'r-', linewidth=2, label='v (perpendicular)')
        ax.plot(magnitude, altitudes, 'k--', linewidth=1.5, label='magnitude')
        
        ax.set_xlabel('Normalized Wind Speed [-]')
        ax.set_ylabel('Height [m]')
        ax.set_title(f'Cluster {cluster["id"]} ({frequencies[i]:.1f}%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([-1.2, 1.5])
        
        # Add zero line
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_clusters, 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster visualization saved to {save_path}")
    
    # Also create a combined plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig2.suptitle('All Wind Profile Clusters - Normalized Components (from YAML)', fontsize=14)
    
    # Color map for clusters
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
        combined_path = save_path.parent / "cluster_components.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"✓ Component visualization saved to {combined_path}")
    
    plt.show()


def main():
    """Main function."""
    
    print("Starting wind profile clustering analysis...")
    
    try:
        success = run_wind_clustering()
        if success:
            print("\\n✓ Analysis completed successfully!")
        else:
            print("\\n✗ Analysis failed. See error messages above.")
    except KeyboardInterrupt:
        print("\\n\\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()