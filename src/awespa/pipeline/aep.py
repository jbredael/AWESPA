"""Annual Energy Production (AEP) calculation pipeline."""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def calculate_aep(
    power_curve_path: Path, 
    wind_resource_path: Path,
    output_path: Optional[Path] = None,
    plot: bool = False,
    plot_output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Calculate Annual Energy Production from power curves and wind resource.
    
    This function computes AEP, capacity factor, and cluster contributions
    from pre-computed power curves and wind resource probability distributions.
    
    Args:
        power_curve_path: Path to power_curves.yml file.
        wind_resource_path: Path to wind_resource.yml file.
        output_path: Optional path to save AEP results YAML. If None, no file is saved.
        plot: If True, generate and display/save plots.
        plot_output_dir: Directory to save plots. If None, plots are shown but not saved.
        
    Returns:
        Dictionary containing AEP results, capacity factor, and cluster contributions.
    """
    # Load data
    with open(power_curve_path, 'r') as f:
        power_data = yaml.safe_load(f)
    
    with open(wind_resource_path, 'r') as f:
        wind_data = yaml.load(f, Loader=yaml.FullLoader)
    
    # Calculate AEP components
    aep_results = _compute_aep_from_data(power_data, wind_data)
    
    # Save results if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(aep_results, f, default_flow_style=False)
        print(f"AEP results saved to {output_path}")
    
    # Generate plots if requested
    if plot:
        _generate_aep_plots(aep_results, power_data, wind_data, plot_output_dir)
    
    return aep_results


def _compute_aep_from_data(power_data: Dict[str, Any], 
                            wind_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute AEP from power curve and wind resource data.
    
    Args:
        power_data: Power curve data dictionary.
        wind_data: Wind resource data dictionary.
        
    Returns:
        Dictionary with AEP results.
    """
    HOURS_PER_YEAR = 8760
    
    # Extract data
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    bin_centers = np.array(power_data['wind_speed_bins']['bin_centers_m_s'])
    
    # Handle both 2D (clusters x wind_speeds) and 3D (clusters x wind_speeds x wind_directions) probability matrices
    if probability_matrix.ndim == 3:
        # Sum across wind direction dimension to get 2D matrix
        probability_matrix_2d = np.sum(probability_matrix, axis=2)
    else:
        probability_matrix_2d = probability_matrix
    
    # Calculate AEP for each cluster
    cluster_contributions = []
    total_aep_wh = 0.0
    
    for i, curve in enumerate(power_data['cluster_power_curves']):
        cluster_id = curve['cluster_id']
        powers = np.array(curve['power_values_w'])
        
        # Get probability distribution for this cluster
        probabilities = probability_matrix_2d[i, :] / 100.0  # Convert % to fraction
        
        # Ensure same length
        min_len = min(len(powers), len(probabilities))
        powers = powers[:min_len]
        probabilities = probabilities[:min_len]
        
        # Calculate expected power for this cluster
        expected_power = np.sum(powers * probabilities)
        
        # Calculate AEP contribution
        cluster_aep_wh = expected_power * HOURS_PER_YEAR
        
        # Get cluster frequency (sum of all probabilities for this cluster)
        cluster_frequency = np.sum(probabilities)
        
        cluster_contributions.append({
            'cluster_id': cluster_id,
            'frequency': float(cluster_frequency),
            'expected_power_w': float(expected_power),
            'aep_wh': float(cluster_aep_wh),
            'aep_kwh': float(cluster_aep_wh / 1000),
            'aep_mwh': float(cluster_aep_wh / 1e6),
        })
        
        total_aep_wh += cluster_aep_wh
    
    # Calculate capacity factor
    rated_power = power_data['aggregate_power_curve']['max_power_w']
    mean_power = total_aep_wh / HOURS_PER_YEAR
    capacity_factor = mean_power / rated_power if rated_power > 0 else 0
    
    return {
        'metadata': {
            'calculation_timestamp': datetime.now().isoformat(),
            'power_curve_source': str(power_data['metadata']),
            'wind_resource_source': str(wind_data['metadata']),
        },
        'total_aep': {
            'wh': float(total_aep_wh),
            'kwh': float(total_aep_wh / 1000),
            'mwh': float(total_aep_wh / 1e6),
            'gwh': float(total_aep_wh / 1e9),
        },
        'rated_power_kw': float(rated_power / 1000),
        'mean_power_kw': float(mean_power / 1000),
        'capacity_factor': float(capacity_factor),
        'cluster_contributions': cluster_contributions,
    }


def _generate_aep_plots(aep_results: Dict[str, Any],
                        power_data: Dict[str, Any],
                        wind_data: Dict[str, Any],
                        output_dir: Optional[Path] = None) -> None:
    """Generate comprehensive AEP analysis plots.
    
    Args:
        aep_results: AEP calculation results.
        power_data: Power curve data.
        wind_data: Wind resource data.
        output_dir: Directory to save plots. If None, plots are shown but not saved.
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Cluster AEP contribution pie chart
    ax1 = plt.subplot(2, 3, 1)
    _plot_cluster_aep_contribution(ax1, aep_results)
    
    # 2. Cluster frequency bar chart
    ax2 = plt.subplot(2, 3, 2)
    _plot_cluster_frequency(ax2, aep_results, wind_data)
    
    # 3. Aggregate power curve
    ax3 = plt.subplot(2, 3, 3)
    _plot_aggregate_power_curve(ax3, power_data)
    
    # 4. Cluster power curves
    ax4 = plt.subplot(2, 3, 4)
    _plot_cluster_power_curves(ax4, power_data)
    
    # 5. Wind speed probability distribution
    ax5 = plt.subplot(2, 3, 5)
    _plot_wind_speed_distribution(ax5, wind_data)
    
    # 6. Capacity factor summary
    ax6 = plt.subplot(2, 3, 6)
    _plot_capacity_factor_summary(ax6, aep_results)
    
    plt.tight_layout()
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "aep_analysis_complete.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def _plot_cluster_aep_contribution(ax, aep_results: Dict[str, Any]) -> None:
    """Plot pie chart of cluster AEP contributions."""
    contributions = aep_results['cluster_contributions']
    cluster_ids = [c['cluster_id'] for c in contributions]
    aep_values = [c['aep_mwh'] for c in contributions]
    
    ax.pie(aep_values, labels=cluster_ids, autopct='%1.1f%%', startangle=90)
    ax.set_title('AEP Contribution by Cluster')


def _plot_cluster_frequency(ax, aep_results: Dict[str, Any], 
                            wind_data: Dict[str, Any]) -> None:
    """Plot bar chart of cluster frequencies."""
    # Calculate frequency from probability matrix
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Handle both 2D and 3D probability matrices
    if probability_matrix.ndim == 3:
        cluster_frequencies = np.sum(probability_matrix, axis=(1, 2)) / 100.0
    else:
        cluster_frequencies = np.sum(probability_matrix, axis=1) / 100.0
        
    cluster_ids = [f"C{i+1}" for i in range(len(cluster_frequencies))]
    
    ax.bar(cluster_ids, cluster_frequencies * 100)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Cluster Occurrence Frequency')
    ax.grid(True, alpha=0.3)


def _plot_aggregate_power_curve(ax, power_data: Dict[str, Any]) -> None:
    """Plot aggregate power curve."""
    aggregate = power_data['aggregate_power_curve']
    wind_speeds = np.array(aggregate['wind_speeds_m_s'])
    powers = np.array(aggregate['power_values_w']) / 1000  # Convert to kW
    
    ax.plot(wind_speeds, powers, 'b-', linewidth=2, label='Aggregate')
    ax.axhline(y=aggregate['max_power_w']/1000, color='r', 
               linestyle='--', label='Rated Power')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Aggregate Power Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_cluster_power_curves(ax, power_data: Dict[str, Any]) -> None:
    """Plot all cluster power curves."""
    for curve in power_data['cluster_power_curves']:
        wind_speeds = np.array(curve['wind_speeds_m_s'])
        powers = np.array(curve['power_values_w']) / 1000  # Convert to kW
        ax.plot(wind_speeds, powers, alpha=0.7, label=f"Cluster {curve['cluster_id']}")
    
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Cluster Power Curves')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_wind_speed_distribution(ax, wind_data: Dict[str, Any]) -> None:
    """Plot wind speed probability distribution."""
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    bin_centers = np.array(wind_data['wind_speed_bins']['bin_centers_m_s'])
    
    # Sum across all clusters (and wind directions if 3D) to get overall distribution
    if probability_matrix.ndim == 3:
        total_distribution = np.sum(probability_matrix, axis=(0, 2)) / 100.0
    else:
        total_distribution = np.sum(probability_matrix, axis=0) / 100.0
    
    ax.bar(bin_centers, total_distribution, width=0.8, alpha=0.7)
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Probability')
    ax.set_title('Overall Wind Speed Distribution')
    ax.grid(True, alpha=0.3)


def _plot_capacity_factor_summary(ax, aep_results: Dict[str, Any]) -> None:
    """Plot capacity factor summary as a bar chart."""
    cf = aep_results['capacity_factor'] * 100
    mean_power = aep_results['mean_power_kw']
    rated_power = aep_results['rated_power_kw']
    
    ax.barh(['Mean Power', 'Rated Power'], [mean_power, rated_power], 
            color=['green', 'blue'], alpha=0.7)
    ax.set_xlabel('Power (kW)')
    ax.set_title(f'Capacity Factor: {cf:.1f}%')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(mean_power, 0, f'{mean_power:.1f} kW', 
            va='center', ha='right', fontsize=10)
    ax.text(rated_power, 1, f'{rated_power:.1f} kW', 
            va='center', ha='right', fontsize=10)
