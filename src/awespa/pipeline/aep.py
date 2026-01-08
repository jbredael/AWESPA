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
    
    Supports both old format (wind_speed_bins, cluster_power_curves) and
    new AWESIO format (reference_wind_speeds_m_s, power_curves).
    
    Args:
        power_data: Power curve data dictionary.
        wind_data: Wind resource data dictionary.
        
    Returns:
        Dictionary with AEP results.
    """
    HOURS_PER_YEAR = 8760
    
    # Extract probability data from wind resource
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Handle both 2D (clusters x wind_speeds) and 3D (clusters x wind_speeds x wind_directions) probability matrices
    if probability_matrix.ndim == 3:
        # Sum across wind direction dimension to get 2D matrix
        probability_matrix_2d = np.sum(probability_matrix, axis=2)
    else:
        probability_matrix_2d = probability_matrix
    
    # Detect power curve format
    if 'reference_wind_speeds_m_s' in power_data:
        # New AWESIO format
        bin_centers = np.array(power_data['reference_wind_speeds_m_s'])
        power_curves = power_data['power_curves']
        is_new_format = True
    else:
        # Old format
        bin_centers = np.array(power_data['wind_speed_bins']['bin_centers_m_s'])
        power_curves = power_data['cluster_power_curves']
        is_new_format = False
    
    # Get wind speed bins from wind resource for probability matching
    wind_bin_centers = np.array(wind_data['wind_speed_bins']['bin_centers_m_s'])
    
    # Calculate AEP for each cluster/profile
    cluster_contributions = []
    total_aep_wh = 0.0
    
    # Check if we have a single flat profile (like Luchsinger model)
    # In this case, aggregate all cluster probabilities
    if is_new_format and len(power_curves) == 1:
        # Single flat profile - aggregate all cluster probabilities
        curve = power_curves[0]
        profile_id = curve.get('profile_id', 1)
        powers = np.array(curve['cycle_power_w'])
        
        # Interpolate power values to wind resource wind speed bins if needed
        if len(bin_centers) != len(wind_bin_centers):
            powers_interp = np.interp(wind_bin_centers, bin_centers, powers)
        else:
            powers_interp = powers
        
        # Aggregate probabilities across all clusters (sum over cluster dimension)
        # This gives the overall wind speed distribution regardless of wind shear profile
        probabilities = np.sum(probability_matrix_2d, axis=0) / 100.0  # Convert % to fraction
        
        # Ensure same length
        min_len = min(len(powers_interp), len(probabilities))
        powers_interp = powers_interp[:min_len]
        probabilities = probabilities[:min_len]
        
        # Calculate expected power using aggregated probabilities
        expected_power = np.sum(powers_interp * probabilities)
        
        # Calculate total AEP
        total_aep_wh = expected_power * HOURS_PER_YEAR
        
        # Get frequency (should sum to ~1.0)
        total_frequency = np.sum(probabilities)
        
        cluster_contributions.append({
            'cluster_id': profile_id,
            'frequency': float(total_frequency),
            'expected_power_w': float(expected_power),
            'aep_wh': float(total_aep_wh),
            'aep_kwh': float(total_aep_wh / 1000),
            'aep_mwh': float(total_aep_wh / 1e6),
        })
    else:
        # Multiple profiles - calculate per cluster/profile
        for i, curve in enumerate(power_curves):
            if is_new_format:
                profile_id = curve.get('profile_id', i + 1)
                powers = np.array(curve['cycle_power_w'])
                probability_weight = curve.get('probability_weight', 1.0 / len(power_curves))
            else:
                profile_id = curve.get('cluster_id', i + 1)
                powers = np.array(curve['power_values_w'])
                probability_weight = curve.get('frequency', 1.0 / len(power_curves))
            
            # Interpolate power values to wind resource wind speed bins if needed
            if len(bin_centers) != len(wind_bin_centers):
                powers_interp = np.interp(wind_bin_centers, bin_centers, powers)
            else:
                powers_interp = powers
            
            # Get probability distribution for this cluster from wind resource
            if i < probability_matrix_2d.shape[0]:
                probabilities = probability_matrix_2d[i, :] / 100.0  # Convert % to fraction
            else:
                # Fallback: equal distribution
                probabilities = np.ones(len(wind_bin_centers)) / len(wind_bin_centers)
            
            # Ensure same length
            min_len = min(len(powers_interp), len(probabilities))
            powers_interp = powers_interp[:min_len]
            probabilities = probabilities[:min_len]
            
            # Calculate expected power for this cluster
            expected_power = np.sum(powers_interp * probabilities)
            
            # Calculate AEP contribution
            cluster_aep_wh = expected_power * HOURS_PER_YEAR
            
            # Get cluster frequency (sum of all probabilities for this cluster)
            cluster_frequency = np.sum(probabilities)
            
            cluster_contributions.append({
                'cluster_id': profile_id,
                'frequency': float(cluster_frequency),
                'expected_power_w': float(expected_power),
                'aep_wh': float(cluster_aep_wh),
                'aep_kwh': float(cluster_aep_wh / 1000),
                'aep_mwh': float(cluster_aep_wh / 1e6),
            })
            
            total_aep_wh += cluster_aep_wh
    
    # Calculate rated power and capacity factor
    if is_new_format:
        # For new format, get max power from the first (and possibly only) profile
        all_powers = []
        for curve in power_curves:
            all_powers.extend(curve['cycle_power_w'])
        rated_power = max(all_powers) if all_powers else 0.0
    else:
        rated_power = power_data['aggregate_power_curve']['max_power_w']
    
    mean_power = total_aep_wh / HOURS_PER_YEAR
    capacity_factor = mean_power / rated_power if rated_power > 0 else 0
    
    return {
        'metadata': {
            'calculation_timestamp': datetime.now().isoformat(),
            'power_curve_source': str(power_data.get('metadata', {})),
            'wind_resource_source': str(wind_data.get('metadata', {})),
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
    fig = plt.figure(figsize=(18, 15))
    
    # 1. Cluster AEP contribution pie chart
    ax1 = plt.subplot(3, 3, 1)
    _plot_cluster_aep_contribution(ax1, aep_results)
    
    # 2. Cluster frequency bar chart
    ax2 = plt.subplot(3, 3, 2)
    _plot_cluster_frequency(ax2, aep_results, wind_data)
    
    # 3. Aggregate power curve
    ax3 = plt.subplot(3, 3, 3)
    _plot_aggregate_power_curve(ax3, power_data)
    
    # 4. Cluster power curves
    ax4 = plt.subplot(3, 3, 4)
    _plot_cluster_power_curves(ax4, power_data)
    
    # 5. Wind speed probability distribution
    ax5 = plt.subplot(3, 3, 5)
    _plot_wind_speed_distribution(ax5, wind_data)
    
    # 6. Capacity factor summary
    ax6 = plt.subplot(3, 3, 6)
    _plot_capacity_factor_summary(ax6, aep_results)
    
    # 7. Wind rose - power by direction
    ax7 = plt.subplot(3, 3, 7, projection='polar')
    _plot_wind_rose_power(ax7, power_data, wind_data, aep_results)
    
    # 8. Wind rose - frequency by direction
    ax8 = plt.subplot(3, 3, 8, projection='polar')
    _plot_wind_rose_frequency(ax8, wind_data)
    
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
    # Handle both old and new AWESIO formats
    if 'reference_wind_speeds_m_s' in power_data:
        # New AWESIO format - use first power curve as aggregate
        wind_speeds = np.array(power_data['reference_wind_speeds_m_s'])
        powers = np.array(power_data['power_curves'][0]['cycle_power_w']) / 1000  # Convert to kW
        max_power = max(powers)
    else:
        # Old format
        aggregate = power_data['aggregate_power_curve']
        wind_speeds = np.array(aggregate['wind_speeds_m_s'])
        powers = np.array(aggregate['power_values_w']) / 1000  # Convert to kW
        max_power = aggregate['max_power_w'] / 1000
    
    ax.plot(wind_speeds, powers, 'b-', linewidth=2, label='Power Curve')
    ax.axhline(y=max_power, color='r', linestyle='--', label='Rated Power')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Aggregate Power Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_cluster_power_curves(ax, power_data: Dict[str, Any]) -> None:
    """Plot all cluster power curves."""
    # Handle both old and new AWESIO formats
    if 'reference_wind_speeds_m_s' in power_data:
        # New AWESIO format
        wind_speeds = np.array(power_data['reference_wind_speeds_m_s'])
        for curve in power_data['power_curves']:
            powers = np.array(curve['cycle_power_w']) / 1000  # Convert to kW
            ax.plot(wind_speeds, powers, alpha=0.7, 
                    label=f"Profile {curve['profile_id']}")
    else:
        # Old format
        for curve in power_data['cluster_power_curves']:
            wind_speeds = np.array(curve['wind_speeds_m_s'])
            powers = np.array(curve['power_values_w']) / 1000  # Convert to kW
            ax.plot(wind_speeds, powers, alpha=0.7, 
                    label=f"Cluster {curve['cluster_id']}")
    
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Curves')
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
    
    # Calculate bar width from bin spacing for clean visualization
    if len(bin_centers) > 1:
        bar_width = np.mean(np.diff(bin_centers)) * 0.9
    else:
        bar_width = 0.5
    
    ax.bar(bin_centers, total_distribution, width=bar_width, alpha=0.7, color='steelblue', edgecolor='navy')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Probability')
    ax.set_title('Overall Wind Speed Distribution')
    ax.grid(True, alpha=0.3, axis='y')


def _plot_capacity_factor_summary(ax, aep_results: Dict[str, Any]) -> None:
    """Plot capacity factor summary as a bar chart."""
    cf = aep_results['capacity_factor'] * 100
    mean_power = aep_results['mean_power_kw']
    rated_power = aep_results['rated_power_kw']
    
    # Calculate average expected power across all clusters
    avg_expected_power = np.mean([c['expected_power_w'] for c in aep_results['cluster_contributions']]) / 1000
    
    labels = ['Mean Power\n(Annual Avg)', 'Expected Power\n(Weighted Avg)', 'Rated Power\n(Max)']
    values = [mean_power, avg_expected_power, rated_power]
    colors = ['green', 'orange', 'blue']
    
    bars = ax.barh(labels, values, color=colors, alpha=0.7)
    ax.set_xlabel('Power (kW)')
    ax.set_title(f'Power Summary - Capacity Factor: {cf:.1f}%')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add text annotations
    for i, (val, bar) in enumerate(zip(values, bars)):
        ax.text(val, i, f' {val:.1f} kW', va='center', ha='left', fontsize=9, fontweight='bold')


def _plot_wind_rose_power(ax, power_data: Dict[str, Any], wind_data: Dict[str, Any], 
                          aep_results: Dict[str, Any]) -> None:
    """Plot wind rose showing power contribution by wind direction."""
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Check if we have directional data
    if probability_matrix.ndim < 3:
        ax.text(0.5, 0.5, 'No directional\ndata available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Power by Wind Direction')
        return
    
    # Get wind speeds and power values
    if 'reference_wind_speeds_m_s' in power_data:
        wind_speeds = np.array(power_data['reference_wind_speeds_m_s'])
        powers = np.array(power_data['power_curves'][0]['cycle_power_w'])
    else:
        return
    
    # Calculate power contribution per direction
    # Sum across all clusters and wind speeds, weighted by probability and power
    n_directions = probability_matrix.shape[2]
    power_by_direction = np.zeros(n_directions)
    
    wind_bin_centers = np.array(wind_data['wind_speed_bins']['bin_centers_m_s'])
    powers_interp = np.interp(wind_bin_centers, wind_speeds, powers)
    
    for d in range(n_directions):
        # Sum across clusters and wind speeds
        direction_prob = probability_matrix[:, :, d] / 100.0  # Convert % to fraction
        power_by_direction[d] = np.sum(direction_prob * powers_interp[np.newaxis, :])
    
    # Convert to kW and normalize for visualization
    power_by_direction_kw = power_by_direction / 1000.0
    
    # Wind directions in radians (0 = North, clockwise)
    direction_bin_width = wind_data['metadata']['wind_direction_bin_width_deg']
    theta = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    width = np.deg2rad(direction_bin_width)
    
    # Create polar bar plot
    bars = ax.bar(theta, power_by_direction_kw, width=width, bottom=0.0, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Color bars by magnitude
    colors = plt.cm.YlOrRd(power_by_direction_kw / power_by_direction_kw.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Power Contribution by Wind Direction', pad=20)
    ax.set_ylabel('Power (kW)', labelpad=30)


def _plot_wind_rose_frequency(ax, wind_data: Dict[str, Any]) -> None:
    """Plot wind rose showing wind frequency by direction."""
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Check if we have directional data
    if probability_matrix.ndim < 3:
        ax.text(0.5, 0.5, 'No directional\ndata available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Wind Frequency by Direction')
        return
    
    # Sum across clusters and wind speeds to get frequency per direction
    n_directions = probability_matrix.shape[2]
    freq_by_direction = np.sum(probability_matrix, axis=(0, 1)) / 100.0  # Convert % to fraction
    
    # Wind directions in radians (0 = North, clockwise)
    direction_bin_width = wind_data['metadata']['wind_direction_bin_width_deg']
    theta = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    width = np.deg2rad(direction_bin_width)
    
    # Create polar bar plot
    bars = ax.bar(theta, freq_by_direction * 100, width=width, bottom=0.0, alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    
    # Color bars by magnitude
    colors = plt.cm.Blues(freq_by_direction / freq_by_direction.max())
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Wind Frequency by Direction', pad=20)
    ax.set_ylabel('Frequency (%)', labelpad=30)
