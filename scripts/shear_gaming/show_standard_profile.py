#!/usr/bin/env python3
"""
Show Standard Profile Comparison

This script visualizes the standard wind shear profile (α = 0.2) compared
to all measured profiles, and highlights which measured profile is closest
to the standard.
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy import stats

# Configuration
ALPHA_STD = 0.2
Z_REF = 200.0
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "shear_gaming"
WIND_RESOURCE_PATH = RESULTS_DIR / "wind_resource.yml"


def fit_power_law_shear(altitudes, wind_speeds, z_ref):
    """Fit power law to determine shear exponent."""
    valid_mask = (altitudes > 0) & (wind_speeds > 0)
    z_valid = altitudes[valid_mask]
    v_valid = wind_speeds[valid_mask]
    
    log_z_ratio = np.log(z_valid / z_ref)
    log_v = np.log(v_valid)
    
    slope, intercept, _, _, _ = stats.linregress(log_z_ratio, log_v)
    return slope, np.exp(intercept)


def plot_probability_distribution(profile_probs, closest_idx):
    """Create simple probability distribution bar chart."""
    n_profiles = len(profile_probs)
    profile_ids = np.arange(1, n_profiles + 1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Bar chart of probabilities by profile
    colors_bar = ['red' if i == closest_idx else 'steelblue' for i in range(n_profiles)]
    bars = ax.bar(profile_ids, profile_probs, color=colors_bar, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Profile ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability of Occurrence', fontsize=14, fontweight='bold')
    ax.set_title('Wind Shear Profile Probability Distribution', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(profile_ids)
    ax.set_ylim([0, max(profile_probs) * 1.1])
    
    # Add mean line
    mean_prob = np.mean(profile_probs)
    ax.axhline(y=mean_prob, color='black', linestyle='--', linewidth=2,
               label=f'Mean = {mean_prob:.4f}', alpha=0.7)
    
    # Add legend for closest profile
    legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='Measured profiles'),
                      Patch(facecolor='red', edgecolor='black', label=f'Closest to standard (Profile {closest_idx+1})'),
                      plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=f'Mean probability')]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    # Save as PDF
    output_path = RESULTS_DIR / "probability_distribution.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Probability distribution plot saved to: {output_path}")
    
    plt.close()


def main():
    """Show standard profile and comparison."""
    print("=" * 80)
    print("STANDARD PROFILE VISUALIZATION")
    print("=" * 80)
    
    # Load wind data
    with open(WIND_RESOURCE_PATH, 'r') as f:
        wind_data = yaml.safe_load(f)
    
    altitudes = np.array(wind_data['altitudes'])
    clusters = wind_data['clusters']
    
    # Calculate probability matrix
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    profile_probs = np.sum(probability_matrix, axis=(1, 2)) / 100.0
    
    # Process all profiles to get alpha values
    alphas = []
    for cluster in clusters:
        u_norm = np.array(cluster['u_normalized'])
        v_ref_sample = 10.0  # arbitrary reference
        wind_speeds = u_norm * v_ref_sample
        alpha, _ = fit_power_law_shear(altitudes, wind_speeds, Z_REF)
        alphas.append(alpha)
    
    alphas = np.array(alphas)
    
    # Find profile closest to standard
    alpha_diffs = np.abs(alphas - ALPHA_STD)
    closest_idx = np.argmin(alpha_diffs)
    closest_profile_id = closest_idx + 1
    
    print(f"\nStandard shear exponent: α_std = {ALPHA_STD}")
    print(f"\nProfile closest to standard:")
    print(f"  Profile {closest_profile_id}: α = {alphas[closest_idx]:.4f}")
    print(f"  Difference: {alpha_diffs[closest_idx]:.4f}")
    print(f"  Probability: {profile_probs[closest_idx]:.4f}")
    
    # Create simple wind profile visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Generate standard profile (normalized so value at z_ref = 1)
    z_plot = np.linspace(0, 500, 100)
    v_std_normalized = (z_plot / Z_REF) ** ALPHA_STD
    
    # Plot standard profile
    ax.plot(v_std_normalized, z_plot, 'k-', linewidth=3.5, label=f'Standard (α={ALPHA_STD})', zorder=10)
    
    # Create distinct colors and line styles for each profile
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    line_styles = ['-', '--', '-.', ':']
    
    # Plot all 20 measured profiles with labels (already normalized)
    for i, cluster in enumerate(clusters):
        u_norm = np.array(cluster['u_normalized'])
        
        color = colors[i]
        line_style = line_styles[i % 4]
        
        if i == closest_idx:
            # Use red for closest profile with solid thick line
            ax.plot(u_norm, altitudes, 'r-', linewidth=2.5, 
                    label=f'Profile {i+1} (closest, α={alphas[i]:.3f})', zorder=9)
        else:
            ax.plot(u_norm, altitudes, color=color, linestyle=line_style, linewidth=1.5,
                    label=f'Profile {i+1} (α={alphas[i]:.3f})', alpha=0.85)
    
    ax.axhline(y=Z_REF, color='gray', linestyle='--', linewidth=2, alpha=0.6, 
               label=f'z_ref = {Z_REF} m')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Normalized Wind Speed (u/u_ref)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Parallel Wind Component: Standard vs All 20 Measured Profiles\n(Normalized to u_ref at {Z_REF} m)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend with two columns to fit all profiles
    ax.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.9)
    ax.set_xlim([0.4, 1.6])
    ax.set_ylim([0, 500])
    
    plt.tight_layout()
    
    # Save figure as PDF
    output_path = RESULTS_DIR / "standard_profile_comparison.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.close()
    
    # Create separate probability distribution plot
    plot_probability_distribution(profile_probs, closest_idx)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
