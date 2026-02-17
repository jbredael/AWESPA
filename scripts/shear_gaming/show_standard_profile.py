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
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Wind speed profiles - Standard vs all measured
    ax1 = plt.subplot(2, 3, 1)
    
    # Generate standard profile
    v_ref = 10.0  # m/s at reference height
    z_plot = np.linspace(0, 500, 100)
    v_std = v_ref * (z_plot / Z_REF) ** ALPHA_STD
    
    ax1.plot(v_std, z_plot, 'k-', linewidth=3, label=f'Standard (α={ALPHA_STD})', zorder=10)
    
    # Plot all measured profiles
    for i, cluster in enumerate(clusters):
        u_norm = np.array(cluster['u_normalized'])
        v_profile = u_norm * v_ref
        alpha = f'α={alphas[i]:.3f}'
        
        if i == closest_idx:
            ax1.plot(v_profile, altitudes, 'r-', linewidth=2.5, 
                    label=f'Profile {i+1} (closest, {alpha})', zorder=9, alpha=0.9)
        else:
            ax1.plot(v_profile, altitudes, 'b-', linewidth=0.8, alpha=0.3)
    
    ax1.axhline(y=Z_REF, color='gray', linestyle='--', alpha=0.5, label=f'z_ref = {Z_REF}m')
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.set_title('Wind Speed Profiles: Standard vs Measured', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 20])
    
    # Plot 2: Alpha distribution
    ax2 = plt.subplot(2, 3, 2)
    
    bins = np.linspace(0, 0.25, 26)
    ax2.hist(alphas, bins=bins, weights=profile_probs, alpha=0.7, color='skyblue', 
             edgecolor='black', label='Measured (weighted by prob)')
    ax2.axvline(x=ALPHA_STD, color='red', linewidth=2, linestyle='--', 
                label=f'Standard α = {ALPHA_STD}')
    ax2.axvline(x=alphas[closest_idx], color='green', linewidth=2, linestyle='-',
                label=f'Closest (Profile {closest_profile_id}, α={alphas[closest_idx]:.3f})')
    
    ax2.set_xlabel('Shear Exponent (α)', fontsize=11)
    ax2.set_ylabel('Probability', fontsize=11)
    ax2.set_title('Distribution of Shear Exponents', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Alpha vs Profile ID
    ax3 = plt.subplot(2, 3, 3)
    
    profile_ids = np.arange(1, len(alphas) + 1)
    colors = ['green' if i == closest_idx else 'blue' for i in range(len(alphas))]
    sizes = [200 if i == closest_idx else 100 for i in range(len(alphas))]
    
    ax3.scatter(profile_ids, alphas, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    ax3.axhline(y=ALPHA_STD, color='red', linewidth=2, linestyle='--', 
                label=f'Standard α = {ALPHA_STD}')
    
    # Annotate closest profile
    ax3.annotate(f'Profile {closest_profile_id}\nα={alphas[closest_idx]:.4f}',
                xy=(closest_profile_id, alphas[closest_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax3.set_xlabel('Profile ID', fontsize=11)
    ax3.set_ylabel('Shear Exponent (α)', fontsize=11)
    ax3.set_title('Shear Exponent by Profile', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.set_xticks(profile_ids)
    
    # Plot 4: Normalized profiles at reference height
    ax4 = plt.subplot(2, 3, 4)
    
    # For each profile, show u_normalized at z_ref
    z_ref_idx = np.argmin(np.abs(altitudes - Z_REF))
    u_norm_at_zref = [cluster['u_normalized'][z_ref_idx] for cluster in clusters]
    
    ax4.bar(profile_ids, u_norm_at_zref, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=1.0, color='red', linewidth=2, linestyle='--', label='Reference (u_norm = 1.0)')
    ax4.set_xlabel('Profile ID', fontsize=11)
    ax4.set_ylabel('Normalized Wind Speed at z_ref', fontsize=11)
    ax4.set_title(f'Normalized Wind Speed at Reference Height ({Z_REF}m)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    ax4.set_xticks(profile_ids)
    
    # Plot 5: Probability distribution
    ax5 = plt.subplot(2, 3, 5)
    
    bars = ax5.bar(profile_ids, profile_probs, color=colors, alpha=0.7, edgecolor='black')
    # Highlight closest profile
    bars[closest_idx].set_color('green')
    bars[closest_idx].set_alpha(0.9)
    
    ax5.set_xlabel('Profile ID', fontsize=11)
    ax5.set_ylabel('Probability of Occurrence', fontsize=11)
    ax5.set_title('Profile Probability Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticks(profile_ids)
    
    # Plot 6: Comparison table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Standard α', f'{ALPHA_STD:.4f}'],
        ['', ''],
        ['Closest Profile:', f'Profile {closest_profile_id}'],
        ['  Measured α', f'{alphas[closest_idx]:.4f}'],
        ['  Difference from std', f'{alpha_diffs[closest_idx]:.4f}'],
        ['  Probability', f'{profile_probs[closest_idx]:.4f}'],
        ['', ''],
        ['All Profiles:', ''],
        ['  Mean α', f'{np.mean(alphas):.4f}'],
        ['  Std Dev α', f'{np.std(alphas):.4f}'],
        ['  Min α', f'{np.min(alphas):.4f} (Prof {alphas.argmin()+1})'],
        ['  Max α', f'{np.max(alphas):.4f} (Prof {alphas.argmax()+1})'],
        ['', ''],
        ['  Weighted mean α', f'{np.sum(alphas * profile_probs):.4f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight closest profile section
    for i in range(4, 8):
        table[(i, 0)].set_facecolor('#ffffcc')
        table[(i, 1)].set_facecolor('#ffffcc')
    
    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Standard Wind Shear Profile Analysis (α_std = {ALPHA_STD})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = RESULTS_DIR / "standard_profile_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
