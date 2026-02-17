#!/usr/bin/env python3
"""
Shear Normalization for AWE Power Curves

This script applies shear normalization to precomputed AWE power curves
corresponding to different wind profiles to assess the impact of wind shear
variation on power production.

The normalization corrects power curves from measured shear profiles to a
standard shear profile using the cubic relationship between wind speed and power.

Date: February 2026
"""

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, List
from scipy import stats

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Standard shear exponent for normalization
ALPHA_STD = 0.2

# Reference height (m)
Z_REF = 200.0

# Reel-out altitude (m) - effective wind speed altitude
Z_RO = 400.0

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "shear_gaming"
POWER_CURVES_PATH = RESULTS_DIR / "luchsinger_power_curves.yml"
WIND_RESOURCE_PATH = RESULTS_DIR / "wind_resource.yml"


# =============================================================================
# STEP 1: DATA LOADING AND PROCESSING
# =============================================================================

def load_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load wind resource and power curve data from YAML files.
    
    Returns:
        Tuple containing power data and wind data dictionaries.
    """
    print("Loading data...")
    
    with open(POWER_CURVES_PATH, 'r') as f:
        power_data = yaml.safe_load(f)
    
    with open(WIND_RESOURCE_PATH, 'r') as f:
        wind_data = yaml.safe_load(f)
    
    print(f"  Loaded {len(power_data['power_curves'])} power curves")
    print(f"  Loaded {len(wind_data['clusters'])} wind profiles")
    
    return power_data, wind_data


def fit_power_law_shear(altitudes: np.ndarray, wind_speeds: np.ndarray, 
                         z_ref: float) -> Tuple[float, float]:
    """Fit power law to wind profile to determine shear exponent.
    
    Uses log-log linear regression:
    log(V) = log(V_ref) + alpha * log(z/z_ref)
    
    Args:
        altitudes: Array of altitudes (m).
        wind_speeds: Array of wind speeds at each altitude (m/s).
        z_ref: Reference height (m).
        
    Returns:
        Tuple of (alpha, V_ref) where alpha is shear exponent and V_ref is
        reference wind speed at z_ref.
    """
    # Remove zero or negative values
    valid_mask = (altitudes > 0) & (wind_speeds > 0)
    z_valid = altitudes[valid_mask]
    v_valid = wind_speeds[valid_mask]
    
    # Transform to log space
    log_z_ratio = np.log(z_valid / z_ref)
    log_v = np.log(v_valid)
    
    # Linear regression in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_z_ratio, log_v)
    
    alpha = slope
    v_ref = np.exp(intercept)
    
    return alpha, v_ref


def calculate_profile_probabilities(wind_data: Dict[str, Any]) -> np.ndarray:
    """Calculate total probability of occurrence for each cluster/profile.
    
    Sums probabilities across all wind speeds and directions.
    
    Args:
        wind_data: Wind resource data dictionary.
        
    Returns:
        Array of probabilities for each cluster.
    """
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Sum across wind speeds (dim 1) and directions (dim 2)
    profile_probs = np.sum(probability_matrix, axis=(1, 2)) / 100.0  # Convert % to fraction
    
    return profile_probs


def extract_profile_data(wind_data: Dict[str, Any], 
                         power_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and process all profile data.
    
    For each profile, extracts wind speeds, fits power law, and prepares
    power curve data.
    
    Args:
        wind_data: Wind resource data dictionary.
        power_data: Power curve data dictionary.
        
    Returns:
        List of dictionaries containing processed profile data.
    """
    print("\nProcessing wind profiles...")
    
    altitudes = np.array(wind_data['altitudes'])
    clusters = wind_data['clusters']
    power_curves = power_data['power_curves']
    profile_probs = calculate_profile_probabilities(wind_data)
    
    profiles = []
    
    for i, (cluster, power_curve) in enumerate(zip(clusters, power_curves)):
        cluster_id = cluster['id']
        
        # Extract normalized wind speeds and denormalize
        u_norm = np.array(cluster['u_normalized'])
        # Assume reference wind speed at z_ref for denormalization
        # We'll use the power curve reference speeds
        
        # Get absolute wind speeds from power curve reference
        ref_wind_speeds = np.array(power_data['reference_wind_speeds_m_s'])
        
        # For each altitude, we need to reconstruct the wind speed profile
        # The u_normalized gives the shape, we need to scale it appropriately
        # At z_ref (index where altitude == Z_REF), u_normalized should be 1.0
        
        # Find index closest to Z_REF
        z_ref_idx = np.argmin(np.abs(altitudes - Z_REF))
        
        # Calculate wind speeds at all altitudes for a reference condition
        # We'll use the middle wind speed bin as reference
        v_ref_sample = ref_wind_speeds[len(ref_wind_speeds)//2]
        wind_speeds_profile = u_norm * v_ref_sample
        
        # Fit power law to determine alpha_meas
        alpha_meas, v_ref_fitted = fit_power_law_shear(altitudes, wind_speeds_profile, Z_REF)
        
        # Extract power curve
        powers = np.array(power_curve['cycle_power_w'])
        
        profiles.append({
            'id': cluster_id,
            'altitudes': altitudes,
            'u_normalized': u_norm,
            'wind_speeds_profile': wind_speeds_profile,
            'alpha_meas': alpha_meas,
            'v_ref_fitted': v_ref_fitted,
            'ref_wind_speeds': ref_wind_speeds,
            'powers': powers,
            'probability': profile_probs[i],
        })
        
        print(f"  Profile {cluster_id:2d}: alpha = {alpha_meas:.4f}, prob = {profile_probs[i]:.4f}")
    
    return profiles


# =============================================================================
# STEP 2: SHEAR NORMALIZATION
# =============================================================================

def apply_shear_normalization(profile: Dict[str, Any], 
                               alpha_std: float,
                               z_ref: float,
                               z_ro: float) -> Dict[str, Any]:
    """Apply shear normalization to a power curve.
    
    Corrects power from measured shear to standard shear using:
    P_corr = P_meas * (V_eff_std / V_eff_meas)^3
    
    where:
    V_eff_meas = V_ref * (z_ro / z_ref)^alpha_meas
    V_eff_std = V_ref * (z_ro / z_ref)^alpha_std
    
    Args:
        profile: Profile data dictionary.
        alpha_std: Standard shear exponent.
        z_ref: Reference height (m).
        z_ro: Reel-out height (m).
        
    Returns:
        Dictionary with normalized power curve and correction factors.
    """
    alpha_meas = profile['alpha_meas']
    ref_wind_speeds = profile['ref_wind_speeds']
    powers_meas = profile['powers']
    
    # Calculate shear factors
    shear_factor_meas = (z_ro / z_ref) ** alpha_meas
    shear_factor_std = (z_ro / z_ref) ** alpha_std
    
    # For each reference wind speed, calculate effective wind speeds
    v_eff_meas = ref_wind_speeds * shear_factor_meas
    v_eff_std = ref_wind_speeds * shear_factor_std
    
    # Apply cubic correction (power ~ V^3)
    correction_factor = (v_eff_std / v_eff_meas) ** 3
    powers_corr = powers_meas * correction_factor
    
    # Apply rated power clipping if needed
    # Find the maximum power in the original curve (rated power)
    rated_power = np.max(powers_meas)
    powers_corr_clipped = np.minimum(powers_corr, rated_power)
    
    return {
        'powers_normalized': powers_corr_clipped,
        'correction_factor': correction_factor,
        'mean_correction': np.mean(correction_factor),
        'v_eff_meas': v_eff_meas,
        'v_eff_std': v_eff_std,
    }


def normalize_all_profiles(profiles: List[Dict[str, Any]],
                           alpha_std: float,
                           z_ref: float,
                           z_ro: float) -> List[Dict[str, Any]]:
    """Apply shear normalization to all profiles.
    
    Args:
        profiles: List of profile data dictionaries.
        alpha_std: Standard shear exponent.
        z_ref: Reference height (m).
        z_ro: Reel-out height (m).
        
    Returns:
        List of profiles with normalization results added.
    """
    print("\nApplying shear normalization...")
    print(f"  Standard shear exponent: {alpha_std}")
    print(f"  Reference height: {z_ref} m")
    print(f"  Reel-out height: {z_ro} m")
    
    for profile in profiles:
        norm_result = apply_shear_normalization(profile, alpha_std, z_ref, z_ro)
        profile.update(norm_result)
        
        print(f"  Profile {profile['id']:2d}: "
              f"alpha_meas = {profile['alpha_meas']:.4f}, "
              f"mean correction = {norm_result['mean_correction']:.4f}")
    
    return profiles


# =============================================================================
# STEP 3: PLOTTING
# =============================================================================

def plot_normalized_curves(profiles: List[Dict[str, Any]], 
                           output_path: Path = None) -> None:
    """Create subplot grid showing original and normalized power curves.
    
    Creates a 4x5 grid (20 subplots) with each profile's original and
    shear-normalized power curves.
    
    Args:
        profiles: List of profile data with normalization results.
        output_path: Optional path to save figure.
    """
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, (ax, profile) in enumerate(zip(axes, profiles)):
        ref_speeds = profile['ref_wind_speeds']
        powers_orig = profile['powers'] / 1000  # Convert to kW
        powers_norm = profile['powers_normalized'] / 1000  # Convert to kW
        
        # Plot curves
        ax.plot(ref_speeds, powers_orig, 'b-', linewidth=2, 
                label='Original', alpha=0.7)
        ax.plot(ref_speeds, powers_norm, 'r--', linewidth=2,
                label='Normalized', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Reference Wind Speed (m/s)', fontsize=8)
        ax.set_ylabel('Power (kW)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
        
        # Title with profile info
        title = (f'Profile {profile["id"]} | '
                f'α = {profile["alpha_meas"]:.3f} | '
                f'p = {profile["probability"]:.3f}')
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Set consistent axis limits for comparison
        ax.set_xlim([4, 25])
        ax.set_ylim([0, 130])
        
        ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Shear Normalization: Original vs Normalized Power Curves\n'
                 f'Standard α = {ALPHA_STD}, z_ref = {Z_REF} m, z_ro = {Z_RO} m',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# STEP 4: SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(profiles: List[Dict[str, Any]]) -> None:
    """Print summary statistics of shear coefficients and corrections.
    
    Args:
        profiles: List of profile data with normalization results.
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    alphas = np.array([p['alpha_meas'] for p in profiles])
    corrections = np.array([p['mean_correction'] for p in profiles])
    probabilities = np.array([p['probability'] for p in profiles])
    
    print("\nShear Exponents (α_meas):")
    print("-" * 80)
    print(f"  Mean:           {np.mean(alphas):.4f}")
    print(f"  Std Dev:        {np.std(alphas):.4f}")
    print(f"  Minimum:        {np.min(alphas):.4f} (Profile {alphas.argmin() + 1})")
    print(f"  Maximum:        {np.max(alphas):.4f} (Profile {alphas.argmax() + 1})")
    print(f"  Standard α:     {ALPHA_STD:.4f}")
    
    print("\nPower Correction Factors (mean across wind speeds):")
    print("-" * 80)
    print(f"  Mean:           {np.mean(corrections):.4f}")
    print(f"  Std Dev:        {np.std(corrections):.4f}")
    print(f"  Minimum:        {np.min(corrections):.4f} (Profile {corrections.argmin() + 1})")
    print(f"  Maximum:        {np.max(corrections):.4f} (Profile {corrections.argmax() + 1})")
    
    print("\nRelative Correction Magnitude:")
    print("-" * 80)
    rel_corrections = np.abs(corrections - 1.0) * 100  # Deviation from 1.0 in %
    print(f"  Mean abs deviation: {np.mean(rel_corrections):.2f}%")
    print(f"  Max abs deviation:  {np.max(rel_corrections):.2f}%")
    
    print("\nProfile Probabilities:")
    print("-" * 80)
    print(f"  Total probability:  {np.sum(probabilities):.4f}")
    print(f"  Mean probability:   {np.mean(probabilities):.4f}")
    print(f"  Min probability:    {np.min(probabilities):.4f} (Profile {probabilities.argmin() + 1})")
    print(f"  Max probability:    {np.max(probabilities):.4f} (Profile {probabilities.argmax() + 1})")
    
    print("\nWeighted Statistics (by probability):")
    print("-" * 80)
    weighted_alpha = np.sum(alphas * probabilities) / np.sum(probabilities)
    weighted_correction = np.sum(corrections * probabilities) / np.sum(probabilities)
    print(f"  Weighted mean α:          {weighted_alpha:.4f}")
    print(f"  Weighted mean correction: {weighted_correction:.4f}")


def save_normalized_data(profiles: List[Dict[str, Any]], 
                         output_path: Path) -> None:
    """Save normalized power curves to YAML file.
    
    Args:
        profiles: List of profile data with normalization results.
        output_path: Path to save YAML file.
    """
    print("\nSaving normalized data...")
    
    output_data = {
        'metadata': {
            'description': 'Shear-normalized power curves',
            'standard_alpha': ALPHA_STD,
            'reference_height_m': Z_REF,
            'reel_out_height_m': Z_RO,
            'n_profiles': len(profiles),
        },
        'profiles': []
    }
    
    for profile in profiles:
        output_data['profiles'].append({
            'profile_id': int(profile['id']),
            'alpha_measured': float(profile['alpha_meas']),
            'probability': float(profile['probability']),
            'mean_correction_factor': float(profile['mean_correction']),
            'reference_wind_speeds_m_s': profile['ref_wind_speeds'].tolist(),
            'power_original_w': profile['powers'].tolist(),
            'power_normalized_w': profile['powers_normalized'].tolist(),
        })
    
    with open(output_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Saved to: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute shear normalization analysis."""
    print("=" * 80)
    print("SHEAR NORMALIZATION FOR AWE POWER CURVES")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Standard shear exponent (α_std): {ALPHA_STD}")
    print(f"  Reference height (z_ref):         {Z_REF} m")
    print(f"  Reel-out height (z_ro):           {Z_RO} m")
    
    # Check if input files exist
    if not POWER_CURVES_PATH.exists():
        print(f"\nERROR: Power curves file not found: {POWER_CURVES_PATH}")
        return False
    
    if not WIND_RESOURCE_PATH.exists():
        print(f"\nERROR: Wind resource file not found: {WIND_RESOURCE_PATH}")
        return False
    
    # Step 1: Load and process data
    power_data, wind_data = load_data()
    profiles = extract_profile_data(wind_data, power_data)
    
    # Step 2: Apply shear normalization
    profiles = normalize_all_profiles(profiles, ALPHA_STD, Z_REF, Z_RO)
    
    # Step 3: Create plots
    plot_output_path = RESULTS_DIR / "shear_normalization_comparison.png"
    plot_normalized_curves(profiles, plot_output_path)
    
    # Step 4: Print summary statistics
    print_summary_statistics(profiles)
    
    # Save normalized data
    data_output_path = RESULTS_DIR / "shear_normalized_power_curves.yml"
    save_normalized_data(profiles, data_output_path)
    
    print("\n" + "=" * 80)
    print("SHEAR NORMALIZATION COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
