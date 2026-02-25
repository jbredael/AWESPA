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

# Use only reel-out phase for effective wind speed calculation
USE_REEL_OUT_ONLY = True

# Standard shear exponent for normalization
ALPHA_STD = 0.2

# Reference height (m)
Z_REF = 200.0

# Exponent for effective wind speed calculation
VeffEXP = 3

# Rayleigh distribution parameters for AEP calculation
MEAN_WIND_SPEED_M_S = 10.0  # Mean wind speed at reference height

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "shear_gaming"
POWER_CURVES_PATH = RESULTS_DIR / "power_curves_direct_simulation.yml"
WIND_RESOURCE_PATH = RESULTS_DIR / "wind_resource.yml"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_power_curves(filepath: Path) -> Dict[str, Any]:
    """Load power curves from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def interpolate_wind_speed(altitude_m: float, altitudes: np.ndarray, 
                          wind_speeds: np.ndarray) -> float:
    """Interpolate wind speed at given altitude."""
    return np.interp(altitude_m, altitudes, wind_speeds)

def compute_effective_wind_speed(time_s: np.ndarray,
                                altitude_history_m: np.ndarray, 
                                altitudes_m: np.ndarray,
                                wind_speeds_m_s: np.ndarray,
                                reel_speed_m_s: np.ndarray = None,
                                use_reel_out_only: bool = False) -> float:
    """Compute effective wind speed using cubic time-averaged formula.
    
    V_eff = (1/T * integral(V(z(t))^3 dt))^(1/3)
    
    Args:
        time_s: Time points
        altitude_history_m: Time series of altitude values
        altitudes_m: Altitude grid for wind profile
        wind_speeds_m_s: Wind speed values at each altitude
        reel_speed_m_s: Reel speed time series (optional, for filtering)
        use_reel_out_only: If True, only use reel-out phase (reel_speed > 0)
        
    Returns:
        Effective wind speed in m/s
    """
    # Filter for reel-out phase if requested
    if use_reel_out_only and reel_speed_m_s is not None:
        mask = reel_speed_m_s > 0
        if not np.any(mask):
            # No reel-out phase, return zero
            return 0.0
        time_filtered = time_s[mask]
        altitude_filtered = altitude_history_m[mask]
    else:
        time_filtered = time_s
        altitude_filtered = altitude_history_m
    
    # Interpolate wind speed at each altitude point in the history
    wind_at_altitude = np.array([
        interpolate_wind_speed(z, altitudes_m, wind_speeds_m_s)
        for z in altitude_filtered
    ])
    
    # Compute integral of V^3 over time using trapezoidal rule
    v_cubed = wind_at_altitude**VeffEXP
    integral_v_cubed = np.trapz(v_cubed, time_filtered)
    
    # Divide by total time T
    total_time = time_filtered[-1] - time_filtered[0]
    if total_time == 0:
        return 0.0
    v_cubed_mean = integral_v_cubed / total_time
    
    # Return cube root
    return v_cubed_mean**(1/VeffEXP)

def create_standard_profile(v_ref_m_s: float, altitudes_m: np.ndarray,
                           alpha_std: float, z_ref: float) -> np.ndarray:
    """Create standardized wind profile with fixed shear exponent.
    
    Args:
        v_ref_m_s: Reference wind speed at reference height
        altitudes_m: Altitude grid
        alpha_std: Standard shear exponent
        z_ref: Reference height in m
        
    Returns:
        Wind speeds at each altitude
    """
    return v_ref_m_s * (altitudes_m / z_ref)**alpha_std

def normalize_power(power_meas_w: float, v_eff_meas: float, 
                   v_eff_std: float) -> float:
    """Apply energy-based power correction.
    
    P_corr = P_meas * (V_eff_std / V_eff_meas)^3
    
    Args:
        power_meas_w: Measured power in W
        v_eff_meas: Effective wind speed under measured conditions
        v_eff_std: Effective wind speed under standard conditions
        
    Returns:
        Corrected power in W
    """
    if v_eff_meas == 0:
        return 0.0
    return power_meas_w * (v_eff_std / v_eff_meas)**3

def rayleigh_pdf(v: float, v_mean: float) -> float:
    """Rayleigh probability density function.
    
    Args:
        v: Wind speed
        v_mean: Mean wind speed
        
    Returns:
        Probability density at wind speed v
    """
    # Scale parameter c from mean wind speed
    c = v_mean / np.sqrt(np.pi / 2)
    
    if v < 0:
        return 0.0
    
    return (v / c**2) * np.exp(-v**2 / (2 * c**2))

def calculate_aep(wind_speeds: np.ndarray, powers: np.ndarray, 
                 v_mean: float) -> float:
    """Calculate Annual Energy Production using Rayleigh distribution.
    
    Args:
        wind_speeds: Array of wind speeds (m/s)
        powers: Array of power values (W)
        v_mean: Mean wind speed for Rayleigh distribution
        
    Returns:
        AEP in MWh
    """
    # Create fine wind speed grid for integration
    v_min = 0.0
    v_max = max(wind_speeds) + 5.0
    v_grid = np.linspace(v_min, v_max, 1000)
    
    # Interpolate power curve to fine grid (extrapolate with boundary values)
    power_grid = np.interp(v_grid, wind_speeds, powers, left=0.0, right=powers[-1])
    
    # Compute Rayleigh PDF at each point
    pdf_grid = np.array([rayleigh_pdf(v, v_mean) for v in v_grid])
    
    # Integrate: AEP = integral(P(v) * f(v) dv) * hours_per_year
    hours_per_year = 8760
    average_power_w = np.trapz(power_grid * pdf_grid, v_grid)
    aep_mwh = (average_power_w * hours_per_year) / 1e6
    
    return aep_mwh

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_shear_normalization():
    """Apply shear normalization to all power curves."""
    
    print("Loading power curves...")
    data = load_power_curves(POWER_CURVES_PATH)
    
    altitudes_m = np.array(data['altitudes_m'])
    n_profiles = len(data['power_curves'])
    
    print(f"Processing {n_profiles} wind profiles...")
    
    # Storage for plotting
    all_original_curves = []
    all_corrected_curves = []
    wind_speeds_grid = []
    
    # Process each profile
    for profile in data['power_curves']:
        profile_id = profile['profile_id']
        print(f"  Profile {profile_id}...")
        
        # Get normalized wind profile
        u_norm = np.array(profile['wind_profile']['u_normalized'])
        
        # Storage for this profile's power curve
        original_powers = []
        corrected_powers = []
        ref_wind_speeds = []
        
        # Process each wind speed point (only successful ones with time history)
        for wind_point in profile['wind_speed_data']:
            # Skip unsuccessful simulations
            if not wind_point.get('success', False) or 'time_history' not in wind_point:
                continue
            
            # Get measured reel-out power from performance metrics
            power_meas = wind_point['performance']['power']['average_reel_out_power_w']
            
            # Skip negative or zero power values
            if power_meas <= 0:
                continue
                
            v_ref = wind_point['wind_speed_m_s']
            ref_wind_speeds.append(v_ref)
            
            # Get actual wind profile: V(z) = V_ref * u_normalized(z)
            wind_profile_meas = v_ref * u_norm
            
            # Get time history data
            time_hist = wind_point['time_history']
            time_s = np.array(time_hist['time_s'])
            altitude_history = np.array(time_hist['altitude_m'])
            reel_speed = np.array(time_hist['reel_speed_m_s'])
            
            original_powers.append(power_meas)
            
            # Step 3: Compute effective wind speed under measured conditions
            v_eff_meas = compute_effective_wind_speed(
                time_s, altitude_history, altitudes_m, wind_profile_meas,
                reel_speed, USE_REEL_OUT_ONLY
            )
            
            # Step 4: Create standardized profile and compute effective wind speed
            wind_profile_std = create_standard_profile(
                v_ref, altitudes_m, ALPHA_STD, Z_REF
            )
            v_eff_std = compute_effective_wind_speed(
                time_s, altitude_history, altitudes_m, wind_profile_std,
                reel_speed, USE_REEL_OUT_ONLY
            )
            
            # Step 5: Apply power correction
            power_corr = normalize_power(power_meas, v_eff_meas, v_eff_std)
            corrected_powers.append(power_corr)
        
        all_original_curves.append((ref_wind_speeds, original_powers, profile_id))
        all_corrected_curves.append((ref_wind_speeds, corrected_powers, profile_id))
    
    print("\nCalculating AEP for each profile...")
    aep_original = []
    aep_corrected = []
    
    for i in range(n_profiles):
        ws, power_orig, pid = all_original_curves[i]
        _, power_corr, _ = all_corrected_curves[i]
        
        aep_orig = calculate_aep(np.array(ws), np.array(power_orig), MEAN_WIND_SPEED_M_S)
        aep_corr = calculate_aep(np.array(ws), np.array(power_corr), MEAN_WIND_SPEED_M_S)
        
        aep_original.append(aep_orig)
        aep_corrected.append(aep_corr)
        
        print(f"  Profile {pid}: Original AEP = {aep_orig:.2f} MWh, Corrected AEP = {aep_corr:.2f} MWh")
    
    # Performance metrics
    print("\n" + "="*70)
    print("SHEAR NORMALIZATION PERFORMANCE METRICS")
    print("="*70)
    
    aep_orig_array = np.array(aep_original)
    aep_corr_array = np.array(aep_corrected)
    
    avg_original = np.mean(aep_orig_array)
    avg_corrected = np.mean(aep_corr_array)
    
    std_original = np.std(aep_orig_array)
    std_corrected = np.std(aep_corr_array)
    
    cv_original = (std_original / avg_original) * 100  # Coefficient of variation in %
    cv_corrected = (std_corrected / avg_corrected) * 100
    
    mad_original = np.mean(np.abs(aep_orig_array - avg_original))  # Mean absolute deviation
    mad_corrected = np.mean(np.abs(aep_corr_array - avg_corrected))
    
    reduction_std = ((std_original - std_corrected) / std_original) * 100
    reduction_cv = ((cv_original - cv_corrected) / cv_original) * 100
    reduction_mad = ((mad_original - mad_corrected) / mad_original) * 100
    
    print(f"\nOriginal AEP Statistics:")
    print(f"  Average AEP:              {avg_original:.2f} MWh")
    print(f"  Standard Deviation:       {std_original:.2f} MWh")
    print(f"  Coefficient of Variation: {cv_original:.2f} %")
    print(f"  Mean Absolute Deviation:  {mad_original:.2f} MWh")
    
    print(f"\nCorrected AEP Statistics:")
    print(f"  Average AEP:              {avg_corrected:.2f} MWh")
    print(f"  Standard Deviation:       {std_corrected:.2f} MWh")
    print(f"  Coefficient of Variation: {cv_corrected:.2f} %")
    print(f"  Mean Absolute Deviation:  {mad_corrected:.2f} MWh")
    
    print(f"\nReduction in Spread (Performance Improvement):")
    print(f"  Standard Deviation:       {reduction_std:+.2f} %")
    print(f"  Coefficient of Variation: {reduction_cv:+.2f} %")
    print(f"  Mean Absolute Deviation:  {reduction_mad:+.2f} %")
    
    if reduction_std > 0:
        print(f"\n✓ Shear normalization REDUCED AEP spread by {reduction_std:.1f}%")
    else:
        print(f"\n✗ Shear normalization INCREASED AEP spread by {abs(reduction_std):.1f}%")
    
    print("="*70 + "\n")
    
    print("\nGenerating comparison plots...")
    plot_power_curve_comparison(all_original_curves, all_corrected_curves)
    plot_all_curves_overlaid(all_original_curves, all_corrected_curves)
    plot_aep_comparison(aep_original, aep_corrected, n_profiles)
    
    print("Done!")

def plot_power_curve_comparison(original_curves: List, corrected_curves: List):
    """Plot all power curves: original vs corrected for each profile.
    
    Args:
        original_curves: List of (wind_speeds, powers, profile_id) tuples
        corrected_curves: List of (wind_speeds, powers, profile_id) tuples
    """
    n_profiles = len(original_curves)
    
    # Create subplot grid (4 rows x 5 columns for 20 profiles)
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot each profile comparison
    for i in range(n_profiles):
        ws_orig, power_orig, pid = original_curves[i]
        ws_corr, power_corr, _ = corrected_curves[i]
        
        ax = axes[i]
        
        # Plot original and corrected curves
        ax.plot(ws_orig, np.array(power_orig)/1000, 'o-', 
                label='Original', linewidth=2, markersize=4, color='C0')
        ax.plot(ws_corr, np.array(power_corr)/1000, 's-', 
                label='Corrected', linewidth=2, markersize=4, color='C1')
        
        ax.set_title(f'Profile {pid}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Only add labels on edge plots
        if i >= 15:  # Bottom row
            ax.set_xlabel('Wind Speed [m/s]', fontsize=9)
        if i % 5 == 0:  # Left column
            ax.set_ylabel('Power [kW]', fontsize=9)
    
    fig.suptitle(f'Shear Normalization Comparison - Reel-Out Power Only (α_std = {ALPHA_STD})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    output_path = RESULTS_DIR / "shear_normalization_comparison_reel_out_only.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()

def plot_all_curves_overlaid(original_curves: List, corrected_curves: List):
    """Plot all power curves overlaid: original (left) vs corrected (right).
    
    Args:
        original_curves: List of (wind_speeds, powers, profile_id) tuples
        corrected_curves: List of (wind_speeds, powers, profile_id) tuples
    """
    n_profiles = len(original_curves)
    colors = plt.cm.tab20(np.linspace(0, 1, n_profiles))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot all original curves
    for i, (ws, power, pid) in enumerate(original_curves):
        ax1.plot(ws, np.array(power)/1000, color=colors[i], 
                label=f'Profile {pid}', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Reference Wind Speed at 200 m [m/s]', fontsize=12)
    ax1.set_ylabel('Average Reel-Out Power [kW]', fontsize=12)
    ax1.set_title('Original Power Curves (Reel-Out Only)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # Plot all corrected curves
    for i, (ws, power, pid) in enumerate(corrected_curves):
        ax2.plot(ws, np.array(power)/1000, color=colors[i], 
                label=f'Profile {pid}', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Reference Wind Speed at 200 m [m/s]', fontsize=12)
    ax2.set_ylabel('Average Reel-Out Power [kW]', fontsize=12)
    ax2.set_title(f'Shear-Normalized Power Curves - Reel-Out Only (α_std = {ALPHA_STD})', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    fig.suptitle('All Power Curves Comparison - Reel-Out Power Only', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    output_path = RESULTS_DIR / "all_curves_overlaid_reel_out_only.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Overlaid curves plot saved to: {output_path}")
    
    plt.close()

def plot_aep_comparison(aep_original: List[float], aep_corrected: List[float], 
                       n_profiles: int):
    """Plot AEP comparison for all profiles.
    
    Args:
        aep_original: List of original AEP values in MWh
        aep_corrected: List of corrected AEP values in MWh
        n_profiles: Number of profiles
    """
    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    x = np.arange(n_profiles)
    width = 0.35
    
    # Convert to numpy arrays for calculations
    aep_orig_array = np.array(aep_original)
    aep_corr_array = np.array(aep_corrected)
    
    # Calculate averages
    avg_original = np.mean(aep_orig_array)
    avg_corrected = np.mean(aep_corr_array)
    
    # Plot bars on primary axis
    bars1 = ax1.bar(x - width/2, aep_original, width, label='Original', 
                   color='C0', alpha=0.8)
    bars2 = ax1.bar(x + width/2, aep_corrected, width, label='Corrected', 
                   color='C1', alpha=0.8)
    
    # Add average lines
    ax1.axhline(y=avg_original, color='C0', linestyle='--', linewidth=2, 
               label=f'Original Avg = {avg_original:.2f} MWh', alpha=0.8)
    ax1.axhline(y=avg_corrected, color='C1', linestyle='--', linewidth=2, 
               label=f'Corrected Avg = {avg_corrected:.2f} MWh', alpha=0.8)
    
    ax1.set_xlabel('Wind Profile', fontsize=12)
    ax1.set_ylabel('Annual Energy Production [MWh]', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in range(n_profiles)])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Create secondary y-axis for differences
    ax2 = ax1.twinx()
    
    # Calculate differences from average
    diff_original = aep_orig_array - avg_original
    diff_corrected = aep_corr_array - avg_corrected
    
    # Plot differences as lines on secondary axis
    ax2.plot(x, diff_original, 'o--', color='C0', linewidth=1.5, markersize=6,
            label='Original - Avg', alpha=0.6)
    ax2.plot(x, diff_corrected, 's--', color='C1', linewidth=1.5, markersize=6,
            label='Corrected - Avg', alpha=0.6)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_ylabel('Difference from Average [MWh]', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    ax1.set_title(f'AEP Comparison: Original vs Shear-Normalized Power Curves (Reel-Out Only)\n'
                 f'(Rayleigh distribution, mean wind speed = {MEAN_WIND_SPEED_M_S} m/s at {Z_REF:.0f} m)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = RESULTS_DIR / "aep_comparison_reel_out_only.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"AEP plot saved to: {output_path}")
    
    plt.close()

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    process_shear_normalization()


