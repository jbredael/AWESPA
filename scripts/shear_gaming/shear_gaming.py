#!/usr/bin/env python3
"""
Shear Gaming Analysis - AEP Calculation

This script calculates the Annual Energy Production (AEP) for shear gaming analysis:
1. Overall AEP weighted by all 20 profile probabilities
2. Individual AEP for each profile assuming 100% probability

Date: February 2026
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path to import awespa
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from awespa.pipeline.aep import calculate_aep

# Constants
HOURS_PER_YEAR = 8760
RESULTS_DIR = PROJECT_ROOT / "results" / "shear_gaming"
POWER_CURVES_PATH = RESULTS_DIR / "luchsinger_power_curves.yml"
WIND_RESOURCE_PATH = RESULTS_DIR / "wind_resource.yml"


def calculate_individual_profile_aep(
    power_data: Dict[str, Any],
    wind_data: Dict[str, Any],
    profile_id: int
) -> Dict[str, Any]:
    """Calculate AEP for a single profile assuming 100% probability.
    
    Args:
        power_data: Power curve data dictionary
        wind_data: Wind resource data dictionary
        profile_id: ID of the profile to calculate AEP for
        
    Returns:
        Dictionary with AEP results for this profile
    """
    # Get the power curve for this profile
    power_curve = None
    for curve in power_data['power_curves']:
        if curve['profile_id'] == profile_id:
            power_curve = curve
            break
    
    if power_curve is None:
        raise ValueError(f"Profile {profile_id} not found in power curves")
    
    # Get wind speed bins and power values
    wind_speeds = np.array(power_data['reference_wind_speeds_m_s'])
    powers = np.array(power_curve['cycle_power_w'])
    
    # Get wind speed bins from wind resource
    wind_bin_centers = np.array(wind_data['wind_speed_bins']['bin_centers_m_s'])
    
    # Interpolate power to wind resource bins
    powers_interp = np.interp(wind_bin_centers, wind_speeds, powers)
    
    # Get probability matrix and sum across all clusters and directions
    # to get overall wind speed distribution
    probability_matrix = np.array(wind_data['probability_matrix']['data'])
    
    # Sum across clusters (dim 0) and directions (dim 2) to get wind speed probabilities
    wind_speed_probs = np.sum(probability_matrix, axis=(0, 2)) / 100.0  # Convert % to fraction
    
    # Ensure same length
    min_len = min(len(powers_interp), len(wind_speed_probs))
    powers_interp = powers_interp[:min_len]
    wind_speed_probs = wind_speed_probs[:min_len]
    
    # Normalize probabilities to sum to 1.0
    wind_speed_probs = wind_speed_probs / np.sum(wind_speed_probs)
    
    # Calculate expected power
    expected_power = np.sum(powers_interp * wind_speed_probs)
    
    # Calculate AEP
    aep_wh = expected_power * HOURS_PER_YEAR
    
    # Calculate rated power (max power)
    rated_power = np.max(powers)
    
    # Calculate capacity factor
    capacity_factor = expected_power / rated_power if rated_power > 0 else 0
    
    return {
        'profile_id': profile_id,
        'aep_wh': float(aep_wh),
        'aep_kwh': float(aep_wh / 1000),
        'aep_mwh': float(aep_wh / 1e6),
        'aep_gwh': float(aep_wh / 1e9),
        'expected_power_w': float(expected_power),
        'expected_power_kw': float(expected_power / 1000),
        'rated_power_w': float(rated_power),
        'rated_power_kw': float(rated_power / 1000),
        'capacity_factor': float(capacity_factor),
    }


def main():
    """Execute shear gaming AEP analysis."""
    print("=" * 80)
    print("SHEAR GAMING AEP ANALYSIS")
    print("=" * 80)
    
    # Check if input files exist
    if not POWER_CURVES_PATH.exists():
        print(f"ERROR: Power curves file not found: {POWER_CURVES_PATH}")
        return False
    
    if not WIND_RESOURCE_PATH.exists():
        print(f"ERROR: Wind resource file not found: {WIND_RESOURCE_PATH}")
        return False
    
    print(f"\nInput files:")
    print(f"  Power curves: {POWER_CURVES_PATH}")
    print(f"  Wind resource: {WIND_RESOURCE_PATH}")
    
    # =========================================================================
    # SECTION 1: OVERALL AEP WITH WEIGHTED PROBABILITIES
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: OVERALL AEP (All profiles with actual probabilities)")
    print("=" * 80)
    
    print("\nCalculating overall AEP...")
    overall_aep = calculate_aep(
        power_curve_path=POWER_CURVES_PATH,
        wind_resource_path=WIND_RESOURCE_PATH,
        output_path=None,
        plot=False
    )
    
    print("\nOverall AEP Results:")
    print("-" * 80)
    print(f"Total AEP:")
    print(f"  {overall_aep['total_aep']['gwh']:.6f} GWh")
    print(f"  {overall_aep['total_aep']['mwh']:.3f} MWh")
    print(f"  {overall_aep['total_aep']['kwh']:.1f} kWh")
    print(f"\nRated Power: {overall_aep['rated_power_kw']:.2f} kW")
    print(f"Mean Power: {overall_aep['mean_power_kw']:.2f} kW")
    print(f"Capacity Factor: {overall_aep['capacity_factor'] * 100:.2f}%")
    
    # =========================================================================
    # SECTION 2: INDIVIDUAL PROFILE AEP (100% probability each)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: INDIVIDUAL PROFILE AEP (Each profile at 100% probability)")
    print("=" * 80)
    
    # Load data
    with open(POWER_CURVES_PATH, 'r') as f:
        power_data = yaml.safe_load(f)
    
    with open(WIND_RESOURCE_PATH, 'r') as f:
        wind_data = yaml.safe_load(f)
    
    # Get number of profiles
    n_profiles = len(power_data['power_curves'])
    print(f"\nCalculating AEP for {n_profiles} individual profiles...")
    
    # Calculate AEP for each profile
    individual_results = []
    for i in range(1, n_profiles + 1):
        result = calculate_individual_profile_aep(power_data, wind_data, i)
        individual_results.append(result)
        print(f"  Profile {i:2d}: {result['aep_mwh']:8.3f} MWh, "
              f"CF: {result['capacity_factor']*100:5.2f}%, "
              f"Mean Power: {result['expected_power_kw']:6.2f} kW")
    
    # =========================================================================
    # SECTION 3: SUMMARY AND COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: SUMMARY AND COMPARISON")
    print("=" * 80)
    
    # Calculate statistics
    individual_aeps_mwh = [r['aep_mwh'] for r in individual_results]
    individual_cfs = [r['capacity_factor'] * 100 for r in individual_results]
    
    print("\nIndividual Profile Statistics:")
    print("-" * 80)
    print(f"Maximum AEP: {max(individual_aeps_mwh):.3f} MWh (Profile {individual_aeps_mwh.index(max(individual_aeps_mwh)) + 1})")
    print(f"Minimum AEP: {min(individual_aeps_mwh):.3f} MWh (Profile {individual_aeps_mwh.index(min(individual_aeps_mwh)) + 1})")
    print(f"Mean AEP: {np.mean(individual_aeps_mwh):.3f} MWh")
    print(f"Std Dev: {np.std(individual_aeps_mwh):.3f} MWh")
    print(f"\nMaximum CF: {max(individual_cfs):.2f}% (Profile {individual_cfs.index(max(individual_cfs)) + 1})")
    print(f"Minimum CF: {min(individual_cfs):.2f}% (Profile {individual_cfs.index(min(individual_cfs)) + 1})")
    print(f"Mean CF: {np.mean(individual_cfs):.2f}%")
    
    print("\nComparison with Overall AEP:")
    print("-" * 80)
    print(f"Overall AEP (weighted): {overall_aep['total_aep']['mwh']:.3f} MWh")
    print(f"Mean of Individual AEPs: {np.mean(individual_aeps_mwh):.3f} MWh")
    print(f"Difference: {overall_aep['total_aep']['mwh'] - np.mean(individual_aeps_mwh):.3f} MWh")
    print(f"Ratio: {overall_aep['total_aep']['mwh'] / np.mean(individual_aeps_mwh):.3f}")
    
    # =========================================================================
    # SECTION 4: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: SAVING RESULTS")
    print("=" * 80)
    
    # Compile all results
    all_results = {
        'metadata': {
            'description': 'Shear gaming AEP analysis',
            'power_curves_source': str(POWER_CURVES_PATH),
            'wind_resource_source': str(WIND_RESOURCE_PATH),
            'n_profiles': n_profiles,
        },
        'overall_aep': {
            'description': 'AEP calculated with all profiles weighted by actual probabilities',
            'aep_gwh': overall_aep['total_aep']['gwh'],
            'aep_mwh': overall_aep['total_aep']['mwh'],
            'aep_kwh': overall_aep['total_aep']['kwh'],
            'rated_power_kw': overall_aep['rated_power_kw'],
            'mean_power_kw': overall_aep['mean_power_kw'],
            'capacity_factor': overall_aep['capacity_factor'],
        },
        'individual_profile_aeps': {
            'description': 'AEP for each profile assuming 100% probability',
            'profiles': individual_results,
            'statistics': {
                'max_aep_mwh': float(max(individual_aeps_mwh)),
                'min_aep_mwh': float(min(individual_aeps_mwh)),
                'mean_aep_mwh': float(np.mean(individual_aeps_mwh)),
                'std_aep_mwh': float(np.std(individual_aeps_mwh)),
                'max_cf': float(max(individual_cfs) / 100),
                'min_cf': float(min(individual_cfs) / 100),
                'mean_cf': float(np.mean(individual_cfs) / 100),
            }
        }
    }
    
    # Save to YAML
    output_path = RESULTS_DIR / "shear_gaming_aep_results.yml"
    with open(output_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
