#!/usr/bin/env python3
"""
Power Curve Comparison Script

Loads two power curve result files (awesIO YAML format) and plots their
average cycle power curves side by side. If either file contains multiple
wind profiles, a grid of subplots is produced with one panel per profile.

Usage:
    python scripts/compare_power_models.py [file1.yml] [file2.yml]

    If no arguments are provided the two default paths defined below are used.

Author: AWESPA Development Team
Date: February 2026
"""

import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Default file paths – edit here or pass as command-line arguments
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_FILE_1 = PROJECT_ROOT / "results" / "luchsinger_power_curves.yml"
DEFAULT_FILE_2 = PROJECT_ROOT / "results" / "power_curves_direct_simulation.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_power_curve_file(file_path):
    """Load a power curve YAML file into a dictionary.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Power curve file not found: {file_path}")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def extract_power_curve(profile):
    """Extract wind speeds and average cycle powers from a profile entry.

    Only entries where ``success`` is ``True`` are included.

    Args:
        profile (dict): A single element from the ``power_curves`` list.

    Returns:
        tuple[list[float], list[float]]: Wind speeds (m/s) and corresponding
            average cycle powers (W).
    """
    windSpeeds = []
    powers = []
    for entry in profile.get("wind_speed_data", []):
        if entry.get("success", False):
            windSpeeds.append(entry["wind_speed_m_s"])
            powers.append(entry["performance"]["power"]["average_cycle_power_w"])
    return windSpeeds, powers


def get_file_label(data, file_path):
    """Return a short human-readable label for a power curve file.

    Args:
        data (dict): Parsed YAML content.
        file_path (Path): Path used as fallback label.

    Returns:
        str: Label string.
    """
    try:
        return data["metadata"]["name"]
    except (KeyError, TypeError):
        return Path(file_path).stem


def build_profile_index(data):
    """Build a mapping from profile_id to profile dict.

    Args:
        data (dict): Parsed YAML content.

    Returns:
        dict[int, dict]: Profile index keyed by profile_id.
    """
    return {p["profile_id"]: p for p in data.get("power_curves", [])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single(ax, data1, data2, profileId, label1, label2):
    """Plot the power curves for a single profile on a given Axes.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        data1 (dict): Parsed YAML content of file 1.
        data2 (dict): Parsed YAML content of file 2.
        profileId (int): The profile ID to plot.
        label1 (str): Legend label for file 1.
        label2 (str): Legend label for file 2.
    """
    index1 = build_profile_index(data1)
    index2 = build_profile_index(data2)

    if profileId in index1:
        ws1, pw1 = extract_power_curve(index1[profileId])
        ax.plot(
            ws1,
            np.array(pw1) / 1000,
            linewidth=2,
            label=label1,
        )
    else:
        ax.annotate(
            f"{label1}\n(no data)",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
        )

    if profileId in index2:
        ws2, pw2 = extract_power_curve(index2[profileId])
        ax.plot(
            ws2,
            np.array(pw2) / 1000,
            linewidth=2,
            linestyle="--",
            label=label2,
        )
    else:
        ax.annotate(
            f"{label2}\n(no data)",
            xy=(0.5, 0.4),
            xycoords="axes fraction",
            ha="center",
        )

    ax.set_title(f"Profile {profileId}", fontsize=11)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def compare_power_curves(filePath1, filePath2):
    """Load two power curve files and produce a comparison plot.

    If both files contain only a single profile, a single panel is created.
    If either file contains multiple profiles, a grid of subplots is produced
    with one panel per unique profile ID found across both files.

    Args:
        filePath1 (Path | str): Path to the first power curve YAML file.
        filePath2 (Path | str): Path to the second power curve YAML file.
    """
    print("AWESPA Power Curve Comparison")
    print("=" * 50)

    data1 = load_power_curve_file(filePath1)
    data2 = load_power_curve_file(filePath2)
    print(f"  File 1: {filePath1}")
    print(f"  File 2: {filePath2}")

    label1 = get_file_label(data1, filePath1)
    label2 = get_file_label(data2, filePath2)

    profileIds1 = sorted(build_profile_index(data1).keys())
    profileIds2 = sorted(build_profile_index(data2).keys())
    allProfileIds = sorted(set(profileIds1) | set(profileIds2))

    print(f"\n  {label1}: {len(profileIds1)} profile(s) – IDs {profileIds1}")
    print(f"  {label2}: {len(profileIds2)} profile(s) – IDs {profileIds2}")
    print(f"\n  Plotting {len(allProfileIds)} profile(s) in total.")

    nProfiles = len(allProfileIds)

    if nProfiles == 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        axes = [ax]
    else:
        nCols = min(3, nProfiles)
        nRows = math.ceil(nProfiles / nCols)
        fig, axes = plt.subplots(
            nRows, nCols,
            figsize=(6 * nCols, 5 * nRows),
            squeeze=False,
        )
        axes = axes.flatten().tolist()

    for idx, profileId in enumerate(allProfileIds):
        plot_single(axes[idx], data1, data2, profileId, label1, label2)

    # Hide any unused subplots
    for idx in range(nProfiles, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Power Curve Comparison", fontsize=14, y=1.01)
    fig.tight_layout()

    resultsDir = PROJECT_ROOT / "results"
    resultsDir.mkdir(exist_ok=True)
    plotPath = resultsDir / "power_curve_comparison.png"
    fig.savefig(plotPath, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {plotPath}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    args = sys.argv[1:]
    if len(args) >= 2:
        filePath1 = Path(args[0])
        filePath2 = Path(args[1])
    else:
        filePath1 = DEFAULT_FILE_1
        filePath2 = DEFAULT_FILE_2

    compare_power_curves(filePath1, filePath2)


if __name__ == "__main__":
    main()
