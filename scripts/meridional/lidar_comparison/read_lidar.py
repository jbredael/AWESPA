"""Lidar Data Reader for Wind Profile Analysis.

This module reads WindCube WLS7-130 lidar data from .rtd files and converts
them to the format expected by the wind profile clustering pipeline.

Data characteristics:
- Instrument: Leosphere WindCube WLS7-130
- Location: Bangor Erris, Ireland (54.1254°N, 9.7801°W)
- Measurement technique: DBS (Doppler Beam Swinging) with 4 azimuthal beams
- Altitude range: 40 m to 250 m above ground
- Temporal resolution: ~4 seconds per complete scan cycle

Each scan cycle produces beam-position rows (0°, 90°, 180°, 270°, V).
Only the 270° row contains the synthesised horizontal wind vector
(X-wind = East, Y-wind = North) for all altitude gates.
"""

import warnings
import re
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


HEADER_SIZE_KEY = "HeaderSize="


def parse_rtd_header(file_path):
    """Parse the header section of an RTD file to extract instrument metadata.

    Args:
        file_path (str or Path): Path to the .rtd file.

    Returns:
        dict: Metadata including:
            - 'n_header_lines' (int): Number of header lines before the column row.
            - 'altitudes' (ndarray): Measurement altitude gates [m].
            - 'location_name' (str): Site name.
    """
    with open(file_path, 'r', encoding='latin-1') as f:
        firstLine = f.readline().strip()

    nHeaderLines = 40  # WLS7-130 default
    if firstLine.startswith(HEADER_SIZE_KEY):
        nHeaderLines = int(firstLine[len(HEADER_SIZE_KEY):])

    metadata = {'n_header_lines': nHeaderLines}

    with open(file_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i >= nHeaderLines:
                break
            line = line.strip()
            if line.startswith('Altitudes (m)='):
                altStr = line.split('=', 1)[1].strip()
                metadata['altitudes'] = np.array([float(a) for a in altStr.split()])
            elif line.startswith('Location='):
                metadata['location_name'] = line.split('=', 1)[1].strip()
            elif line.startswith('GPS Location='):
                metadata['gps_location'] = line.split('=', 1)[1].strip()

    return metadata


def read_rtd_file(file_path):
    """Read a single RTD lidar file and extract horizontal wind profile data.

    Only rows with Position == '270' are retained.  These rows contain the
    synthesised horizontal wind vector after a complete 4-beam DBS sweep.

    Args:
        file_path (str or Path): Path to the .rtd file.

    Returns:
        tuple: (pandas.DataFrame indexed by datetime, metadata dict).
            Returns (None, None) on failure.
    """
    file_path = Path(file_path)
    metadata = parse_rtd_header(file_path)

    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            skiprows=metadata['n_header_lines'] + 1,  # +1 to skip the '***' separator line
            header=0,
            index_col=False,   # prevent auto-index when data has 1 more field than header
            na_values=['NaN', 'nan', ''],
            engine='c',
            encoding='latin-1',
            on_bad_lines='skip',
        )
    except Exception as e:
        warnings.warn(f"Failed to read {file_path.name}: {e}")
        return None, None

    df.columns = [c.strip() for c in df.columns]

    if 'Position' not in df.columns:
        warnings.warn(f"No 'Position' column in {file_path.name}")
        return None, None

    # Keep only complete-scan rows (Position == 270) which carry the wind vector
    # Position is object dtype (string) because beam 'V' prevents numeric coercion
    df = df[df['Position'].astype(str).str.strip() == '270'].copy()

    if df.empty:
        return None, None

    df['datetime'] = pd.to_datetime(
        df['Timestamp'].str.strip(),
        format='%Y/%m/%d %H:%M:%S.%f',
        errors='coerce',
    )
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    return df, metadata


def extract_wind_components(df, altitudes):
    """Extract east (X) and north (Y) wind components at each altitude gate.

    Args:
        df (pandas.DataFrame): RTD data filtered to Position == 270 rows.
        altitudes (ndarray): Altitude gates to extract [m].

    Returns:
        tuple: (windEast, windNorth, validAltitudes) arrays of shape
            (nTime, nAltitudes).
    """
    eastCols, northCols, validAlt = [], [], []

    for alt in altitudes:
        eastName = f"{int(alt)}m X-wind (m/s)"
        northName = f"{int(alt)}m Y-wind (m/s)"
        if eastName in df.columns and northName in df.columns:
            eastCols.append(eastName)
            northCols.append(northName)
            validAlt.append(alt)

    if not eastCols:
        raise ValueError(
            f"No X-wind / Y-wind columns found for altitudes {altitudes}. "
            f"Available columns: {list(df.columns[:20])}"
        )

    windEast = df[eastCols].values.astype(float)
    windNorth = df[northCols].values.astype(float)
    return windEast, windNorth, np.array(validAlt)


def read_data(config=None):
    """Read lidar wind data from RTD files and combine into a single dataset.

    Reads all WindCube WLS7-130 .rtd files in the specified directory and
    combines them into a single dataset compatible with the wind profile
    clustering pipeline.

    Args:
        config (dict, optional): Configuration dictionary with optional keys:
            - 'data_dir' (str or Path): Directory containing .rtd files.
            - 'date_range' (tuple): (startDate, endDate) as 'YYYY-MM-DD' strings.
            - 'altitudes' (list): Altitude gates to extract [m]. Uses all by default.

    Returns:
        dict: Dataset containing:
            - 'wind_speed_east' (ndarray): East component [m/s] (nSamples x nAlt).
            - 'wind_speed_north' (ndarray): North component [m/s] (nSamples x nAlt).
            - 'n_samples' (int): Number of time samples.
            - 'datetime' (ndarray): Datetime values.
            - 'altitude' (ndarray): Altitude gates [m].
            - 'years' (tuple): (firstYear, lastYear).
    """
    if config is None:
        config = {}

    defaultDataDir = (
        Path(__file__).parent.parent.parent.parent / 'data' / 'wind_data' / 'lidar'
    )
    dataDir = Path(config.get('data_dir', defaultDataDir))
    dateRange = config.get('date_range', None)
    targetAltitudes = config.get('altitudes', None)

    rtdFiles = sorted(glob(str(dataDir / '*.rtd')))
    if not rtdFiles:
        raise FileNotFoundError(f"No .rtd files found in {dataDir}")

    print(f"Found {len(rtdFiles)} RTD files in {dataDir}")

    # Optional date filtering based on the date embedded in the filename
    if dateRange is not None:
        startDate = pd.Timestamp(dateRange[0])
        endDate = pd.Timestamp(dateRange[1])
        filtered = []
        for fp in rtdFiles:
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})', Path(fp).stem)
            if match:
                fileDate = pd.Timestamp(
                    f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                )
                if startDate <= fileDate <= endDate:
                    filtered.append(fp)
            else:
                filtered.append(fp)  # include if we cannot parse the date
        rtdFiles = filtered
        print(f"Filtered to {len(rtdFiles)} files ({dateRange[0]} to {dateRange[1]})")

    if not rtdFiles:
        raise ValueError("No files remain after date filtering")

    allEast, allNorth, allDatetime = [], [], []
    commonAltitudes = None

    for i, fp in enumerate(rtdFiles):
        print(f"  [{i + 1}/{len(rtdFiles)}] {Path(fp).name}")
        df, metadata = read_rtd_file(fp)
        if df is None or df.empty:
            continue

        fileAltitudes = metadata.get('altitudes')
        if fileAltitudes is None:
            continue

        # Establish the shared altitude grid from the first successful file
        if commonAltitudes is None:
            commonAltitudes = (
                np.array(targetAltitudes, dtype=float)
                if targetAltitudes is not None
                else fileAltitudes
            )

        try:
            windEast, windNorth, validAlt = extract_wind_components(df, commonAltitudes)
        except Exception as e:
            warnings.warn(f"Skipping {Path(fp).name}: {e}")
            continue

        # If some altitudes are missing, pad with NaN
        if not np.array_equal(validAlt, commonAltitudes):
            nT = len(df)
            fullEast = np.full((nT, len(commonAltitudes)), np.nan)
            fullNorth = np.full((nT, len(commonAltitudes)), np.nan)
            for j, alt in enumerate(commonAltitudes):
                idx = np.where(validAlt == alt)[0]
                if len(idx) > 0:
                    fullEast[:, j] = windEast[:, idx[0]]
                    fullNorth[:, j] = windNorth[:, idx[0]]
            windEast, windNorth = fullEast, fullNorth

        allEast.append(windEast)
        allNorth.append(windNorth)
        allDatetime.extend(df.index.tolist())

    if not allEast:
        raise ValueError("No data could be read from any RTD files")

    combinedEast = np.concatenate(allEast, axis=0)
    combinedNorth = np.concatenate(allNorth, axis=0)
    combinedDatetime = np.array(allDatetime)

    # Sort chronologically
    sortIdx = np.argsort(combinedDatetime)
    combinedEast = combinedEast[sortIdx]
    combinedNorth = combinedNorth[sortIdx]
    combinedDatetime = combinedDatetime[sortIdx]

    yearsIdx = pd.DatetimeIndex(combinedDatetime).year
    yearRange = (int(yearsIdx.min()), int(yearsIdx.max()))

    nSamples = len(combinedDatetime)
    print(f"\nTotal samples : {nSamples}")
    print(f"Altitude [m]  : {commonAltitudes}")
    print(f"Time range    : {combinedDatetime[0]} → {combinedDatetime[-1]}")
    print(f"Years         : {yearRange[0]}–{yearRange[1]}")

    return {
        'wind_speed_east': combinedEast,
        'wind_speed_north': combinedNorth,
        'n_samples': nSamples,
        'datetime': combinedDatetime,
        'altitude': commonAltitudes,
        'years': yearRange,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_mean_wind_profile(data, savePath=None):
    """Plot mean horizontal wind speed profile with interquartile range band.

    Args:
        data (dict): Output of read_data().
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    speed = np.sqrt(data['wind_speed_east'] ** 2 + data['wind_speed_north'] ** 2)
    altitudes = data['altitude']

    meanSpeed = np.nanmean(speed, axis=0)
    q25 = np.nanpercentile(speed, 25, axis=0)
    q75 = np.nanpercentile(speed, 75, axis=0)

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.fill_betweenx(altitudes, q25, q75, alpha=0.3, color='steelblue', label='IQR (25–75 %)')
    ax.plot(meanSpeed, altitudes, 'o-', color='steelblue', lw=2, label='Mean')
    ax.set_xlabel('Horizontal wind speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Mean wind speed profile\n(lidar – Bangor Erris)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(altitudes[0] - 10, altitudes[-1] + 10)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


def plot_time_series(data, altitudeIdxs=None, savePath=None):
    """Plot horizontal wind speed time series at selected altitude gates.

    Args:
        data (dict): Output of read_data().
        altitudeIdxs (list, optional): Altitude indices to plot. Defaults to
            3 evenly spaced levels.
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    speed = np.sqrt(data['wind_speed_east'] ** 2 + data['wind_speed_north'] ** 2)
    altitudes = data['altitude']
    datetimes = pd.DatetimeIndex(data['datetime'])

    if altitudeIdxs is None:
        nAlt = len(altitudes)
        altitudeIdxs = np.linspace(0, nAlt - 1, min(4, nAlt), dtype=int).tolist()

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(altitudeIdxs)))

    fig, ax = plt.subplots(figsize=(14, 4))
    for color, idx in zip(colors, altitudeIdxs):
        ax.plot(
            datetimes, speed[:, idx],
            lw=0.4, alpha=0.7, color=color,
            label=f"{int(altitudes[idx])} m",
        )

    ax.set_xlabel('Date')
    ax.set_ylabel('Wind speed (m/s)')
    ax.set_title('Wind speed time series at selected altitude gates')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


def plot_wind_speed_heatmap(data, savePath=None):
    """Plot a time–altitude curtain of horizontal wind speed (hourly means).

    Args:
        data (dict): Output of read_data().
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    speed = np.sqrt(data['wind_speed_east'] ** 2 + data['wind_speed_north'] ** 2)
    altitudes = data['altitude']
    datetimes = pd.DatetimeIndex(data['datetime'])

    df = pd.DataFrame(speed, index=datetimes, columns=altitudes)
    dfHourly = df.resample('1h').mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    mesh = ax.pcolormesh(
        dfHourly.index,
        dfHourly.columns,
        dfHourly.values.T,
        cmap='viridis',
        shading='auto',
        vmin=0,
        vmax=25,
    )
    fig.colorbar(mesh, ax=ax, label='Wind speed (m/s)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Wind speed – altitude curtain plot (hourly means)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


def plot_wind_direction_profile(data, savePath=None):
    """Plot mean wind direction profile with circular standard deviation band.

    Args:
        data (dict): Output of read_data().
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    u = data['wind_speed_east']
    v = data['wind_speed_north']
    altitudes = data['altitude']

    # Meteorological direction: direction FROM which wind blows
    windDir = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    sinMean = np.nanmean(np.sin(np.radians(windDir)), axis=0)
    cosMean = np.nanmean(np.cos(np.radians(windDir)), axis=0)
    meanDir = (np.degrees(np.arctan2(sinMean, cosMean)) + 360) % 360

    # Circular standard deviation
    R = np.hypot(sinMean, cosMean)
    R = np.clip(R, 1e-6, 1.0)  # guard against log(0)
    stdDir = np.degrees(np.sqrt(-2.0 * np.log(R)))

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.fill_betweenx(
        altitudes, meanDir - stdDir, meanDir + stdDir,
        alpha=0.25, color='darkorange', label='±1 std (circular)',
    )
    ax.plot(meanDir, altitudes, 'o-', color='darkorange', lw=2, label='Mean direction')
    ax.set_xlabel('Wind direction (° from North, meteorological)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Mean wind direction profile\n(lidar – Bangor Erris)')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['N\n0°', 'E\n90°', 'S\n180°', 'W\n270°', 'N\n360°'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(altitudes[0] - 10, altitudes[-1] + 10)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


def plot_data_availability(data, savePath=None):
    """Plot daily data availability as a bar chart.

    Args:
        data (dict): Output of read_data().
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    datetimes = pd.DatetimeIndex(data['datetime'])
    midIdx = len(data['altitude']) // 2
    validMask = ~np.isnan(data['wind_speed_east'][:, midIdx])

    dfValid = pd.Series(validMask.astype(int), index=datetimes)
    dailyValid = dfValid.resample('D').sum()
    dailyTotal = pd.Series(1, index=datetimes).resample('D').count()
    availability = (dailyValid / dailyTotal * 100).clip(upper=100)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(availability.index, availability.values, width=0.8, color='steelblue', alpha=0.8)
    ax.axhline(80, color='orange', lw=1.5, linestyle='--', label='80 % threshold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valid scans (%)')
    ax.set_title(f"Daily data availability at {int(data['altitude'][midIdx])} m")
    ax.legend()
    ax.set_ylim(0, 108)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


def plot_wind_rose(data, altitudeIdx=None, savePath=None):
    """Plot a wind rose showing speed-weighted directional frequency.

    Args:
        data (dict): Output of read_data().
        altitudeIdx (int, optional): Altitude index to use. Defaults to
            mid-altitude gate.
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    if altitudeIdx is None:
        altitudeIdx = len(data['altitude']) // 2

    u = data['wind_speed_east'][:, altitudeIdx]
    v = data['wind_speed_north'][:, altitudeIdx]
    speed = np.hypot(u, v)
    windDir = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    valid = ~np.isnan(speed) & ~np.isnan(windDir)
    speed = speed[valid]
    windDir = windDir[valid]

    nDirBins = 16
    dirEdges = np.linspace(0, 360, nDirBins + 1)
    dirCentres = 0.5 * (dirEdges[:-1] + dirEdges[1:])
    speedBins = [0, 5, 10, 15, 20, np.inf]
    speedLabels = ['0–5 m/s', '5–10 m/s', '10–15 m/s', '15–20 m/s', '>20 m/s']
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(speedLabels)))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    barWidth = 2 * np.pi / nDirBins

    cumCounts = np.zeros(nDirBins)
    for k, (sLow, sHigh) in enumerate(zip(speedBins[:-1], speedBins[1:])):
        inSpeed = (speed >= sLow) & (speed < sHigh)
        counts = np.array([
            np.sum(inSpeed & (windDir >= dirEdges[j]) & (windDir < dirEdges[j + 1]))
            for j in range(nDirBins)
        ]) / len(speed) * 100

        ax.bar(
            np.radians(dirCentres), counts,
            width=barWidth, bottom=cumCounts,
            color=colors[k], alpha=0.9, label=speedLabels[k],
        )
        cumCounts += counts

    alt = int(data['altitude'][altitudeIdx])
    ax.set_title(
        f"Wind rose at {alt} m\n"
        f"(Bangor Erris, {data['years'][0]}–{data['years'][1]})",
        pad=15,
    )
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=8)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, dpi=150)
        print(f"Saved: {savePath}")
    return fig


if __name__ == '__main__':
    print("Reading lidar data...")
    data = read_data()

    outDir = Path(__file__).parent.parent.parent.parent / 'results' / 'plots'
    outDir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_mean_wind_profile(data, savePath=outDir / 'lidar_mean_wind_profile.png')
    plot_time_series(data, savePath=outDir / 'lidar_wind_speed_time_series.png')
    plot_wind_speed_heatmap(data, savePath=outDir / 'lidar_wind_speed_heatmap.png')
    plot_wind_direction_profile(data, savePath=outDir / 'lidar_wind_direction_profile.png')
    plot_data_availability(data, savePath=outDir / 'lidar_data_availability.png')
    plot_wind_rose(data, savePath=outDir / 'lidar_wind_rose.png')

    plt.show()
    print("\nDone. Plots saved to", outDir)
