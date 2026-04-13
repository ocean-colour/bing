"""
Utilities for working with JR (Frouin) processed Rrs spectra
from PACE L1B matchups.
"""

import numpy as np
import pandas as pd
import re
import os


def load_jr_data(jr_file: str = None):
    """Load the JR matchup CSV file.

    Args:
        jr_file: Path to the JR CSV file. Defaults to
            Analysis/Frouin/jr_test_matchup_L1B.csv

    Returns:
        jr_df: DataFrame with JR matchup data
    """
    if jr_file is None:
        # Default path relative to this module
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        jr_file = os.path.join(base_dir, 'Frouin',
                               'jr_test_matchup_L1B.csv')
    jr_df = pd.read_csv(jr_file)
    # Parse the time column
    jr_df['time'] = pd.to_datetime(jr_df['time'])
    return jr_df


def load_argo_data(argo_file: str = None):
    """Load the matched Argo BGC profiles CSV file.

    Args:
        argo_file: Path to the Argo CSV file. Defaults to
            Analysis/matched_argo_bgc_profiles_bbp_v3.csv

    Returns:
        argo_df: DataFrame with Argo BGC matchup data
    """
    if argo_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        argo_file = os.path.join(base_dir,
                                 'matched_argo_bgc_profiles_bbp_v3.csv')
    argo_df = pd.read_csv(argo_file)
    argo_df['time'] = pd.to_datetime(argo_df['time'], format='mixed')
    return argo_df


def parse_wavelengths(jr_df: pd.DataFrame):
    """Extract wavelength array from JR Rrs_mean column names.

    Args:
        jr_df: JR matchup DataFrame

    Returns:
        wavelengths: numpy array of wavelengths in nm
        mean_cols: list of Rrs_mean column names
        std_cols: list of Rrs_std column names
    """
    # Identify Rrs_mean and Rrs_std columns
    mean_cols = [c for c in jr_df.columns if c.startswith('Rrs_mean_')]
    std_cols = [c for c in jr_df.columns if c.startswith('Rrs_std_')]

    # Parse wavelengths from column names (e.g. "Rrs_mean_339.16nm, 1/sr")
    wavelengths = np.array([
        float(re.search(r'Rrs_mean_([0-9.]+)nm', c).group(1))
        for c in mean_cols
    ])

    return wavelengths, mean_cols, std_cols


def extract_rrs(cruise: int, profile: int,
                jr_file: str = None, argo_file: str = None):
    """Extract JR Rrs spectrum for a given Argo float cruise and profile.

    Loads the Argo BGC table, finds the row matching the given cruise
    and profile, then matches to the JR file by closest lat, lon, and time.

    Args:
        cruise: Argo float cruise (WMO) number
        profile: Argo float profile number
        jr_file: Path to JR CSV file (optional)
        argo_file: Path to Argo CSV file (optional)

    Returns:
        dict with keys:
            'wavelengths': np.array of wavelengths (nm)
            'Rrs_mean': np.array of mean Rrs values (1/sr)
            'Rrs_std': np.array of Rrs standard deviations (1/sr)
            'jr_row': matched JR DataFrame row
            'argo_row': matched Argo DataFrame row
            'match_dist_km': approximate distance between Argo and JR (km)
            'match_dt_hours': time difference in hours

    Raises:
        ValueError: if cruise/profile not found in Argo table or
                    no JR match found
    """
    # Load data
    argo_df = load_argo_data(argo_file)
    jr_df = load_jr_data(jr_file)

    # Find the Argo row
    mask = (argo_df['cruise'] == cruise) & (argo_df['profile'] == profile)
    if mask.sum() == 0:
        raise ValueError(
            f'No Argo entry found for cruise={cruise}, profile={profile}')
    argo_row = argo_df[mask].iloc[0]

    # Match to JR file by lat, lon, time
    # Compute distance (approximate, using cos(lat) correction)
    lat_argo = argo_row['lat']
    lon_argo = argo_row['lon']
    time_argo = argo_row['time']

    # Haversine-like approximate distance in km
    dlat = jr_df['PACE_lat'].values - lat_argo
    dlon = jr_df['PACE_lon'].values - lon_argo
    cos_lat = np.cos(np.radians(lat_argo))
    dist_deg = np.sqrt(dlat**2 + (dlon * cos_lat)**2)
    dist_km = dist_deg * 111.0  # approximate km per degree

    # Time difference in hours
    dt = np.abs((jr_df['time'] - time_argo).dt.total_seconds()) / 3600.0

    # Combined metric: distance in km + time penalty
    # Weight time: 1 hour ~ 10 km for matching purposes
    metric = dist_km + dt * 10.0

    best_idx = metric.idxmin()
    jr_row = jr_df.loc[best_idx]

    # Parse wavelengths and extract spectra
    wavelengths, mean_cols, std_cols = parse_wavelengths(jr_df)
    Rrs_mean = jr_row[mean_cols].values.astype(float)
    Rrs_std = jr_row[std_cols].values.astype(float)

    return {
        'wavelengths': wavelengths,
        'Rrs_mean': Rrs_mean,
        'Rrs_std': Rrs_std,
        'jr_row': jr_row,
        'argo_row': argo_row,
        'match_dist_km': dist_km[best_idx],
        'match_dt_hours': dt[best_idx],
    }
