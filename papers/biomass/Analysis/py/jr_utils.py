"""
Utilities for working with JR (Frouin) processed Rrs spectra
from PACE L1B matchups.
"""

import numpy as np
import pandas as pd
import re
import os

from IPython import embed


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


def match_argo_to_jr(cruise: int, profile: int,
                     jr_file: str = None, argo_file: str = None,
                     latlon_tol_deg: float = 0.5,
                     time_tol_hours: float = 0.1):
    """Match an Argo (cruise, profile) to its JR matchup row.

    Matching uses, from the matched CSV row for ``(cruise, profile)``:
    ``closest_file`` (exact equality with JR ``AOP_file``), ``lat``/``lon``
    (Argo float coordinates, until ``PACE_lat``/``PACE_lon`` are added to
    the matched CSV), and the closest profile time.  Note that the JR
    ``time`` column currently stores the Argo profile time -- the same
    value as the matched CSV's ``time`` column, not its ``closest_time``
    (PACE granule time) column -- so we use ``time`` here to honor the
    "nearly exact" requirement.  Swap to ``closest_time`` once the JR
    extractor adopts the granule timestamp.

    Parameters
    ----------
    cruise, profile : int
        Argo float identifiers.
    jr_file, argo_file : str, optional
        Override paths for the JR and matched-Argo CSVs.
    latlon_tol_deg : float
        Allowed |Δlat| and |Δlon| (degrees) between the Argo float
        location and the JR row's PACE pixel.  Defaults to 0.5° (~50 km)
        because the matched CSV currently stores the Argo float
        coordinates, not the PACE pixel.
    time_tol_hours : float
        Max allowed |Δt| in hours between the matched-CSV profile time
        and the JR row's time.  "Nearly exact" -> tight default.

    Returns
    -------
    dict
        ``{'jr_idx', 'argo_row', 'aop_file', 'dist_km', 'dt_hours'}``.

    Raises
    ------
    ValueError
        If the (cruise, profile) is missing from the matched CSV, no JR
        row shares the closest_file, or no candidate satisfies the
        lat/lon and time tolerances.
    """
    # Resolve the matched Argo row for this (cruise, profile)
    argo_df = load_argo_data(argo_file)
    mask = (argo_df['cruise'] == cruise) & (argo_df['profile'] == profile)
    if mask.sum() == 0:
        raise ValueError(
            f'No Argo entry found for cruise={cruise}, profile={profile}')
    argo_row = argo_df[mask].iloc[0]

    # Primary key: filter JR by AOP_file == matched closest_file
    jr_df = load_jr_data(jr_file)
    sel = jr_df[jr_df['AOP_file'] == argo_row['closest_file']]
    if len(sel) == 0:
        raise ValueError(
            f'No JR row with AOP_file == {argo_row["closest_file"]!r} '
            f'(cruise={cruise}, profile={profile})')

    # Restrict to rows whose PACE pixel is near the Argo float location
    cos_lat = np.cos(np.radians(argo_row['lat']))
    dlat = sel['PACE_lat'].values - argo_row['lat']
    dlon = sel['PACE_lon'].values - argo_row['lon']
    within = ((np.abs(dlat) < latlon_tol_deg)
              & (np.abs(dlon) < latlon_tol_deg))
    if not within.any():
        raise ValueError(
            f'No JR candidate within {latlon_tol_deg} deg of Argo '
            f'lat={argo_row["lat"]}, lon={argo_row["lon"]}')
    sel = sel[within]

    # Pick the closest-in-time candidate and require near-exact agreement
    dt_hours = np.abs(
        (sel['time'] - argo_row['time']).dt.total_seconds()) / 3600.0
    best_local = int(np.argmin(dt_hours.values))
    best_idx = int(sel.index[best_local])
    best_dt = float(dt_hours.iloc[best_local])
    if best_dt > time_tol_hours:
        raise ValueError(
            f'Best JR time gap {best_dt:.4f} h exceeds tolerance '
            f'{time_tol_hours} h for cruise={cruise}, profile={profile}')

    # Diagnostics: actual distance between Argo float and PACE pixel
    best_lat = float(jr_df['PACE_lat'].loc[best_idx])
    best_lon = float(jr_df['PACE_lon'].loc[best_idx])
    dist_km = float(np.sqrt(
        (best_lat - argo_row['lat'])**2
        + ((best_lon - argo_row['lon']) * cos_lat)**2) * 111.0)

    return {
        'jr_idx': best_idx,
        'argo_row': argo_row,
        'aop_file': argo_row['closest_file'],
        'dist_km': dist_km,
        'dt_hours': best_dt,
    }


def match_jr_to_argo(jr_idx:int, 
                     jr_file: str = None, argo_file: str = None,
                     latlon_tol_deg: float = 0.5,
                     time_tol_hours: float = 0.1):
    """Match an Argo (cruise, profile) to its JR matchup row.

    Matching uses, from the matched CSV row for ``(cruise, profile)``:
    ``closest_file`` (exact equality with JR ``AOP_file``), ``lat``/``lon``
    (Argo float coordinates, until ``PACE_lat``/``PACE_lon`` are added to
    the matched CSV), and the closest profile time.  Note that the JR
    ``time`` column currently stores the Argo profile time -- the same
    value as the matched CSV's ``time`` column, not its ``closest_time``
    (PACE granule time) column -- so we use ``time`` here to honor the
    "nearly exact" requirement.  Swap to ``closest_time`` once the JR
    extractor adopts the granule timestamp.

    Parameters
    ----------
    cruise, profile : int
        Argo float identifiers.
    jr_file, argo_file : str, optional
        Override paths for the JR and matched-Argo CSVs.
    latlon_tol_deg : float
        Allowed |Δlat| and |Δlon| (degrees) between the Argo float
        location and the JR row's PACE pixel.  Defaults to 0.5° (~50 km)
        because the matched CSV currently stores the Argo float
        coordinates, not the PACE pixel.
    time_tol_hours : float
        Max allowed |Δt| in hours between the matched-CSV profile time
        and the JR row's time.  "Nearly exact" -> tight default.

    Returns
    -------
    dict
        ``{'jr_idx', 'argo_row', 'aop_file', 'dist_km', 'dt_hours'}``.

    Raises
    ------
    ValueError
        If the (cruise, profile) is missing from the matched CSV, no JR
        row shares the closest_file, or no candidate satisfies the
        lat/lon and time tolerances.
    """
    # Resolve the matched Argo row for this (cruise, profile)
    argo_df = load_argo_data(argo_file)

    # Primary key: filter JR by AOP_file == matched closest_file
    jr_df = load_jr_data(jr_file)
    jr_s = jr_df.iloc[jr_idx]
    sel = argo_df[jr_s['AOP_file'] == argo_df['closest_file']]
    if len(sel) == 0:
        raise ValueError(
            f'No Argo row with PACE AOP_file == {jr_s["closest_file"]!r} ')

    # Restrict to rows whose PACE pixel is near the Argo float location
    cos_lat = np.cos(np.radians(jr_s['PACE_lat']))
    dlat = jr_s['PACE_lat'] - argo_df['lat'].values
    dlon = jr_s['PACE_lon'] - argo_df['lon'].values
    within = ((np.abs(dlat) < latlon_tol_deg)
              & (np.abs(dlon) < latlon_tol_deg))
    if not within.any():
        raise ValueError(
            f'No Argo float within within {latlon_tol_deg} deg of JR analysis '
            f'lat={jr_s["PACE_lat"]}, lon={jr_s["PACE_lon"]}')
    argo_df = argo_df[within]

    # Pick the closest-in-time candidate and require near-exact agreement
    dt_hours = np.abs(
        (jr_s['time'] - argo_df['time']).dt.total_seconds()) / 3600.0
    best_local = int(np.argmin(dt_hours.values))
    best_idx = int(argo_df.index[best_local])
    best_dt = float(dt_hours.iloc[best_local])
    if best_dt > time_tol_hours:
        raise ValueError(
            f'Best time gap {best_dt:.4f} h exceeds tolerance '
            f'{time_tol_hours} h for JR idx {jr_idx}')

    # Parse
    argo_row = argo_df.loc[best_idx]

    # Diagnostics: actual distance between Argo float and PACE pixel
    best_lat = float(argo_row['lat'])
    best_lon = float(argo_row['lon'])
    dist_km = float(np.sqrt(
        (best_lat - jr_s['PACE_lat'])**2
        + ((best_lon - jr_s['PACE_lon']) * cos_lat)**2) * 111.0)

    # Return
    return {
        'jr_idx': jr_idx,
        'argo_row': best_idx,
        'aop_file': argo_row['closest_file'],
        'dist_km': dist_km,
        'dt_hours': best_dt,
        'cruise': int(argo_row['cruise']),
        'profile': int(argo_row['profile']),
    }


def extract_rrs(jr_idx: int,
                jr_file: str = None, argo_file: str = None,
                latlon_tol_deg: float = 0.5,
                time_tol_hours: float = 0.1):
    """Extract the JR Rrs spectrum for a given Argo float cruise/profile.

    Delegates row selection to :func:`match_jr_to_pace`, which uses the
    matched CSV's ``closest_file``/``lat``/``lon``/``time`` columns.

    Args:
        jr_idx: Index of Frouin extraction
        jr_file: Path to JR CSV file (optional)
        argo_file: Path to Argo CSV file (optional)
        latlon_tol_deg: Lat/lon match tolerance (degrees)
        time_tol_hours: Time match tolerance (hours)

    Returns:
        dict with keys:
            'wavelengths': np.array of wavelengths (nm)
            'Rrs_mean': np.array of mean Rrs values (1/sr)
            'Rrs_std': np.array of Rrs standard deviations (1/sr)
            'jr_row': matched JR DataFrame row
            'argo_row': matched Argo DataFrame row
            'match_dist_km': Argo float -> PACE pixel distance (km)
            'match_dt_hours': time difference (hours)
    """
    # Identify the matching JR row via closest_file + lat/lon + time

    # Re-load the JR table to pull the full spectrum columns
    jr_df = load_jr_data(jr_file)
    jr_row = jr_df.iloc[jr_idx]

    # Parse wavelengths and extract spectra
    wavelengths, mean_cols, std_cols = parse_wavelengths(jr_df)
    Rrs_mean = jr_row[mean_cols].values.astype(float)
    Rrs_std = jr_row[std_cols].values.astype(float)

    return {
        'wavelengths': wavelengths,
        'Rrs_mean': Rrs_mean,
        'Rrs_std': Rrs_std,
        'jr_row': jr_row,
    }
