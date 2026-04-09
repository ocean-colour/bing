"""
ocean_biogeochem2.py
--------------------
Process BGC-Argo NetCDF profiles and extract median surface (top 25 m)
particle backscattering at 700 nm (bbp700), matched against a reference CSV.

Typical usage
-------------
    from ocean_biogeochem2 import run

    run(
        nc_files=[
            "path/to/GulfofMexico.nc",
            "path/to/Mediterranean.nc",
        ],
        csv_path="matched_argo_bgc_profiles_bbp_v3.csv",
        out_path="ocean_biogeochem_top25m_bbp.csv",
    )
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

from IPython import embed

# ================================
# CONSTANTS
# ================================
VAR_BBP    = "Particle_backscattering_at_700_nm_adjusted_"
VAR_BBP_QC = "Particle_backscattering_at_700_nm_adjusted__qc"
VAR_DEPTH  = "Pressure_adjusted_"

CRUISE_OPTIONS  = ["Platform_Number", "cruise_id"]
TIME_OPTIONS    = ["JULD", "date_time"]
LAT_OPTIONS     = ["latitude", "LATITUDE"]
LON_OPTIONS     = ["longitude", "LONGITUDE"]
PROFILE_DIM_OPTIONS = ["N_STATIONS", "station"]


# ================================
# HELPERS
# ================================
def get_var(ds: xr.Dataset, options: list[str]) -> str:
    """Return the first variable name from *options* that exists in *ds*."""
    for v in options:
        if v in ds:
            return v
    raise KeyError(f"None of {options} found in dataset variables")


def get_dim(ds: xr.Dataset, options: list[str]) -> str:
    """Return the first dimension name from *options* that exists in *ds*."""
    for d in options:
        if d in ds.dims:
            return d
    raise KeyError(f"None of {options} found in dataset dimensions")


# ================================
# PROFILE PROCESSING
# ================================
def process_profile(
    prof: xr.Dataset,
    surface_depth: float = 25.0,
    qc_threshold: int = 50,
    min_surface: int = 3,
) -> dict | None:
    """
    Extract the median bbp700 for the surface layer of a single profile.

    Parameters
    ----------
    prof : xr.Dataset
        A single profile (already sliced from the full dataset).
    surface_depth : float
        Maximum pressure (dbar) considered as the surface layer.
    qc_threshold : int
        Keep only values whose QC flag is <= this value.
    min_surface : int
        Minimum number of valid surface values required; returns None otherwise.

    Returns
    -------
    dict with keys ``bbp700_top25m_median`` and ``n_values_used``, or None.
    """
    good = prof[VAR_BBP_QC].data <= qc_threshold
    if not np.any(good):
        return None

    depth = prof[VAR_DEPTH].data[good]
    bbp   = prof[VAR_BBP].data[good]

    surface_mask = depth <= surface_depth
    vals = bbp[surface_mask]

    if len(vals) < min_surface:
        raise ValueError(f"Not enough surface values: {len(vals)} < {min_surface}")

    return {
        "bbp700_top25m_median": float(np.median(vals)),
        "n_values_used": int(len(vals)),
    }

# ================================
# FILE PROCESSING
# ================================
def process_file(
    path: str,
    csv_df: pd.DataFrame,
    surface_depth: float = 25.0,
    qc_threshold: int = 50,
    min_surface: int = 3,
    coord_tol: float = 1e-3,
) -> list[dict]:
    """
    Match profiles in a NetCDF file against rows in *csv_df* and extract
    surface bbp700 statistics.

    Parameters
    ----------
    path : str
        Path to the BGC-Argo NetCDF file.
    csv_df : pd.DataFrame
        Reference table with columns: filename, cruise, lat/latitude, lon/longitude.
    surface_depth : float
        Passed through to :func:`process_profile`.
    qc_threshold : int
        Passed through to :func:`process_profile`.
    min_surface : int
        Passed through to :func:`process_profile`.
    coord_tol : float
        Absolute tolerance (degrees) for latitude/longitude matching.

    Returns
    -------
    List of row dicts ready to be converted to a DataFrame.
    """
    rows = []
    fname = os.path.basename(path)
    print(f"\nProcessing: {fname}")

    ds = xr.open_dataset(path)

    var_cruise  = get_var(ds, CRUISE_OPTIONS)
    var_time    = get_var(ds, TIME_OPTIONS)
    var_lat     = get_var(ds, LAT_OPTIONS)
    var_lon     = get_var(ds, LON_OPTIONS)
    dim_profile = get_dim(ds, PROFILE_DIM_OPTIONS)

    n_profiles = ds.sizes[dim_profile]

    meta = pd.DataFrame({
        "idx":    np.arange(n_profiles),  # Time sorted indices
        "cruise": ds[var_cruise].data.astype(str),
        "time":   ds[var_time].data,
        "lat":    ds[var_lat].data,
        "lon":    ds[var_lon].data,
    })

    # Filter the CSV to rows that belong to this file
    base_name = fname.split(".")[0]
    csv_sub = csv_df[csv_df["filename"].str.contains(base_name, na=False)]
    print(f"  -> {len(csv_sub)} relevant CSV rows")

    # Loop on the matched
    for _, csv_row in csv_sub.iterrows():
        cruise_target = csv_row["cruise"]
        prof_id = int(csv_row["profile"])
        lat_target    = csv_row.get("lat", csv_row.get("latitude"))
        lon_target    = csv_row.get("lon", csv_row.get("longitude"))

        group = (
            meta[meta["cruise"] == str(cruise_target)]
            .sort_values("time")
            .reset_index(drop=True)
        )
        if group.empty:
            continue


        # Grab group + profile
        igroup = group.iloc[prof_id]
        ds_idx = igroup['idx']
        prof   = ds.isel(**{dim_profile: int(ds_idx)}) #meta_row["idx"])})

        # Check lat, lon
        #embed(header='185 of ocean_biogeochem2.py')
        lat_match = float(prof.latitude)
        lon_match = float(prof.longitude)

        assert np.abs(lat_match - lat_target) < coord_tol
        assert np.abs(lon_match - lon_target) < coord_tol

        result = process_profile(
            prof,
            surface_depth=surface_depth,
            qc_threshold=qc_threshold,
            min_surface=min_surface,
        )
        if result is not None:
            rows.append({
                "cruise":               cruise_target,
                "profile_id":           prof_id,
                "time":                 igroup["time"],
                "latitude":             igroup["lat"],
                "longitude":            igroup["lon"],
                "bbp700_top25m_median": result["bbp700_top25m_median"],
                "n_values_used":        result["n_values_used"],
            })
        else:
            raise ValueError("Bad result")

    ds.close()
    print(f"  -> {len(rows)} matched profiles so far")
    return rows


# ================================
# PUBLIC ENTRY POINT
# ================================
def run(
    nc_files: list[str],
    csv_path: str,
    out_path: str = None,
    surface_depth: float = 25.0,
    qc_threshold: int = 50,
    min_surface: int = 3,
    coord_tol: float = 1e-3,
) -> pd.DataFrame:
    """
    Process a list of BGC-Argo NetCDF files and write matched surface bbp700
    statistics to a CSV.

    Parameters
    ----------
    nc_files : list[str]
        Paths to the NetCDF files to process.
    csv_path : str
        Path to the reference CSV (matched_argo_bgc_profiles_bbp_v3.csv).
    out_path : str, optional
        Destination path for the output CSV.
    surface_depth, qc_threshold, min_surface, coord_tol
        Forwarded to :func:`process_file` / :func:`process_profile`.

    Returns
    -------
    pd.DataFrame with all matched results (also saved to *out_path*).
        Columns:
        - cruise
        - profile_id
        - time
        - latitude
        - longitude
        - bbp700_top25m_median
        - n_values_used
    """
    csv_df   = pd.read_csv(csv_path)
    all_rows = []

    for path in nc_files:
        all_rows.extend(
            process_file(
                path,
                csv_df,
                surface_depth=surface_depth,
                qc_threshold=qc_threshold,
                min_surface=min_surface,
                coord_tol=coord_tol,
            )
        )

    # Table
    df = pd.DataFrame(all_rows)

    # Write to disk?
    if out_path is not None:
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to: {out_path}")
    return df


# ================================
# SCRIPT ENTRY POINT
# ================================
if __name__ == "__main__":
    nc_path = os.path.join(os.getenv('OS_DATA'), 'Argo', 'Med_Mexico')
    NC_FILES = [
        os.path.join(nc_path, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_GulfofMexico.nc"),
        os.path.join(nc_path, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_Mediterranean.nc"),
    ]
    CSV_PATH = "matched_argo_bgc_profiles_bbp_v3.csv"
    OUT_PATH = "ocean_biogeochem_top25m_bbp.csv"

    run(nc_files=NC_FILES, csv_path=CSV_PATH, out_path=OUT_PATH)
