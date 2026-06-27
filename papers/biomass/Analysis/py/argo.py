""" Routines related to Argo """

import os
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import xarray
import pandas

import pysolar

from shapely.vectorized import contains

# Local imports
import biomass_io

from IPython import embed


def _check_granule(args):
    """
    Check spatial and temporal match for a single PACE granule against all Argo profiles.

    Returns:
        tuple: (granule_index, inside_array, ok_time_array)
    """
    ss, polygon, pace_time, argo_lons, argo_lats, argo_times, dtime = args
    # Spatial containment
    inside = np.array(contains(polygon, argo_lons, argo_lats))
    # Temporal match
    dt = pandas.Timestamp(pace_time) - argo_times
    ok_time = np.array(np.abs(dt) < pandas.Timedelta(dtime))
    return ss, inside, ok_time


def load_orig_argo(csv_file:str='argo_bgc_profiles_bbp.csv',
              cut_for_pace:bool=True):
    """ Load original Argo profiles with good bbp data from a CSV file """

    print(f'Loading original Argo profiles from {csv_file}')
    df = pandas.read_csv(csv_file)
    df['time'] = pandas.to_datetime(df.time.values, utc=True)

    # Cut for PACE?
    if cut_for_pace:
        print(f'Cutting for PACE dates')
        old_enough = df.time > pandas.Timestamp('2024-04-01', tz='UTC')
        df = df[old_enough].copy()
        # Drop index
        df.reset_index(drop=True, inplace=True)

    # Deal with lon
    lons = df.lon.values
    lons[lons > 180] -= 360
    df['lon'] = lons

    # Return
    return df

def match_argo_to_pace(granule_file:str, out_file:str, dtime:str='1 day',
                       n_cores:int=15):
    """
    Matches Argo profiles to PACE granules based on spatial and temporal criteria.

    Parameters:
        granule_file (str): The file path to the JSON file containing PACE granule data.
        out_file (str): The file path where the matched Argo profiles will be saved as a CSV.
        dtime (str, optional): The time window for matching Argo profiles to PACE granules.
                                Defaults to '1 day'. The format should be compatible with
                                pandas.Timedelta.
        n_cores (int, optional): Number of CPU cores for parallel granule matching.
                                  Defaults to 15.

    Description:
        - Loads Argo profiles and PACE granules.
        - Checks if Argo profiles are spatially contained within PACE granules.
        - Filters Argo profiles based on the specified time window relative to PACE granule timestamps.
        - Matches Argo profiles to PACE granules and assigns corresponding PACE IDs to the profiles.
        - Saves the matched Argo profiles to the specified output file.

    Outputs:
        - A CSV file containing the matched Argo profiles with additional information about 
            the corresponding PACE granules.

    Prints:
        - The number of unique Argo profiles matched to PACE granules.
        - The number of profiles written to the output file.
    """
    # Load up Argo profiles, already cut to PACE dates
    argo_pace = load_orig_argo()

    # Load up PACE granules
    granules, pace = biomass_io.load_granules_from_json(granule_file)

    # Build args for parallel spatial+temporal matching over PACE granules
    argo_lons = argo_pace.lon.values
    argo_lats = argo_pace.lat.values
    argo_times = argo_pace.time
    args_list = [
        (ss, pace.polygon.values[ss], pace.iloc[ss].time,
         argo_lons, argo_lats, argo_times, dtime)
        for ss in range(len(pace))
    ]

    # Parallel check of spatial containment and time window per granule
    n_granules = len(pace)
    all_inside = np.empty((n_granules, len(argo_pace)), dtype=bool)
    ok_times = np.empty((n_granules, len(argo_pace)), dtype=bool)

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(_check_granule, a): a[0] for a in args_list}
        for future in as_completed(futures):
            ss, inside, ok_time = future.result()
            all_inside[ss] = inside
            ok_times[ss] = ok_time

    # Match
    match = all_inside & ok_times
    good_idx = np.where(match)
    pace_idx = good_idx[0]
    argo_idx = good_idx[1]

    print(f'Found {np.unique(argo_idx).size} unique Argo profiles in PACE granules within {dtime}')

    keep = [False]*len(argo_pace)
    # Loop on good_argo indices 
    for jj, idx in enumerate(argo_idx):
        # No need to redo
        if keep[idx]:
            continue

        # Find all matches
        ss = argo_idx == idx
        mt_pace = np.unique(pace_idx[ss])
        ids = ','.join(pace.iloc[mt_pace].id.values)
        #embed(header='81 of argo')
        #
        argo_pace.loc[idx, 'pace_ids'] = ids
        # Keep
        keep[idx] = True

    # Cut down
    argo_matched = argo_pace[keep].copy()
    # Reset index
    argo_matched.reset_index(drop=True, inplace=True)

    # Write
    argo_matched.to_csv(out_file, index=False)
    print(f'Wrote {len(argo_matched)} profiles to {out_file}')

def scan_mbari_profiles(surface:float=20., N_surface:int=3, 
                  MLD:float=200., N_MLD:int=5,
                  argo_path:str=None, outfile:str=None): 
    """ Search for Argo profiles from MBARI processing with sufficient data """

    if argo_path is None:
        argo_path = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'SOCCOM_GO-BGC_HiResQC_LIAR_26Jun2025_netcdf')

    # Grab all .nc files
    all_files = glob.glob(os.path.join(argo_path,'*.nc'))
    # Sort
    all_files = sorted(all_files)

    filenames = []
    cruises = []
    profiles = []
    lats = []
    lons = []
    times = []
    solar_angles = []

    # Loop on em
    for ifile in all_files:
        base_file = os.path.basename(ifile)
        print(f'Examining {base_file}')

        # Load the dataset
        ds = xarray.open_dataset(ifile)

        # Check for bbp
        if 'b_bp700_QF' not in ds.variables:
            continue

        # Loop on profiles
        for iprof in ds.N_PROF.values:
            # Grab the profile
            prof = ds.sel(N_PROF=iprof)

            # QC
            good = prof['b_bp700_QF'].data == b'0'
            if not np.any(good):
                continue

            # Depth
            good_depth = prof.Depth.data[good]

            # Examine
            near_surface = np.sum(good_depth < surface) > N_surface
            if not near_surface:
                continue

            # Do 200m too
            inMLD = np.sum((good_depth > surface) & (good_depth < MLD)) > N_MLD
            if not inMLD:
                continue

            # Keep em!
            filenames.append(base_file)
            cruises.append(str(prof.Cruise.data.astype(str)).strip())
            profiles.append(int(iprof))
            lats.append(float(prof.Lat.data))
            lons.append(float(prof.Lon.data))

            times.append(prof.JULD.values)
            tstamp = pandas.to_datetime(times[-1], utc=True)
            solar_angles.append(
                float(pysolar.solar.get_altitude(lats[-1],
                              lons[-1],
                              tstamp)))
        #embed(header='76 of argo')

    # Generate a DataFrame
    df = pandas.DataFrame({
        'cruise': cruises,
        'filename': filenames,
        'profile': profiles,
        'lat': lats,
        'lon': lons,
        'time': times,
        'solar_angle': solar_angles
    })

    # Write
    if outfile is not None:
        df.to_csv(outfile, index=False)
        print(f'Wrote {len(df)} profiles to {outfile}')
    
    # Return
    return df



def scan_ocean_bio_profiles(data_file:str, surface:float=25., N_surface:int=3, 
                  MLD:float=200., N_MLD:int=5, outfile:str=None): 
    """ Search for Argo profiles from Ocean Biogeochemistry processing with sufficient data """

    filenames = []
    cruises = []
    profiles = []
    lats = []
    lons = []
    times = []
    solar_angles = []

    # Loop on em
    base_file = os.path.basename(data_file)
    print(f'Examining {base_file}')

    # Load the dataset
    ds = xarray.open_dataset(data_file)
    # Check for bbp
    if 'Particle_backscattering_at_700_nm_adjusted__qc' not in ds.variables:
        raise ValueError(f'No bbp data in {data_file}')

    # Loop on unique cruise
    uni_cruises = np.unique(ds.cruise_id.values)

    print(f'Found {len(uni_cruises)} unique cruises')
    print(f'Cruises: {uni_cruises}')

    for cruise in uni_cruises:
        cruise_idx = np.where(ds.cruise_id.values == cruise)[0]

        # Sort by date
        ptimes = ds.date_time.data[cruise_idx]
        srt = np.argsort(ptimes)
        # Indices of this cruise, sorted by time
        cruise_idx = cruise_idx[srt]

        # Loop on profiles
        for iprof, idx in enumerate(cruise_idx):
            prof = ds.isel(N_STATIONS=idx)
            # QC
            good = prof['Particle_backscattering_at_700_nm_adjusted__qc'].data <= 50

            if not np.any(good):
                continue

            # Depth
            good_depth = prof['Pressure_adjusted_'].data[good]

            # Examine
            near_surface = np.sum(good_depth < surface) > N_surface
            if not near_surface:
                continue

            # Do 200m too
            inMLD = np.sum((good_depth > surface) & (good_depth < MLD)) > N_MLD
            if not inMLD:
                continue

            # Keep em!
            filenames.append(base_file)
            cruises.append(str(prof.cruise_id.data.astype(str)).strip())
            # Index from the time sorted set
            profiles.append(int(iprof))
            # Lat, lon
            lats.append(float(prof.latitude.data))
            lons.append(float(prof.longitude.data))

            times.append(prof.date_time.values)
            tstamp = pandas.to_datetime(times[-1], utc=True)
            solar_angles.append(
                float(pysolar.solar.get_altitude(lats[-1],
                              lons[-1],
                              tstamp)))

    # Generate a DataFrame
    df = pandas.DataFrame({
        'cruise': cruises,
        'filename': filenames,
        'profile': profiles,
        'lat': lats,
        'lon': lons,
        'time': times,
        'solar_angle': solar_angles
    })

    # Write
    if outfile is not None:
        df.to_csv(outfile, index=False)
        print(f'Wrote {len(df)} profiles to {outfile}')
    
    # Return
    #embed(header='310 of argo')
    return df


def calc_bbp700_mbari(csv_path:str, argo_dir:str=None, 
    out_path:str=None, surface_depth: float = 25.0) -> pandas.DataFrame:

    print("="*80)
    print("Calculating bbp in the Argo MBARI data")
    if argo_dir is None:
        argo_dir = os.path.join(os.getenv('OS_DATA'), 'Argo',
            'SOCCOM_GO-BGC_HiResQC_LIAR_26Jun2025_netcdf')
            #'SOCCOM_GO-BGC_LoResQC_LIAR_26Jun2025_netcdf')

    # -------------------------
    # load matchup CSV
    # -------------------------
    match = pandas.read_csv(csv_path)

    # -------------------------
    # find MBARI Argo files
    # -------------------------
    files = sorted(set(glob.glob(os.path.join(argo_dir, "*QC.nc")) + 
        glob.glob(os.path.join(argo_dir, "*HRQC.nc"))))
    base_files = [os.path.basename(file) for file in files]

    if len(files) == 0:
        raise ValueError(f"No files found in {argo_dir}")

    rows = []

    #def get_wmo(fname):
    #    m = re.match(r"(\d+)", fname)
    #    return m.group(1) if m else None

    # -------------------------
    # process floats
    # -------------------------

    for ss,f in enumerate(files):

        # Match
        in_match = match.filename == os.path.basename(f)
        if np.sum(in_match) == 0:
            continue

        # Open
        ds = xarray.open_dataset(f)
        bbp = ds["b_bp700"]
        bbp_qc = ds["b_bp700_QF"]
        depth = ds["Depth"]

        # Loop on profiles
        cruise_id = match[in_match].iloc[0].cruise
        profiles = match.profile.values[in_match]


        for pp, profile in enumerate(profiles):

            # Parse
            bbp_prof = bbp.sel(N_PROF=profile).values
            qc_prof = bbp_qc.sel(N_PROF=profile).values
            depth_prof = depth.sel(N_PROF=profile).values

            # QC
            good = qc_prof == b'0'
            if not np.any(good):
                raise ValueError(f"No good values for profile {p}")

            # Depth
            mask = depth_prof <= surface_depth

            mask = mask & good
            if np.sum(mask) < 4:
                raise ValueError(f"Not enough good values for profile {p}")

            # Calculate
            vals = bbp_prof[mask]
            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                med = float(np.median(vals))
                n_used = len(vals)
            else:
                med = np.nan
                n_used = 0

            rows.append([cruise_id, profile, f, med, n_used])

        ds.close()

    # -------------------------
    # save output
    # -------------------------
    df = pandas.DataFrame(
        rows,
        columns=[
            "cruise",
            "profile",
            "file",
            "bbp700_top25m_median",
            "n_values_used"
        ]
    )

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print("Saved to:", out_path)

    #embed(header='463 of argo')
    # Return
    print(f"Processed {len(df)} profiles")
    print("="*80)
    return df


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
def get_var(ds: xarray.Dataset, options: list[str]) -> str:
    """Return the first variable name from *options* that exists in *ds*."""
    for v in options:
        if v in ds:
            return v
    raise KeyError(f"None of {options} found in dataset variables")


def get_dim(ds: xarray.Dataset, options: list[str]) -> str:
    """Return the first dimension name from *options* that exists in *ds*."""
    for d in options:
        if d in ds.dims:
            return d
    raise KeyError(f"None of {options} found in dataset dimensions")


# ================================
# PROFILE PROCESSING
# ================================
def process_profile(
    prof: xarray.Dataset,
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
    csv_df: pandas.DataFrame,
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

    ds = xarray.open_dataset(path)

    var_cruise  = get_var(ds, CRUISE_OPTIONS)
    var_time    = get_var(ds, TIME_OPTIONS)
    var_lat     = get_var(ds, LAT_OPTIONS)
    var_lon     = get_var(ds, LON_OPTIONS)
    dim_profile = get_dim(ds, PROFILE_DIM_OPTIONS)

    n_profiles = ds.sizes[dim_profile]

    meta = pandas.DataFrame({
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
                "profile":              prof_id,
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
def calc_obgc_bbp(
    csv_path: str,
    nc_files: list[str] = None,
    out_path: str = None,
    surface_depth: float = 25.0,
    qc_threshold: int = 50,
    min_surface: int = 3,
    coord_tol: float = 1e-3,
) -> pandas.DataFrame:
    """
    Process a list of BGC-Argo NetCDF files and write matched surface bbp700
    statistics to a CSV.

    Parameters
    ----------
    csv_path : str
        Path to the reference CSV (matched_argo_bgc_profiles_bbp_v3.csv).
    nc_files : list[str]
        Paths to the NetCDF files to process.
    out_path : str, optional
        Destination path for the output CSV.
    surface_depth, qc_threshold, min_surface, coord_tol
        Forwarded to :func:`process_file` / :func:`process_profile`.

    Returns
    -------
    pandas.DataFrame with all matched results (also saved to *out_path*).
        Columns:
        - cruise
        - profile_id
        - time
        - latitude
        - longitude
        - bbp700_top25m_median
        - n_values_used
    """
    print("="*80)
    print("Calculating bbp in the Argo OBGC data")

    csv_df   = pandas.read_csv(csv_path)
    all_rows = []

    # OBGC files
    nc_path = os.path.join(os.getenv('OS_DATA'), 'Argo', 'Med_Mexico')
    if nc_files is None:
        nc_files = [
            os.path.join(nc_path, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_GulfofMexico.nc"),
            os.path.join(nc_path, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_Mediterranean.nc"),
        ]
    
    print(f"Working on files: {nc_files}")

    # Process them
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
    df = pandas.DataFrame(all_rows)

    # Write to disk?
    if out_path is not None:
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to: {out_path}")

    print("="*80)

    return df
