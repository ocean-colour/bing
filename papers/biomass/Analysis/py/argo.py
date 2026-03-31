""" Routines related to Argo """

import os
import glob
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
            profiles.append(int(iprof))
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

'''
NOW RUN FROM end_to_end_workflow.py

if __name__ == '__main__':

    scan = False
    match = True

    # Scan Argo profiles
    #https://library.ucsd.edu/dc/object/bb1310816p
    if scan:
        argo_path = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'SOCCOM_GO-BGC_LoResQC_LIAR_26Jun2025_netcdf')
        scan_profiles(argo_path=argo_path)
            

    # Match
    if match:
        out_file='matched_argo_bgc_profiles_bbp.csv'
        match_argo_to_pace(out_file, dtime='1 day')
'''
