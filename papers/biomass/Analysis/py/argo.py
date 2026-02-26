""" Routines related to Argo """

import os
import glob
import datetime

import numpy as np

import xarray
import pandas

import pysolar

from shapely.vectorized import contains

# Local imports
import grab_pace_granules
import gpolygons

from IPython import embed



def load_orig_argo(csv_file:str='argo_bgc_profiles_bbp.csv',
              cut_for_pace:bool=True):
    df = pandas.read_csv(csv_file)
    df['time'] = pandas.to_datetime(df.time.values, utc=True)

    # Cut for PACE?
    if cut_for_pace:
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

def match_argo_to_pace(out_file:str, dtime:str='1 day'):
    """
    Matches Argo profiles to PACE granules based on spatial and temporal criteria.

    Parameters:
        out_file (str): The file path where the matched Argo profiles will be saved as a CSV.
        dtime (str, optional): The time window for matching Argo profiles to PACE granules. 
                                Defaults to '1 day'. The format should be compatible with 
                                pandas.Timedelta.

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
    granules, pace = grab_pace_granules.load_from_json('PACE_50clouds.json')

    # Check if in PACE granule
    all_inside = []
    for ss in range(len(pace)):
        inside = contains(pace.polygon.values[ss], argo_pace.lon, argo_pace.lat)
        all_inside.append(np.array(inside))
    all_inside = np.array(all_inside)

    # Time window
    ok_times = []
    for ss in range(len(pace)):
        dt = pandas.Timestamp(pace.iloc[ss].time) - argo_pace.time
        good_dt = np.abs(dt) < pandas.Timedelta(dtime)
        ok_times.append(np.array(good_dt))
    ok_times = np.array(ok_times)

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

def scan_profiles(surface:float=20., N_surface:int=3, 
                  MLD:float=200., N_MLD:int=5,
                  argo_path:str=None):

    """ Search for Argo profiles with sufficient data """
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
            good = prof.b_bp700_QF.data == b'0'
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
            lats.append(float(prof.Lat))
            lons.append(float(prof.Lon))

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
    outfile = 'argo_bgc_profiles_bbp.csv'
    df.to_csv(outfile, index=False)
    print(f'Wrote {len(df)} profiles to {outfile}')


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
