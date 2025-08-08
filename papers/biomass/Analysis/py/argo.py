""" Routines related to Argo """

import os
import glob
import datetime

import numpy as np

import xarray
import pandas

import pysolar

from IPython import embed

def load_argo(csv_file='argo_bgc_profiles_bbp.csv',
              cut_for_pace:bool=True):
    df = pandas.read_csv('argo_bgc_profiles_bbp.csv')
    df['time'] = pandas.to_datetime(df.time.values, utc=True)

    # Cut for PACE?
    if cut_for_pace:
        old_enough = df.time > pandas.Timestamp('2024-04-01', tz='UTC')
        df = df[old_enough].copy()

    # Deal with lon
    lons = df.lon.values
    lons[lons > 180] -= 360
    df['lon'] = lons

    # Return
    return df


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

    # Scan Argo profiles
    #https://library.ucsd.edu/dc/object/bb1310816p
    argo_path = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'SOCCOM_GO-BGC_LoResQC_LIAR_26Jun2025_netcdf')
    scan_profiles(argo_path=argo_path)
            
