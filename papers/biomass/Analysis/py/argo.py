""" Routines related to Argo """

import os
import glob
import datetime

import numpy as np

import xarray
import pandas

import pysolar

from IPython import embed

def scan_profiles(surface:float=20., N_surface:int=3, MLD:float=200., N_MLD:int=5):
    """ Search for Argo profiles with sufficient data """
    argo_path = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'SOCCOM_GO-BGC_HiResQC_LIAR_26Jun2025_netcdf')

    # Grab all .nc files
    all_files = glob.glob(os.path.join(argo_path,'*.nc'))

    filenames = []
    IDs = []
    profiles = []
    lats = []
    lons = []
    times = []
    solar_angles = []

    # Loop on em
    for ifile in all_files:
        base_file = os.path.basename(ifile)
        # Load the dataset
        ds = xarray.open_dataset(ifile)

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
            #IDs.append(prof.ARGO_PROFILE.values)
            profiles.append(int(iprof))
            lats.append(float(prof.Lat))
            lons.append(float(prof.Lon))

            times.append(prof.JULD.values)
            tstamp = pandas.to_datetime(times[-1], utc=True)
            solar_angles.append(
                pysolar.solar.get_altitude(lats[-1],
                              lons[-1],
                              tstamp)
            embed(header='Found a good profile!')



if __name__ == '__main__':

    # Scan Argo profiles
    scan_profiles()
            
