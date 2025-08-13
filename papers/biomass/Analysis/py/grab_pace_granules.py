
import os
import subprocess
import numpy as np

import earthaccess

import pandas

from ocpy.utils import io as ocpy_io
from ocpy.pace import io as pace_io
from ocpy.utils import coords as ocpy_coords

from remote_sensing.download import earthaccess as rs_ea

from IPython import embed

# PACE Granule path
PACE_L2_AOP_PATH = os.path.join(os.getenv('OS_COLOR'),
                                 'PACE',
                                 'L2_AOP')

def build_json(outfile:str='PACE_50clouds.json', cloud_cover=(0,50)):
    """
    Fetches granules from the PACE dataset with specified cloud cover constraints,
    then saves the results to a JSON file.

    Args:
        outfile (str): The name of the output JSON file where the granules will be saved.
                        Defaults to 'PACE_50clouds.json'.
        cloud_cover (tuple): A tuple specifying the range of acceptable cloud cover percentages
                                (min, max). Defaults to (0, 50).

    Returns:
        None

    Side Effects:
        - Authenticates the user using the `earthaccess` library.
        - Searches for granules in the PACE dataset with the specified cloud cover range.
        - Converts the granules to a dictionary format.
        - Saves the dictionary as a JSON file to the specified output file.
        - Prints the number of granules written to the output file.
    """

    # Authorize
    auth = earthaccess.login(persist=True)

    # Grab em
    all_results = earthaccess.search_data(
        short_name="PACE_OCI_L2_AOP",
        cloud_cover=cloud_cover,
    )

    # Generat dict
    full_dict = rs_ea.granules_to_dict(all_results)

    # JSON
    jdict = ocpy_io.jsonify(full_dict)
    ocpy_io.savejson(outfile, jdict)
    print(f'Wrote {len(full_dict)} granules to {outfile}')

def download_matched(match_file:str):

    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    # Load up PACE granules
    granules, pace = load_from_json('PACE_50clouds.json')

    # Loop on Argo profiles
    for irow in range(len(matched)):
        row = matched.iloc[irow]

        # Get the PACE IDs
        pace_ids = row['pace_ids'].split(',')

        # Find the granules
        for pace_id in pace_ids:
            ss = np.where(pace.id == pace_id)[0][0]
            granule = pace.iloc[ss]
            #embed(header=f'Granule for {pace_id}')

            outfile = os.path.join(PACE_L2_AOP_PATH, 
                os.path.basename(granule.url))
            # Check if already downloaded
            if os.path.exists(outfile):
                print(f'Already downloaded {outfile}')
                continue
            # wget
            subprocess.run(['wget', '-O', outfile, granule.url])
    print(f'Downloaded {len(matched)} Argo profiles to {PACE_L2_AOP_PATH}')        

def find_closest(match_file:str, iRrs:int=38,
                 debug:bool=False, skip_to:int=None):

    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    # Load up PACE granules
    granules, pace = load_from_json('PACE_50clouds.json')

    # Items to add to the table
    sv_ids = []
    sv_dist = []
    sv_time = []
    sv_file = []

    # Loop on Argo profiles
    for irow in range(len(matched)):
        row = matched.iloc[irow]
        if skip_to is not None and irow < (skip_to-1):
            continue

        # Get the PACE IDs
        pace_ids = row['pace_ids'].split(',')

        mind = 1e9
        # Find the granules
        for jj, pace_id in enumerate(pace_ids):
            print(f'Processing {irow+1}/{len(matched)}: {pace_id} ({jj+1}/{len(pace_ids)})')
            ss = np.where(pace.id == pace_id)[0][0]
            granule = pace.iloc[ss]
            #embed(header=f'Granule for {pace_id}')

            pace_file = os.path.join(PACE_L2_AOP_PATH,
                os.path.basename(granule.url))

            # Load up
            xds, flags = pace_io.load_oci_l2(pace_file)
            Rrs_ok = xds.Rrs_unc.values[:,:,iRrs] > 0.
            if not np.any(Rrs_ok):
                print(f'No valid Rrs found in {pace_file}, skipping')
                continue

            # Closest good Rrs
            coords = np.stack((xds.latitude.values.flatten(), 
                   xds.longitude.values.flatten()), axis=1)
            d = ocpy_coords.distance_from_latlon((
                row.lat, row.lon), coords)
            dmin = d[Rrs_ok.flatten()].min()
            # 
            if dmin < mind:
                mind = dmin
                best_g = granule

        # Save best
        if mind == 1e9:
            # This is a hack, but distance is very large
            best_g = granule
        sv_ids.append(best_g.id)
        sv_file.append(os.path.basename(best_g.url))
        sv_time.append(best_g.time)

        sv_dist.append(mind)
        #embed(header='133 of grab')

        # Debug?
        if debug and irow > 4:
            break

    if skip_to is not None:
        return

    # Debug?
    if debug:
        embed(header='152 of grab')
        print(f'Debug mode, only processed {irow+1} of {len(matched)}')
        cut = np.array([False]*len(matched))
        cut[:irow+1] = True
        matched = matched[cut].copy()
        matched.reset_index(drop=True, inplace=True)

    # Add to table
    #embed(header='160 of grab')
    matched['closest_id'] = sv_ids
    matched['closest_file'] = sv_file
    matched['closest_dist_km'] = sv_dist
    matched['closest_time'] = sv_time

    # Write
    matched.to_csv(match_file, index=False)

def closest_Rrs(xds, lat_lon:tuple, iRrs:int=38):

    Rrs_ok = xds.Rrs_unc.values[:,:,iRrs] > 0.
    coords = np.stack((xds.latitude.values.flatten(), 
        xds.longitude.values.flatten()), axis=1)
    d = ocpy_coords.distance_from_latlon(lat_lon, coords)

    # Unravel
    imin = np.argmin(d)
    dmin_ij = np.unravel_index(imin, xds.latitude.shape)

    return d.min(), dmin_ij

def load_from_json(json_file:str):
    # Load
    granules = ocpy_io.loadjson(json_file)

    # Build the table
    df = rs_ea.build_granule_table(granules, 
                                   fix_antimeridian=True)

    # Return
    return granules, df
        

# Command line
if __name__ == '__main__':

    build_json = False
    download = False
    closest = True

    if build_json:
        # Build the JSON file
        build_json(outfile='PACE_50clouds.json', cloud_cover=(0,50))

    # Download nearest granules
    if download:
        # Download nearest granules
        download_matched('matched_argo_bgc_profiles_bbp.csv')

    if closest:
        # Find closest granules
        find_closest('matched_argo_bgc_profiles_bbp.csv',
                     debug=False)#, skip_to=799)