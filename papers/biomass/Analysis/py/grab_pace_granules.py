
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
PACE_L2_IOP_PATH = PACE_L2_AOP_PATH.replace('AOP', 'IOP')

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

def download_matched(match_file:str, IOP:bool=False):
    """ Downloads PACE granules matched to Argo profiles from a given CSV file.

    Args:
        match_file (str): Path to the CSV file containing matched Argo profiles and PACE IDs.
        IOP (bool, optional): If True, downloads IOP granules instead of AOP granules. Defaults to False.

    """

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

            url = granule.url
            if IOP: 
                path = PACE_L2_IOP_PATH
                url = url.replace('AOP', 'IOP')
                url = url.replace('V3_0', 'V3_1')
            else:
                path = PACE_L2_AOP_PATH
            outfile = os.path.join(path,
                os.path.basename(url))
            # Check if already downloaded
            if os.path.exists(outfile):
                print(f'Already downloaded {outfile}')
                continue
            # wget
            subprocess.run(['wget', '-O', outfile, url])
    print(f'Downloaded {len(matched)} Argo profiles to {path}')

def find_closest(match_file:str, iRrs:int=38,
                 debug:bool=False, skip_to:int=None):
    """
    Finds the closest PACE granule for each Argo profile in the given match file.

    This function processes a CSV file containing matched Argo profiles and PACE IDs,
    identifies the closest valid PACE granule for each profile based on geospatial distance,
    and appends the results to the CSV file.

    Args:
        match_file (str): Path to the CSV file containing matched Argo profiles and PACE IDs.
        iRrs (int, optional): Index of the Rrs band to use for validation. Defaults to 38.
        debug (bool, optional): If True, processes only the first few rows for debugging. Defaults to False.
        skip_to (int, optional): If provided, skips processing rows until the specified index. Defaults to None.

    Returns:
        None: The function modifies the input CSV file in place by adding columns for the closest granule's
        ID, file name, distance, and time.

    Notes:
        - The function assumes the existence of a JSON file ('PACE_50clouds.json') containing PACE granule data.
        - The function uses the `ocpy_coords.distance_from_latlon` method to calculate distances.
        - The function relies on external modules like `pandas`, `numpy`, and `pace_io` for data processing.
        - If no valid Rrs values are found for a granule, it is skipped.
        - Debug mode allows for quick testing by limiting the number of rows processed.
    """

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

def closest_Rrs(xds, lat_lon:tuple, iRrs:int=38, nclosest:int=1):
    """
    Find the closest valid Rrs (Remote Sensing Reflectance) value to a given latitude and longitude.

    Parameters:
        xds (xarray.Dataset): The dataset containing Rrs data, latitude, and longitude arrays.
        lat_lon (tuple): A tuple containing the target latitude and longitude (lat, lon).
        iRrs (int, optional): The index of the Rrs band to analyze. Defaults to 38.
        nclosest (int, optional): The number of closest valid Rrs values to consider. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - float: The minimum distance(s) to the closest valid Rrs value.
            - tuple: The indices (i, j) of the closest valid Rrs value in the dataset.
    """
    Rrs_ok = xds.Rrs_unc.values[:,:,iRrs] > 0.
    lat_ok = np.isfinite(xds.latitude.values)
    lon_ok = np.isfinite(xds.longitude.values)
    ok_idx = np.where((Rrs_ok & lat_ok & lon_ok).flatten())[0]

    if len(ok_idx) == 0:
        return None, None

    # Find distance
    coords = np.stack((xds.latitude.values.flatten()[ok_idx],
        xds.longitude.values.flatten()[ok_idx]), axis=1)
    d = ocpy_coords.distance_from_latlon(lat_lon, coords)

    # Closest (deal with indices)
    srt = np.argsort(d)
    idx_srt = srt[:nclosest]
    closest_idx = ok_idx[idx_srt]

    # Unravel
    dmin_ij = np.unravel_index(closest_idx, xds.latitude.shape)

    return d[idx_srt], dmin_ij

def load_from_json(json_file:str):
    """
    Load granule data from a JSON file and build a corresponding data table.

    Args:
        json_file (str): Path to the JSON file containing granule data.

    Returns:
        tuple: A tuple containing:
            - granules (dict): The loaded granule data as a dictionary.
            - df (pandas.DataFrame): A DataFrame representing the granule data table,
                with optional antimeridian fixes applied.
    """
    # Load
    granules = ocpy_io.loadjson(json_file)

    # Build the table
    df = rs_ea.build_granule_table(granules, 
                                   fix_antimeridian=True)

    # Return
    return granules, df
        

# Command line
if __name__ == '__main__':

    build = False
    download = True
    closest = False

    if build:
        # Build the JSON file
        build_json(outfile='PACE_50clouds.json', cloud_cover=(0,50))

    # Download nearest granules
    if download:
        # Download nearest granules
        # AOP granules
        #download_matched('matched_argo_bgc_profiles_bbp.csv')
        
        # IOP granules
        download_matched('matched_argo_bgc_profiles_bbp.csv', IOP=True)

    if closest:
        # Find closest granules
        find_closest('matched_argo_bgc_profiles_bbp.csv',
                     debug=False)#, skip_to=799)