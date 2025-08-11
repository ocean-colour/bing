
import os
import subprocess
import numpy as np

import earthaccess

import pandas

from ocpy.utils import io as ocpy_io
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
    download = True

    if build_json:
        # Build the JSON file
        build_json(outfile='PACE_50clouds.json', cloud_cover=(0,50))

    # Download nearest granules
    if download:
        # Download nearest granules
        download_matched('matched_argo_bgc_profiles_bbp.csv')