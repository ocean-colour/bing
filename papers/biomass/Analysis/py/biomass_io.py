
import pandas
import numpy as np
import os

from ocpy.utils import io as ocpy_io

from remote_sensing.download import earthaccess as rs_ea

#PACE_L2_AOP_PATH = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP')
PACE_L2_AOP_PATH = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP_V3_1')
PACE_L2_IOP_PATH = PACE_L2_AOP_PATH.replace('AOP', 'IOP')
PACE_L1B_PATH = os.path.join(os.getenv('OS_COLOR'),
                              'PACE', 'L1B')
PACE_L1C_PATH = os.path.join(os.getenv('OS_COLOR'),
                              'PACE', 'L1C')

def load_granules_from_json(json_file:str):
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
        

def load_matched_data(match_file='matched_argo_bgc_profiles_bbp_v3.csv', 
                      pace_json='PACE_50clouds.json'):
    """
    Load matched Argo BGC profiles and PACE granules.
    
    Parameters
    ----------
    match_file : str, optional
        Path to CSV file containing matched Argo profiles (default: 'matched_argo_bgc_profiles_bbp.csv')
    pace_json : str, optional
        Path to JSON file containing PACE granule information (default: 'PACE_50clouds.json')
    
    Returns
    -------
    matched : pandas.DataFrame
        DataFrame containing matched Argo profile data
    granules : list
        List of PACE granules
    pace : dict or object
        PACE data structure from JSON
    """
    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)
    
    # Load up PACE granules
    granules, pace = load_granules_from_json(pace_json)
    
    return matched, granules, pace

def get_matched_profile(matched, cruise, profile):
    """
    Get a matched profile from the matched DataFrame.
    
    Parameters
    ----------
    matched : pandas.DataFrame
        The matched DataFrame
    cruise : int
        The cruise number
    profile : int
        The profile number

    Returns
    -------
    pandas.Series
        A row from the matched profiles DataFrame containing 'cruise' and 'profile' fields
    """
    mask = (matched['cruise'] == cruise) & (matched['profile'] == profile)
    if mask.sum() == 0:
        raise ValueError(f"No matched profile found for cruise={cruise}, profile={profile}")
    return matched[mask].iloc[0]

def get_fit_file_path(matched_profile, base_dir=None):
    """
    Construct the path to a fit file for a given matched profile.
    
    Parameters
    ----------
    matched_profile : pandas.Series
        A row from the matched profiles DataFrame containing 'cruise' and 'profile' fields
    base_dir : str, optional
        Base directory for fits. If None, uses OS_COLOR environment variable
    
    Returns
    -------
    str
        Full path to the fit file
    
    Raises
    ------
    AssertionError
        If the fit file does not exist
    """
    if base_dir is None:
        base_dir = os.getenv('OS_COLOR')
    
    fit_file = os.path.join(base_dir, 'Biomass', 'Fits',
                            f'Argo_{matched_profile.cruise}_{matched_profile.profile:03d}_fits.npz')
    #assert os.path.isfile(fit_file), f"Fit file not found: {fit_file}"
    
    return fit_file


def load_fit_data(fit_file):
    """
    Load fit data from an NPZ file.
    
    Parameters
    ----------
    fit_file : str
        Path to the fit file (.npz format)
    
    Returns
    -------
    numpy.lib.npyio.NpzFile
        Loaded fit data containing arrays for 'Rrs', 'Rrs_sig', 'wave', 'chains', 'model_names', etc.
    """
    return np.load(fit_file)