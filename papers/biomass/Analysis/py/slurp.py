"""
Slurping functions for the biomass paper

"""

import os
import numpy as np
import pandas

# OCPY imports
from ocpy.pace import io as pace_io

# BING imports

# Local imports
import fitting as m_fitting
import biomass_io
import ocean_biogeochem2
import bbp700_mbari

from IPython import embed

match_file = 'matched_argo_bgc_profiles_bbp.csv'


def slurp_pace_lat_lon(debug:bool=False):

    # Load data
    matched, granules, pace = biomass_io.load_matched_data()

    PACE_lats = []
    PACE_lons = []

    # Loop me
    for ss in range(len(matched)):
        print(ss)

        imatched = matched.iloc[ss]

        # Fits
        fit_file = biomass_io.get_fit_file_path(imatched)
        try:
            fits = biomass_io.load_fit_data(fit_file)
        except:
            PACE_lats.append(np.nan)
            PACE_lons.append(np.nan)
            continue

        # x,y of closest
        try:
            ix0, iy0 = fits['Rrs_idx'][0]
        except:
            PACE_lats.append(np.nan)
            PACE_lons.append(np.nan)
            continue

        # Save
        PACE_lats.append(fits['lat'][0])
        PACE_lons.append(fits['lon'][0])

    # Add to dataframe
    if debug:
        embed(header='59 of slurp')
    matched['PACE_lat'] = PACE_lats
    matched['PACE_lon'] = PACE_lons

    # Write
    if debug:
        outfile = 'debug_matched_profiles.csv'
        embed(header='64 of slupr')
    else:
        outfile = match_file
    matched.to_csv(outfile, index=False)
    print(f'Wrote {len(matched)} profiles to {outfile}')

def slurp_flh_giop(debug:bool=False):

    # Load data
    matched = pandas.read_csv(match_file)

    # Prep
    FLH_list = []
    bbp700_list = []
    bbp_442_list = []
    bbp_s_list = []
    adg_s_list = []

    # Loop me
    for ss in range(len(matched)):
        print(ss)
        # Fill with np.nan?
        if len(FLH_list) < ss:
            FLH_list.append(np.nan)
        if len(bbp700_list) < ss:
            bbp700_list.append(np.nan)
            bbp_442_list.append(np.nan)
            bbp_s_list.append(np.nan)
            #adg_s_list.append(np.nan)
        if debug and ss>2:
            continue
        imatched = matched.iloc[ss]

        # Fits
        fits_file = m_fitting.set_outfile(imatched)
        try:
            fits = np.load(fits_file)
        except:
            continue

        # x,y of closest
        try:
            ix0, iy0 = fits['Rrs_idx'][0]
        except:
            continue

        # AOP
        aop_file = imatched.closest_file
        aop_file = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP', 
                     os.path.basename(aop_file))
        os.path.exists(aop_file)
        try:
            xds_aop, flags = pace_io.load_oci_l2(aop_file)
        except:
            continue
        FLH = xds_aop['FLH'].data[ix0,iy0]
        FLH_list.append(FLH)

        # GIOP
        iop_file = imatched.closest_file.replace('AOP', 'IOP')
        iop_file = iop_file.replace('V3_0', 'V3_1')
        iop_file
        iop_file = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_IOP', 
                     iop_file)
        try:
            xds_iop, flags = pace_io.load_iop_l2(iop_file)
        except:
            continue
        bbp_442 = xds_iop.bbp_442.data[ix0, iy0]
        bbp_s = xds_iop.bbp_s.data[ix0, iy0]
        bbp_700 =  bbp_442 * (700./442.)**(-1*bbp_s)

        # Save
        bbp700_list.append(bbp_700)
        bbp_442_list.append(bbp_442)
        bbp_s_list.append(bbp_s)
        #adg_s_list.append(xds_iop.adg_s.data[ix0, iy0])

    if debug:
        FLH_list.append(np.nan)
        bbp700_list.append(np.nan)
        bbp_442_list.append(np.nan)
        bbp_s_list.append(np.nan)
        #adg_s_list.append(np.nan)
        embed(header='slurp_flh_giop debug')

    if len(FLH_list) < len(matched):
        FLH_list.append(np.nan)
    if len(bbp700_list) < len(matched):
        bbp700_list.append(np.nan)
        bbp_442_list.append(np.nan)
        bbp_s_list.append(np.nan)

    # Add to dataframe
    matched['FLH'] = FLH_list
    matched['GIOP_bbp_700'] = bbp700_list
    matched['GIOP_bbp_442'] = bbp_442_list
    matched['GIOP_bbp_s'] = bbp_s_list


    # Write
    if debug:
        outfile = 'debug_flh_giop_matched_profiles.csv'
    else:
        outfile = match_file
    matched.to_csv(outfile, index=False)
    print(f'Wrote {len(matched)} profiles to {outfile}')




def slurp_bing_fits(match_file:str, debug:bool=False):
    """
    Processes matched Argo BGC profiles and extracts specific parameters for analysis.

    This function reads a CSV file containing matched Argo BGC profiles, loads corresponding
    data files for each profile, extracts specific parameters (Bnw, beta, and aph), and appends
    these parameters to the original dataset. The updated dataset is then saved back to the same
    CSV file.

    Steps:
    1. Reads the matched Argo BGC profiles from a CSV file.
    2. Iterates through each profile, loading associated data files.
    3. Extracts the median values of Bnw, beta, and aph from the loaded data.
    4. Appends the extracted values to the dataset.
    5. Saves the updated dataset back to the CSV file.



    Args:
        match_file (str): Path to the matched CSV file.
        debug (bool, optional): If True, enables debugging mode with an interactive session. 
                                Default is False.

    Raises:
        FileNotFoundError: If a required data file does not exist.
        KeyError: If the expected keys ('med') are not found in the loaded data.

    Notes:
        - The function assumes the existence of a helper function `get_fit_file_path` to determine
          the output file path for each profile.
        - The function uses the `embed` function for debugging when a file is missing.

    Outputs:
        - Updates the input CSV file with new columns: 'Bnw', 'beta', and 'aph'.
        - Prints the number of profiles written to the file.

    Dependencies:
        - Requires the `pandas` and `numpy` libraries.
        - Assumes the presence of the `get_fit_file_path` and `embed` functions.
    """

    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    beta_vals = []
    Bnw_vals = []
    Bnw_lsig = []
    Bnw_hsig = []
    Bnw_std = []
    aph_vals = []

    nskip = 0

    for ss in range(len(matched)):
        imatched = matched.iloc[ss]
        outfile = biomass_io.get_fit_file_path(imatched)
        print(f'Working on {ss+1}/{len(matched)}: {os.path.basename(outfile)}...')

        # Load
        if not os.path.exists(outfile):
            embed(header=f"303: Missing {outfile}...; ss={ss}")
            raise FileNotFoundError(f"Missing {outfile}...")
        d = np.load(outfile)

        if 'chains' not in d:
            print(f"Skipping {outfile}...")
            beta_vals.append(np.nan)
            Bnw_vals.append(np.nan)
            aph_vals.append(np.nan)
            Bnw_std.append(np.nan)
            Bnw_lsig.append(np.nan)
            Bnw_hsig.append(np.nan)
            nskip += 1
            continue

        #if debug:
        #    embed(header='305 of fitting.py')
        #    return
        # Closest
        Bnw_vals.append(10**d['med'][0,3])
        beta_vals.append(d['med'][0,4])
        aph_vals.append(10**d['med'][0,2])
        # Std
        Bnw_std.append(np.std(10**d['med'][:,3]))
        # Sigma
        Bnw_lsig.append(10**d['med'][0,3] - 10**d['p14'][0,3])
        Bnw_hsig.append(10**d['p86'][0,3] - 10**d['med'][0,3])
        if debug:
            break

    if debug:
        embed(header='275 of slurp.py')
        return

    # Add to matched
    matched['BING_Bnw'] = np.array(Bnw_vals)
    matched['BING_Bnw_std'] = np.array(Bnw_std)
    matched['BING_Bnw_lsig'] = np.array(Bnw_lsig)
    matched['BING_Bnw_hsig'] = np.array(Bnw_hsig)
    matched['BING_beta'] = beta_vals
    matched['BING_aph'] = aph_vals

    # Write
    matched.to_csv(match_file, index=False)
    print(f'Wrote {len(matched)} profiles to {match_file}')

    print(f"Skipped {nskip} profiles")


def add_argo_bbp(match_file:str, argo_dfs:list[pandas.pandas.DataFrame],
            debug:bool=False):

    argo_key = 'argo_bbp700'
    argo_calc_key = 'bbp700_top25m_median'

    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    # Init
    if argo_key not in matched.columns:
        argo_bbp = np.ones(len(matched)) * np.nan
    else:
        argo_bbp = matched[argo_key].values

    ## Loop on argo_df
    for argo_df in argo_dfs:
        for ss in range(len(argo_df)):
            iargo = argo_df.iloc[ss]
            idx = np.where((matched.cruise.values == iargo.cruise) & (
                matched.profile.values == iargo.profile))[0]
            if len(idx) != 1:
                raise ValueError(f"{iargo} not found")
            # Save
            argo_bbp[idx] = iargo[argo_calc_key]
    
    # Update
    assert np.sum(np.isnan(argo_bbp)) == 0
    print("All bbp values have been filled")
    matched[argo_key] = argo_bbp

    # Write
    if not debug:
        matched.to_csv(match_file, index=False)
        print(f'Wrote: {match_file}')
    #else:
    #    embed(header='336 of slurp.py')


# Run it
if __name__ == '__main__':
    
    # PACE FLH and GIOP
    #slurp_flh_giop(debug=False)
    
    # PACE lat, lon of Rrs analysis
    slurp_pace_lat_lon(debug=False)