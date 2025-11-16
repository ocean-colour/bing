"""
PACE slurping

"""

import os
import numpy as np
import pandas
from matplotlib import pyplot as plt

# OCPY imports
from ocpy.utils import plotting
from ocpy.pace import io as pace_io

# BING imports
from bing.parameters import standard
from bing.models import utils as model_utils

# Local imports
from grab_pace_granules import load_from_json
import fitting as m_fitting

match_file = 'matched_argo_bgc_profiles_bbp.csv'

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
        if debug and ss>2:
            break
        imatched = matched.iloc[ss]

        # Fits
        fits_file = m_fitting.set_outfile(imatched)
        fits = np.load(fits_file)

        # x,y of closest
        ix0, iy0 = fits['Rrs_idx'][0]

        # AOP
        aop_file = imatched.closest_file
        aop_file = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP', 
                     os.path.basename(aop_file))
        os.path.exists(aop_file)
        xds_aop, flags = pace_io.load_oci_l2(aop_file)
        FLH = xds_aop['FLH'].data[ix0,iy0]

        # GIOP
        iop_file = imatched.closest_file.replace('AOP', 'IOP')
        iop_file = iop_file.replace('V3_0', 'V3_1')
        iop_file
        iop_file = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_IOP', 
                     iop_file)
        xds_iop, flags = pace_io.load_iop_l2(iop_file)
        bbp_442 = xds_iop.bbp_442[ix0, iy0]
        bbp_s = xds_iop.bbp_s[ix0, iy0]
        bbp_700 =  bbp_442 * (700./442.)**(-1*bbp_s)

        # Save
        FLH_list.append(FLH)
        bbp700_list.append(bbp_700)
        bbp_442_list.append(bbp_442)
        bbp_s_list.append(bbp_s)
        adg_s_list.append(xds_iop.adg_s[ix0, iy0])

    # Add to dataframe
    matched['FLH'] = FLH_list
    matched['GIOP_bbp_700'] = bbp700_list
    matched['GIOP_bbp_442'] = bbp_442_list
    matched['GIOP_bbp_s'] = bbp_s_list


    # Write
    if not debug:
        matched.to_csv(match_file, index=False)
        print(f'Wrote {len(matched)} profiles to {match_file}')

# Run it
if __name__ == '__main__':
    slurp_flh_giop(debug=True)
    