""" Fit the full L23 dataset with BING 2.0 """
import os, sys
import numpy as np


from ocpy.hydrolight import loisel23
from ocpy.satellites import modis as sat_modis
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw
from bing.models import utils as model_utils
from bing import inference as bing_inf
from bing import rt as bing_rt
from bing import chisq_fit
from bing import priors as bing_priors


import anly_utils 

sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20
import dev_fits
import prep_for_fits

import single_bing_fits

from IPython import embed


def main(flg):
    flg = int(flg)

    # ExpBricaud Pow
    if flg == 1:
        p = single_bing_fits.standard_expb_pow()
        batch_fit(p, seed=54321)#, debug=True)

    # GIOP
    if flg == 2:
        p = single_bing_fits.standard_giop()
        batch_fit(p, seed=54321)#, debug=True)

    # k=2b
    if flg == 3:
        p = single_bing_fits.p_k2b()
        batch_fit(p, seed=54321)#, debug=True)

    

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

    else:
        flg = sys.argv[1]

    main(flg)