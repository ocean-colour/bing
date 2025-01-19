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


def batch_fit(p, n_batch:int=5, n_cores:int=15, debug:bool=False,
        seed:bool=None): 
    """
    Fits the data with or without considering any errors.

    Args:
        edict (dict): A dictionary containing the necessary information for fitting.
        Nspec (int): The number of spectra to fit. Default is None = all
        abs_sig (float): The absolute value of the error to consider. Default is None.
            if None, use no error!
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.
        max_wv (float): The maximum wavelength to consider. Default is None.
        use_log_ab (bool): Whether to use log(ab) in the priors. Default is False.
        use_NMF_pos (bool): Whether to use positive priors for NMF. Default is False.

    """
    if seed is not None:
        np.random.seed(seed)

    # Load L23
    ds = loisel23.load_ds(4,0)
    # Prep
    all_idx = np.arange(ds.Rrs.shape[0]).tolist()
    if debug:
        #idx = idx[0:2]
        all_idx = [170, 180, 200, 250]
        #idx = [2706]

    Rrs = []
    varRrs = []
    params = []
    Chls = []
    Ys = []
    for idx in all_idx:
        if idx % 50 == 0:
            print("Working on {:d}".format(idx))
        prep_dict = prep_for_fits.one_l23(p, idx)

        # Others
        varRrs.append(prep_dict['model_varRrs'])
        Rrs.append(prep_dict['model_Rrs'])
        Chls.append(prep_dict['odict']['Chl'])
        Ys.append(prep_dict['odict']['Y'])
        params.append(prep_dict['p0'])

    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Add to pdict
    pdict = prep_dict['pdict']
    max_idx = np.max(all_idx)
    # Brutal kludge for multi-processing
    pdict['Chl'] = np.zeros(max_idx+1)
    pdict['Y'] = np.zeros(max_idx+1)
    for ss, idx in enumerate(all_idx):
        pdict['Chl'][idx] = Chls[ss]
        pdict['Y'][idx] = Ys[ss]

    # Models
    models = prep_dict['models']

    i0 = 0
    while i0 < len(all_idx):
        # Batch
        i1 = i0 + n_batch*n_cores
        i1 = min(i1, len(all_idx))
        print("Working on {:d} to {:d}".format(i0, i1))

        # Build the items
        items = []
        for idx in all_idx[i0:i1]:
            ss = all_idx.index(idx)
            item = (Rrs[ss], varRrs[ss], params[ss], idx)
            items.append(item)

        # Fit
        all_samples, sub_idx = bing_inf.fit_batch(
            models, pdict, items, n_cores=n_cores)

        # Check
        assert np.all([item[3] for item in items] == sub_idx)

        # Output
        for ss, item in enumerate(items):
            # Unpack
            iRrs, ivarRrs, iparams, idx = item
            chains = all_samples[ss]
            outfile = anly_utils_20.chain_filename(p, idx=idx)
            anly_utils_20.save_fits(chains, idx, outfile, 
                            extras=dict(wave=models[0].wave, 
                                        obs_Rrs=iRrs, 
                                        varRrs=ivarRrs, 
                                        Chl=pdict['Chl'][idx],
                                        Y=pdict['Y'][idx]))

        # Update i0
        i0 = i1

    print("Done!")



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

    

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

    else:
        flg = sys.argv[1]

    main(flg)