""" Fit the full L23 dataset as PACE

PACE error derived by JXP
400 - 700nm
5nm resolution

"""

import os
from importlib.resources import files
import numpy as np

import pandas

from oceancolor.hydrolight import loisel23

from boring.models import anw as boring_anw
from boring.models import bbnw as boring_bbnw
from boring.models import utils as model_utils
from boring import inference as boring_inf
from boring import rt as boring_rt
from boring import chisq_fit
from boring.satellites import pace as boring_pace 

import anly_utils 

from IPython import embed

def fit(model_names:list, Nspec:int=None, scl_noise:float=0.02, 
        debug:bool=False): 
    """
    Fits the data with or without considering any errors.
    """
    raise DeprecationWarning("Use fit_l23.py instead")
    # Load L23
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data

    PACE_wave = np.arange(400, 701, 5)
    PACE_error = boring_pace.gen_noise_vector(PACE_wave)

    # Init the models
    models = model_utils.init(model_names, PACE_wave)
    
    # Prep
    if Nspec is None:
        idx = np.arange(ds.Rrs.shape[0])
    else:
        idx = np.arange(Nspec)
    if debug:
        #idx = idx[0:2]
        idx = [170, 180]
        #idx = [2706]

    # Calculate the Rrs and more
    Rrs = []
    varRrs = []
    params = []
    Chls = []
    Ys = []
    for ss in idx:
        odict = anly_utils.prep_l23_data(
            ss, scl_noise=scl_noise, ds=ds)
        # Rrs
        gordon_Rrs = boring_rt.calc_Rrs(odict['a'], odict['bb'])
        # Params
        if models[0].uses_Chl:
            models[0].set_aph(odict['Chl'])
        if models[1].uses_basis_params:  # Lee
            models[1].set_basis_func(odict['Y'])
        # Interpolate
        PACE_Rrs = np.interp(PACE_wave, l23_wave, gordon_Rrs)
        PACE_a = np.interp(PACE_wave, l23_wave, odict['a'])
        PACE_bb = np.interp(PACE_wave, l23_wave, odict['bb'])

        Rrs.append(PACE_Rrs)
        # varRrs
        ivarRrs = PACE_error**2
        varRrs.append(ivarRrs)

        p0_a = models[0].init_guess(PACE_a)
        p0_b = models[1].init_guess(PACE_bb)
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        params.append(p0)
        # Others
        Chls.append(odict['Chl'])
        Ys.append(odict['Y'])
    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file
    outfile = anly_utils.chain_filename(
        model_names, scl_noise, False, PACE=True)

    # Fit
    all_ans = []
    all_cov = []
    all_idx = []
    # Fit
    for item in items:
        if models[0].name == 'ExpBricaud':
            models[0].set_aph(Chls[item[3]])
        try:
            ans, cov, idx = chisq_fit.fit(item, models)
        except RuntimeError:
            print("*****************")
            print(f"Failed on {item[3]}")
            print("*****************")
            ans = np.zeros_like(prev_ans)
            cov = np.zeros_like(prev_cov)
        all_ans.append(ans)
        all_cov.append(cov)
        all_idx.append(idx)
        # Save in case of crash
        prev_ans = ans.copy()
        prev_cov = cov.copy()
    # Save
    outfile = outfile.replace('BORING', 'BORING_LM')
    np.savez(outfile, ans=all_ans, cov=all_cov,
            wave=PACE_wave, obs_Rrs=Rrs, varRrs=varRrs,
            idx=all_idx, Chl=Chls, Y=Ys)
    print(f"Saved: {outfile}")                        


def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50)

    # Full L23 with LM; constant relative error
    if flg == 2:
        fit(['Cst', 'Cst'])
        fit(['Exp', 'Cst'])
        fit(['Exp', 'Pow'])
        fit(['ExpBricaud', 'Pow'])

    # GIOP variants
    if flg == 3:
        # GIOPm : Sdg allowed to vary
        #fit(['ExpBricaud', 'Lee'])
        # GIOP : Sdg fixed
        fit(['GIOP', 'Lee'])

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

    else:
        flg = sys.argv[1]

    main(flg)