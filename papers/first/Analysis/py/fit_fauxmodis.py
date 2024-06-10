""" Fit the full L23 dataset as MODIS 

MODIS error
   https://oceancolor.gsfc.nasa.gov/resources/atbd/rrs/#sec_4

"""

import os
import numpy as np

from oceancolor.hydrolight import loisel23

from boring.models import anw as boring_anw
from boring.models import bbnw as boring_bbnw
from boring import inference as boring_inf
from boring import rt as boring_rt
from boring import chisq_fit
from boring.satellites import modis as boring_modis

import anly_utils 

from IPython import embed

def fit(model_names:list, Nspec:int=None, 
        prior_approach:str='log', scl_noise:float=0.02, 
        debug:bool=False): 
    """
    Fits the data with or without considering any errors.
    """
    # Load L23
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data

    # MODIS wavelengths
    modis_wave = boring_modis.modis_wave #[412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748]# , 859, 869] # nm
    modis_wave = np.array(modis_wave)

    # Init the models
    anw_model = boring_anw.init_model(model_names[0], modis_wave)
    bbnw_model = boring_bbnw.init_model(model_names[1], modis_wave)
    models = [anw_model, bbnw_model]
    
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
    for ss in idx:
        odict = anly_utils.prep_l23_data(
            ss, scl_noise=scl_noise, ds=ds)
        # Rrs
        gordon_Rrs = boring_rt.calc_Rrs(odict['a'], odict['bb'])
        # Params
        if models[0].name in ['ExpBricaud', 'GIOP']:
            models[0].set_aph(odict['Chl'])
        # Interpolate
        modis_Rrs = np.interp(modis_wave, l23_wave, gordon_Rrs)
        modis_a = np.interp(modis_wave, l23_wave, odict['a'])
        modis_bb = np.interp(modis_wave, l23_wave, odict['bb'])

        Rrs.append(modis_Rrs)
        # varRrs
        ivarRrs = (scl_noise * modis_Rrs)**2
        varRrs.append(ivarRrs)

        p0_a = anw_model.init_guess(modis_a)
        p0_b = bbnw_model.init_guess(modis_bb)
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        params.append(p0)
        # Others
        Chls.append(odict['Chl'])
    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file
    outfile = anly_utils.chain_filename(
        model_names, scl_noise, False, MODIS=True)

    # Fit
    all_ans = []
    all_cov = []
    all_idx = []
    # Fit
    for item in items:
        if models[0].name in ['ExpBricaud', 'GIOP']:
            models[0].set_aph(odict['Chl'])
        if models[1].name == 'Lee':
            models[1].set_Y(odict['Y'])
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
            wave=modis_wave, obs_Rrs=Rrs, varRrs=varRrs,
            idx=all_idx, Chl=Chls)
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
        fit(['GIOP', 'Lee'])
        fit(['ExpFix', 'Lee'])

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

    else:
        flg = sys.argv[1]

    main(flg)