""" Fit the full L23 dataset """
import os
import numpy as np


from oceancolor.hydrolight import loisel23

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw
from bing.models import utils as model_utils
from bing import inference as bing_inf
from bing import rt as bing_rt
from bing import chisq_fit

from oceancolor.satellites import modis as sat_modis
from oceancolor.satellites import pace as sat_pace
from oceancolor.satellites import seawifs as sat_seawifs

import anly_utils 

from IPython import embed


def fit(model_names:list, 
        Nspec:int=None, 
        nsteps=80000, nburn=8000,
        use_chisq:bool=False,
        min_wave:float=None,
        max_wave:float=None,
        MODIS:bool=False, PACE:bool=False, SeaWiFS:bool=False,
        scl_noise:float=0.02, add_noise:bool=False,
        n_cores:int=20, debug:bool=False): 
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
    # Load L23
    ds = loisel23.load_ds(4,0)
    gd_wave = np.ones_like(ds.Lambda.data, dtype=bool)
    if max_wave is not None:
        gd_wave &= ds.Lambda.data <= max_wave
    if min_wave is not None:
        gd_wave &= ds.Lambda.data >= min_wave

    wave = ds.Lambda.data[gd_wave]

    # Wavelenegths
    if MODIS:
        model_wave = sat_modis.modis_wave
    elif PACE:
        model_wave = anly_utils.PACE_wave
        PACE_error = sat_pace.gen_noise_vector(model_wave)
    elif SeaWiFS:
        model_wave = sat_seawifs.seawifs_wave
    else:
        model_wave = wave
        
    # Models
    models = model_utils.init(model_names, model_wave)

    # Initialize the MCMC
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)

    # Prep
    if Nspec is None:
        idx = np.arange(ds.Rrs.shape[0])
    else:
        idx = np.arange(Nspec)
    if debug:
        #idx = idx[0:2]
        idx = [170, 180]
        #idx = [2706]

    # Calcualte the Rrs
    Rrs = []
    varRrs = []
    params = []
    Chls = []
    Ys = []
    for ss in idx:
        odict = anly_utils.prep_l23_data(
            ss, scl_noise=scl_noise, ds=ds,
            max_wave=max_wave)
        # Rrs
        gordon_Rrs = bing_rt.calc_Rrs(odict['a'], odict['bb'])
        # Internals
        if models[0].uses_Chl:
            models[0].set_aph(odict['Chl'])
        if models[1].uses_basis_params:  # Lee
            models[1].set_basis_func(odict['Y'])

        # Interpolate
        l23_wave = odict['true_wave']
        model_Rrs = anly_utils.convert_to_satwave(l23_wave, gordon_Rrs, model_wave)
        model_anw = anly_utils.convert_to_satwave(l23_wave, odict['anw'], model_wave)
        model_bbnw = anly_utils.convert_to_satwave(l23_wave, odict['bbnw'], model_wave)

        model_varRrs = anly_utils.scale_noise(scl_noise, model_Rrs, model_wave)

        p0_a = models[0].init_guess(model_anw)
        p0_b = models[1].init_guess(model_bbnw)
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        params.append(p0)
        # Others
        varRrs.append(model_varRrs)
        Rrs.append(model_Rrs)
        Chls.append(odict['Chl'])
        Ys.append(odict['Y'])
    # Arrays
    Rrs = np.array(Rrs)
    params = np.array(params)
    varRrs = np.array(varRrs)

    flags = np.zeros_like(Rrs) # Binary flags for failed fits

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file
    outfile = anly_utils.chain_filename(
        model_names, scl_noise, add_noise, 
        MODIS=MODIS, PACE=PACE, SeaWiFS=SeaWiFS)

    # Fit
    if use_chisq:
        all_ans = []
        all_cov = []
        all_idx = []
        # Fit
        for item in items:
            if models[0].uses_Chl:
                models[0].set_aph(Chls[item[3]])
            if models[1].uses_basis_params:  # Lee
                models[1].set_basis_func(Ys[item[3]])
            try:
                ans, cov, idx = chisq_fit.fit(item, models)
            except RuntimeError:
                print("*****************")
                print(f"Failed on {item[3]}")
                print("*****************")
                ans = np.zeros_like(params[0])
                cov = np.zeros((len(params[0]), len(params[0])))
                idx = item[3]
                flags[idx] += 1 # Failed fit
            if np.any(np.isnan(cov)):
                flags[idx] += 2
            all_ans.append(ans)
            all_cov.append(cov)
            all_idx.append(idx)
            #
            prev_cov = cov
        # Save
        outfile = outfile.replace('BING', 'BING_LM')
        #embed(header='165 of fits')
        np.savez(outfile, ans=all_ans, cov=all_cov,
              wave=model_wave, obs_Rrs=Rrs, varRrs=varRrs,
              idx=all_idx, Chl=Chls, Y=Ys, flags=flags)
    else:
        embed(header='fit 116; need to deal with Chl')
        all_samples, all_idx = big_inf.fit_batch(
            models, pdict, items, n_cores=n_cores)
        # Save
        anly_utils.save_fits(all_samples, all_idx, outfile,
                         extras=dict(Rrs=Rrs, varRrs=varRrs))
    print(f"Saved: {outfile}")                        


def main(flg):
    flg = int(flg)

    MODIS = False
    PACE = False
    SeaWiFS = False
    scl_noise = 0.02

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50, nsteps=10000, nburn=1000)

    # Full L23
    if flg == 2:
        fit(['Exp', 'Pow'], nsteps=50000, nburn=5000)

    # Full L23 with LM; constant relative error
    if flg == 3:
        fit(['Cst', 'Cst'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['Exp', 'Cst'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['Exp', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['ExpBricaud', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['ExpNMF', 'Pow'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['GIOP', 'Lee'], use_chisq=True, max_wave=700., min_wave=400.)
        fit(['GSM', 'GSM'], use_chisq=True, max_wave=700., min_wave=400.)


    # MODIS
    if flg in [4,7]:
        MODIS = True

    # All models
    if flg in [5,8]:
        PACE = True

    if flg in [6,9]:
        SeaWiFS = True

    if flg in [7,8,9]:
        if MODIS:
            scl_noise = 'MODIS_Aqua'
        elif SeaWiFS:
            scl_noise = 'SeaWiFS'
        elif PACE:
            scl_noise = 'SPACEeaWiFS'

    #embed(header='main 168')
    if flg in [4,5,6,7,8,9]:
        fit(['Cst', 'Cst'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['Exp', 'Cst'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['Exp', 'Pow'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['ExpBricaud', 'Pow'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['ExpNMF', 'Pow'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['GIOP', 'Pow'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['GIOP', 'Lee'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)
        fit(['GSM', 'GSM'], use_chisq=True, PACE=PACE, SeaWiFS=SeaWiFS, MODIS=MODIS, scl_noise=scl_noise)

    

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Testing
        #flg += 2 ** 1  # 2 -- No priors
        #flg += 2 ** 2  # 4 -- bb_water

    else:
        flg = sys.argv[1]

    main(flg)