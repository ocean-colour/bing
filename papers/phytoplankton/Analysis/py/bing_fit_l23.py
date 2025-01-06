""" Fit the full L23 dataset """
import os
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

from IPython import embed


def fit(p,
        Nspec:int=None, 
        nsteps=80000, nburn=8000,
        use_chisq:bool=False,
        reduce_by_in_situ:float=None,
        n_cores:int=20, debug:bool=False,
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
    if Nspec is None:
        idx = np.arange(ds.Rrs.shape[0])
    else:
        idx = np.arange(Nspec)
    if debug:
        #idx = idx[0:2]
        idx = [170, 180]
        #idx = [2706]

    # Setup the model

    # Wavelenegths
    if p.satellite == 'MODIS':
        model_wave = sat_modis.modis_wave
    elif p.satellite == 'PACE':
        model_wave = anly_utils_20.pace_wave(wv_min=p.wv_min,
                                             wv_max=p.wv_max)
    elif p.satellite == 'SeaWiFS':
        model_wave = sat_seawifs.seawifs_wave
    else:
        raise IOError("Satellite not recognized")

    # Wavelengths
    i400 = np.argmin(np.abs(model_wave-400))
    i440 = np.argmin(np.abs(model_wave-440))

    # Priors
    if p.model_names[0] == 'ExpB':
        use_model_names = ['Exp', p.model_names[1]]
    else:
        use_model_names = p.model_names.copy()

    # Models
    models = model_utils.init(use_model_names, model_wave)

    # Set priors
    bing_priors.set_standard_priors(models, p)

    # Initialize the MCMC
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)

    # Calcualte the Rrs
    Rrs = []
    varRrs = []
    params = []
    Chls = []
    Ys = []
    for ss in idx:
        odict = anly_utils_20.prep_l23_data(
            ss, wv_min=p.wv_min, wv_max=p.wv_max)
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

        # Noise
        model_varRrs = anly_utils.scale_noise(
            p.scl_noise, model_Rrs, model_wave,
            reduce_by_in_situ=reduce_by_in_situ)

        # Add noise?
        if p.add_noise:
            model_Rrs = anly_utils.add_noise(
                model_Rrs, abs_sig=np.sqrt(model_varRrs))

        p0_a = models[0].init_guess(model_anw)
        p0_b = models[1].init_guess(model_bbnw)
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                         np.log10(np.atleast_1d(p0_b))))
        # Deal with S
        if models[0].name in ['Exp', 'ExpBricaud', 'ExpBricaudFix']:
            p0[1] = 10**p0[1]
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

    flags = np.zeros_like(Rrs, dtype=int) # Binary flags for failed fits

    # Build the items
    items = [(Rrs[i], varRrs[i], params[i], i) for i in idx]

    # Output file
    outfile = anly_utils_20.chain_filename(p, idx=idx)
    embed(header='fit 158')

    all_samples, all_idx = big_inf.fit_batch(
        models, pdict, items, n_cores=n_cores)
    # Save
    anly_utils.save_fits(all_samples, all_idx, outfile,
                        extras=dict(Rrs=Rrs, varRrs=varRrs))


def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        fit(['Exp', 'Pow'], Nspec=50, nsteps=10000, nburn=1000)


    

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