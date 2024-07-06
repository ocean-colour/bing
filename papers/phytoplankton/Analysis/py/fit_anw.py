""" Run BING on anw data """

import os

import numpy as np

from matplotlib import pyplot as plt

from functools import partial
from scipy.optimize import curve_fit

from bing.models import anw as bing_anw
from bing.models import bbnw as bing_bbnw
from bing.models import utils as model_utils
from bing import inference as bing_inf
from bing import rt as bing_rt
from bing import chisq_fit
from bing import plotting as bing_plot
from bing import priors as bing_priors

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs

from IPython import embed

import anly_utils 

def fit(model_name:str, idx:int, outfile:str,
        nsteps:int=10000, nburn:int=1000, 
        scl_noise:float=0.02, use_chisq:bool=False,
        add_noise:bool=False,
        max_wave:float=None,
        chk_guess:bool=False,
        show:bool=False,
        MODIS:bool=False,
        SeaWiFS:bool=False,
        PACE:bool=False):

    odict = anly_utils.prep_l23_data(idx, scl_noise=scl_noise,
                                     max_wave=max_wave)

    # Unpack
    wave = odict['wave']
    l23_wave = odict['true_wave']
    anw_true = odict['anw']

    # Wavelenegths
    if MODIS:
        model_wave = sat_modis.modis_wave
    elif PACE:
        model_wave = sat_pace.pace_wave
        PACE_error = sat_pace.gen_noise_vector(PACE_wave)
    elif SeaWiFS:
        model_wave = sat_seawifs.seawifs_wave
    else:
        model_wave = wave

    # Model
    model = bing_anw.init_model(model_name, model_wave)

    # Set priors
    if not use_chisq:
        prior_dict = dict(flavor='uniform', pmin=-6, pmax=5)
        prior_dicts = [prior_dict]*model.nparam
        # Special cases
        if model_name == 'ExpB':
            prior_dicts[1] = dict(flavor='uniform', 
                                    pmin=np.log10(0.007), 
                                    pmax=np.log10(0.02))
        model.priors = bing_priors.Priors(prior_dicts)
                    
    # Initialize the MCMC
    if not use_chisq:
        raise ValueError("Not ready for this")
        pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)
    
    # Gordon Rrs
    gordon_Rrs = bing_rt.calc_Rrs(odict['a'], odict['bb'])

    # Internals
    if model.uses_Chl:
        model.set_aph(odict['Chl'])

    # Bricaud?
    # Interpolate
    model_Rrs = anly_utils.convert_to_satwave(l23_wave, gordon_Rrs, model_wave)
    model_anw = anly_utils.convert_to_satwave(l23_wave, odict['anw'], model_wave)
    model_bbnw = anly_utils.convert_to_satwave(l23_wave, odict['bbnw'], model_wave)

    model_varRrs = anly_utils.scale_noise(scl_noise, model_Rrs, model_wave)

    if add_noise:
        model_Rrs = anly_utils.add_noise(
                model_Rrs, abs_sig=np.sqrt(model_varRrs))

    # Initial guess
    p0_a = model.init_guess(model_anw)
    p0 = p0_a


    def show_fit(p0):
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(model_wave, model.eval_anw(p0).flatten(), 'b-', label='Guess')
        ax.plot(model_wave, model_anw, 'k-', label='True')
        ax.legend()
        plt.show()
        
    # Check
    if chk_guess:
        show_fit(p0)

    # Set the items
    items = [(model_anw, model_varRrs, p0, idx)]

    # Bayes
    if not use_chisq:
        raise ValueError("Not ready for this")
        prior_dict = dict(flavor='uniform', pmin=-6, pmax=5)

        for jj in range(2):
            models[jj].priors = bing_priors.Priors(
                [prior_dict]*models[jj].nparam)

        # Fit
        chains, idx = bing_inf.fit_one(items[0], models=models, pdict=pdict, chains_only=True)

        # Save
        anly_utils.save_fits(chains, idx, outfile, 
                             extras=dict(wave=model_wave, 
                                         obs_Rrs=model_Rrs, 
                                         varRrs=model_varRrs, 
                                         Chl=odict['Chl'], 
                                         Y=odict['Y']))
    else: # chi^2

        def fit_func(wave, *params, model=None):
            anw = model.eval_anw(np.array(params))
            return anw.flatten()
        partial_func = partial(fit_func, model=model)

        # Fit
        bounds = model.priors.gen_bounds()
        ans, cov =  curve_fit(partial_func, None, 
                          model_anw, p0=p0, #sigma=0.05,
                          full_output=False, bounds=bounds,
                          maxfev=10000)
        # Show?
        if show:
            show_fit(ans)

        # Save
        if outfile is not None:
            np.savez(outfile, ans=ans, cov=cov,
                wave=wave, obs_anw=model_anw,
                Chl=odict['Chl'], Y=odict['Y'])
            print(f"Saved: {outfile}")
        #
        return model, model_anw, p0, ans, cov



def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        odict = fit('Chase2017', 170, 'fitanw_170_Chase2017.npz',
                    show=True, use_chisq=True)


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