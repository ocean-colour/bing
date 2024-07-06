""" Run BING on anw data """

import os

import numpy as np

from matplotlib import pyplot as plt

from functools import partial
from scipy.optimize import curve_fit
import emcee

from bing.models import anw as bing_anw
from bing import priors as bing_priors

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs

from IPython import embed

import anly_utils 

def fit_one(items:list, model=None, pdict:dict=None, chains_only:bool=False):
    """
    Fits a model to a set of input data using the MCMC algorithm.

    Args:
        models (list): The list of model objects, a_nw, bb_nw
        items (list): A list containing the 
            Rrs (numpy.ndarray): The reflectance data.
            varRrs (numpy.ndarray): The variance of the reflectance data.
            params (numpy.ndarray): The initial guess for the parameters.
            idx (int): The index of the item.
        pdict (dict, optional): A dictionary containing the model and fitting parameters. Defaults to None.
        chains_only (bool, optional): If True, only the chains are returned. Defaults to False.

    Returns:
        tuple: A tuple containing the MCMC sampler object and the index.
    """
    # Unpack
    anw, var, params, idx = items

    # Run
    print(f"idx={idx}")
    sampler = run_emcee(
        model, anw, var, pdict['ndim'],
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        nburn=pdict['nburn'],
        skip_check=True,
        p0=params,
        save_file=pdict['save_file'])

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def init_mcmc(model, nsteps:int=10000, nburn:int=1000):
    """
    Initializes the MCMC parameters.

    Args:
        emulator: The emulator model.
        ndim (int): The number of dimensions.
        nsteps (int, optional): The number of steps to run the sampler. Defaults to 10000.
        nburn (int, optional): The number of steps to run the burn-in. Defaults to 1000.

    Returns:
        dict: A dictionary containing the MCMC parameters.
    """
    pdict = {}
    ndim = model.nparam 
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['ndim'] = ndim
    pdict['nsteps'] = nsteps
    pdict['nburn'] = nburn
    pdict['save_file'] = None
    #
    return pdict


def log_prob(params, model, anw:np.ndarray, var:np.ndarray):
    """
    Calculate the logarithm of the probability of the given parameters.

    Args:
        params (array-like): The parameters to be used in the model prediction.
        model (str): The model name
        Rs (array-like): The observed values.

    Returns:
        float: The logarithm of the probability.
    """
    # Unpack for convenience
    aparams = params[:model.nparam]

    # Priors
    a_prior = model.priors.calc(aparams)

    if np.any(np.isneginf(a_prior)):
        return -np.inf

    # Proceed
    fit_anw = model.eval_anw(aparams).flatten()

    # Evaluate
    eeval = (fit_anw-anw)**2 / var
    # Finish
    prob = -0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob + a_prior 

def run_emcee(model, anw, var, ndim, nwalkers:int=32, 
              nburn:int=1000,
              nsteps:int=20000, save_file:str=None, 
              p0=None, skip_check:bool=False):
    """
    Run the emcee sampler for Bayesian inference.

    Args:
        models (list): The list of model objects, a_nw, bb_nw
        Rrs (numpy.ndarray): The input data.
        varRrs (numpy.ndarray): The error data.
        nwalkers (int, optional): The number of walkers in the ensemble. Defaults to 32.
        nsteps (int, optional): The number of steps to run the sampler. Defaults to 20000.
        save_file (str, optional): The file path to save the backend. Defaults to None.
        p0 (numpy.ndarray, optional): The initial positions of the walkers. Defaults to None.
        skip_check (bool, optional): Whether to skip the initial state check. Defaults to False.

    Returns:
        emcee.EnsembleSampler: The emcee sampler object.
    """

    # Initialize
    if p0 is None:
        raise ValueError("Must provide p0")
    p0 = np.tile(p0, (nwalkers, 1))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    # Init
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=[model, anw, var],
        backend=backend)#, pool=pool)

    # Burn in
    print("Running burn-in")
    #embed(header='159 of fit')
    state = sampler.run_mcmc(p0, nburn,
        skip_initial_state_check=skip_check,
        progress=True)
    sampler.reset()

    # Run
    print("Running full model")
    sampler.run_mcmc(state, nsteps,
        skip_initial_state_check=skip_check,
        progress=True)

    if save_file is not None:
        print(f"All done: Wrote {save_file}")

    # Return
    return sampler

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
        if model_name == 'Chase2017':
            pass
        else:
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
        pdict = init_mcmc(model, nsteps=nsteps, nburn=nburn)
    
    # Internals
    if model.uses_Chl:
        model.set_aph(odict['Chl'])

    # Bricaud?
    # Interpolate
    model_anw = anly_utils.convert_to_satwave(l23_wave, odict['anw'], model_wave)
    model_var = anly_utils.scale_noise(scl_noise, model_anw, model_wave)

    if add_noise:
        model_Rrs = anly_utils.add_noise(
                model_Rrs, abs_sig=np.sqrt(model_var))

    # Initial guess
    p0_a = model.init_guess(model_anw)
    p0 = p0_a


    def show_fit(anw, errs:list=None):
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(model_wave, anw, 'r-', label='Guess')
        if errs is not None:
            ax.fill_between(model_wave, errs[0], errs[1], color='r', alpha=0.5)
        ax.plot(model_wave, model_anw, 'k-', label='True')
        ax.legend()
        plt.show()
        
    # Check
    if chk_guess:
        show_fit(p0)

    # Set the items
    items = [(model_anw, model_var, p0, idx)]

    # Bayes
    if not use_chisq:
        # Fit
        chains, idx = fit_one(items[0], model=model, 
                              pdict=pdict, chains_only=True)



        # Save
        anly_utils.save_fits(chains, idx, outfile, 
                             extras=dict(wave=model_wave, 
                                         obs_Rrs=model_anw, 
                                         var=model_var, 
                                         Chl=odict['Chl'], 
                                         Y=odict['Y']))
        embed(header='show_fit 290')

        # Show
        if show and False:
            burn=7000 
            thin=1
            chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
            anw = model.eval_anw(chains)
            # Calculate the mean and standard deviation
            a_mean = np.median(anw, axis=0)
            a_5, a_95 = np.percentile(anw, [5, 95], axis=0)
            #
            show_fit(a_mean, errs=[a_5, a_95])
            embed(header='show_fit 280')

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
            anw = model.eval_anw(p0).flatten()
            show_fit(anw)

        # Save
        if outfile is not None:
            np.savez(outfile, ans=ans, cov=cov,
                wave=wave, obs_anw=model_anw,
                Chl=odict['Chl'], Y=odict['Y'])
            print(f"Saved: {outfile}")
        #
        return model, model_anw, p0, ans, cov


def quick_plt():
    d = np.load('fitanw_170_MCMC_Chase2017.npz')
    chains = d['chains']
    burn=7000 
    thin=1
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    model = bing_anw.init_model('Chase2017', d['wave'])
    anw = model.eval_anw(chains)
    # Calculate the mean and standard deviation
    a_mean = np.median(anw, axis=0)
    a_5, a_95 = np.percentile(anw, [5, 95], axis=0)
    #

    embed(header='quick_plt 334')
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(model_wave, anw, 'r-', label='Guess')
    if errs is not None:
        ax.fill_between(model_wave, errs[0], errs[1], color='r', alpha=0.5)
    ax.plot(model_wave, model_anw, 'k-', label='True')
    ax.legend()
    plt.show()

def main(flg):
    flg = int(flg)

    # Testing
    if flg == 1:
        #odict = fit('Chase2017', 170, 'fitanw_170_Chase2017.npz',
        #            show=True, use_chisq=True)
        odict = fit('Chase2017', 170, 'fitanw_170_MCMC_Chase2017.npz', 
                    show=True, use_chisq=False, nsteps=20000)

    if flg == 2:
        quick_plt()

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