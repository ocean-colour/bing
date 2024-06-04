""" Inference methods for BIG """
import numpy as np

from big import rt as big_rt

import emcee

from IPython import embed

def log_prob(params, models:list, Rrs:np.ndarray, varRrs):
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
    aparams = params[:models[0].nparam]
    bparams = params[models[0].nparam:]

    # Priors
    # TODO -- allow for more complex priors
    a_prior = models[0].priors.calc(aparams)
    b_prior = models[1].priors.calc(bparams)

    if a_prior or b_prior:
        return -np.inf

    # Proceed
    a = models[0].eval_a(aparams)
    bb = models[1].eval_bb(bparams)

    # TODO -- allow for non-standard Gordon coefficients
    pred = big_rt.calc_Rrs(a, bb) 

    # Evaluate
    eeval = (pred-Rrs)**2 / varRrs
    # Finish
    prob = -0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob

def init_mcmc(models:list, nsteps:int=10000, nburn:int=1000):
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
    ndim = np.sum([model.nparam for model in models])
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = nsteps
    pdict['nburn'] = nburn
    pdict['save_file'] = None
    #
    return pdict


def fit_one(models, items:list, pdict:dict=None, chains_only:bool=False):
    """
    Fits a model to a set of input data using the MCMC algorithm.

    Args:
        models (list): The list of model objects, a_nw, bb_nw
        items (list): A list containing the input data, errors, response data, and index.
        pdict (dict, optional): A dictionary containing the model and fitting parameters. Defaults to None.
        chains_only (bool, optional): If True, only the chains are returned. Defaults to False.

    Returns:
        tuple: A tuple containing the MCMC sampler object and the index.
    """
    # Unpack
    Rs, varRs, params, idx = items

    # Run
    print(f"idx={idx}")
    sampler = run_emcee(
        models, Rs, varRs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        nburn=pdict['nburn'],
        skip_check=True,
        p0=params,
        save_file=pdict['save_file'],
        pdict=pdict)

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def run_emcee(models:list, Rrs, varRrs, nwalkers:int=32, 
              nburn:int=1000,
              nsteps:int=20000, save_file:str=None, 
              p0=None, skip_check:bool=False, ndim:int=None,
              pdict:dict=None):
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
        #priors = grab_priors(model)
        #ndim = priors.shape[0]
        #p0 = np.random.uniform(priors[:,0], priors[:,1], size=(nwalkers, ndim))
    else:
        # Replicate for nwalkers
        ndim = len(p0)
        p0 = np.tile(p0, (nwalkers, 1))
        # Perturb 
        p0 += p0*np.random.uniform(-1e-2, 1e-2, size=p0.shape)
        #r = 10**np.random.uniform(-0.5, 0.5, size=p0.shape[0])
        #for ii in range(p0.shape[0]):
        #    p0[ii] *= r[ii]
        #embed(header='108 of fgordon')

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
        args=[models, Rrs, varRrs],
        backend=backend)#, pool=pool)

    # Burn in
    print("Running burn-in")
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