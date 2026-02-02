"""
MCMC Inference Module for BING
==============================

This module implements Markov Chain Monte Carlo (MCMC) inference for
bio-optical parameter retrieval using the emcee ensemble sampler.

The module provides functions for:
- Computing log-probability for Bayesian inference
- Initializing and running MCMC sampling
- Single-spectrum and batch fitting workflows
- Parallel processing support for large datasets

The inference framework follows a standard Bayesian approach:
    posterior ∝ likelihood × prior

where the likelihood is Gaussian based on Rrs measurement uncertainties,
and priors are specified via the model objects.

References
----------
- Foreman-Mackey, D. et al. (2013). "emcee: The MCMC Hammer,"
  PASP 125, 306-312.

Examples
--------
>>> from bing.fitting import inference
>>> from bing.models import utils as model_utils
>>>
>>> # Initialize models and MCMC configuration
>>> models = model_utils.init(['ExpBricaud', 'Pow'], wave)
>>> pdict = inference.init_mcmc(models, nsteps=40000, nburn=1000)
>>>
>>> # Fit a single spectrum
>>> items = (Rrs, varRrs, p0, idx)
>>> chains, idx = inference.fit_one(items, models=models, pdict=pdict,
...                                  chains_only=True, rt_dict=rt_dict)
"""
import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from bing.models import utils as model_utils
from bing import evaluate as bing_eval

import emcee

from IPython import embed

def log_prob(params, models:list, Rrs:np.ndarray,
             varRrs:np.ndarray, rt_dict:dict):
    """
    Compute the log-posterior probability for given parameters.

    This is the objective function for MCMC sampling. It combines the
    log-prior (from model priors) with the log-likelihood (Gaussian,
    based on Rrs residuals and measurement variance).

    Parameters
    ----------
    params : np.ndarray
        Combined parameter vector [a_params, bb_params] in model-specific
        space (typically log10 for amplitudes, linear for slopes).
    models : list
        List of two model objects: [absorption_model, backscattering_model].
        Each must have `priors` attribute and `nparam` count.
    Rrs : np.ndarray
        Observed remote sensing reflectance [sr^-1].
    varRrs : np.ndarray
        Variance of Rrs measurements [sr^-2].
    rt_dict : dict
        Radiative transfer configuration dictionary with keys:
        - 'variable_Gordon' : bool - Use wavelength-dependent Gordon coefficients
        - 'include_Raman' : bool - Include Raman scattering correction

    Returns
    -------
    float
        Log-posterior probability. Returns -np.inf if parameters are
        outside prior bounds or if calculation produces NaN.

    Notes
    -----
    The log-likelihood is computed as:
        log(L) = -0.5 × Σ[(Rrs_model - Rrs_obs)² / varRrs]

    The total log-posterior is:
        log(P) = log(L) + log(prior_a) + log(prior_bb)
    """
    # Unpack for convenience
    aparams = params[:models[0].nparam]
    bparams = params[models[0].nparam:]

    # Priors
    # TODO -- allow for more complex priors
    a_prior = models[0].priors.calc(aparams)
    b_prior = models[1].priors.calc(bparams)

    if np.any(np.isneginf([a_prior, b_prior])):
        return -np.inf

    # Proceed
    pred = bing_eval.calc_Rrs_from_models(models[0], aparams, models[1],
        bparams, rt_dict)

    # Evaluate
    eeval = (pred-Rrs)**2 / varRrs
    # Finish
    prob = -0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob + a_prior + b_prior

def init_mcmc(models:list, nsteps:int=10000, nburn:int=1000):
    """
    Initialize MCMC configuration dictionary.

    Creates a configuration dictionary with parameters needed for emcee
    ensemble sampling. The number of walkers is automatically set based
    on the total number of model parameters.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
        Used to determine total number of parameters (ndim).
    nsteps : int, optional
        Number of MCMC steps to run after burn-in. Default is 10000.
    nburn : int, optional
        Number of burn-in steps (discarded). Default is 1000.

    Returns
    -------
    dict
        MCMC configuration dictionary with keys:
        - 'nwalkers' : int - Number of ensemble walkers (max(16, 2×ndim))
        - 'nsteps' : int - Steps after burn-in
        - 'nburn' : int - Burn-in steps
        - 'save_file' : str or None - Path for HDF5 backend (None = no save)
        - 'Chl' : np.ndarray or None - Chlorophyll values for batch processing
        - 'Y' : np.ndarray or None - Backscattering slope values for batch

    Notes
    -----
    The number of walkers must be at least 2×ndim for emcee. We use
    max(16, 2×ndim) to ensure adequate sampling even for low-dimensional
    problems.

    Examples
    --------
    >>> pdict = init_mcmc(models, nsteps=40000, nburn=2000)
    >>> print(pdict['nwalkers'])
    16
    """
    pdict = {}
    ndim = np.sum([model.nparam for model in models])
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = nsteps
    pdict['nburn'] = nburn
    pdict['save_file'] = None
    #
    return pdict


def fit_one(items:list, models:list=None, pdict:dict=None,
            chains_only:bool=False, rt_dict:dict=None):
    """
    Fit a single spectrum using MCMC.

    Runs emcee ensemble sampling for a single Rrs spectrum, automatically
    handling model initialization (Chl, Y parameters) and walker setup.

    Parameters
    ----------
    items : tuple
        Tuple containing (Rrs, varRrs, params, idx):
        - Rrs : np.ndarray - Observed remote sensing reflectance [sr^-1]
        - varRrs : np.ndarray - Variance of Rrs [sr^-2]
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index (for batch tracking and Chl/Y lookup)
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    pdict : dict
        MCMC configuration from init_mcmc(), plus:
        - 'Chl' : np.ndarray - Chlorophyll values indexed by spectrum idx
        - 'Y' : np.ndarray - Backscattering slope values indexed by idx
    chains_only : bool, optional
        If True, returns only the chain array (float32) instead of the
        full sampler object. Useful for memory efficiency. Default is False.
    rt_dict : dict
        Radiative transfer configuration dictionary.

    Returns
    -------
    sampler_or_chains : emcee.EnsembleSampler or np.ndarray
        If chains_only=False: Full emcee sampler object
        If chains_only=True: Chain array with shape (nsteps, nwalkers, nparam)
    idx : int
        Input index (echoed for tracking in batch processing)

    Notes
    -----
    The function updates model internals (Chl for Bricaud, Y for Lee)
    before running MCMC. These values are looked up from pdict using idx.

    See Also
    --------
    fit_batch : Fit multiple spectra in parallel
    run_emcee : Lower-level emcee interface
    """
    # Unpack
    Rrs, varRrs, params, idx = items

    Chl = pdict['Chl'][idx] if pdict['Chl'] is not None else None
    Y = pdict['Y'][idx] if pdict['Y'] is not None else None

    # Update the model as need be
    _ = model_utils.init_other_bits(
        models, Chl=Chl, Y=Y, Rrs=Rrs)

    # Run
    print(f"idx={idx}")
    sampler = run_emcee(
        models, Rrs, varRrs, rt_dict,
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

def run_emcee(models:list, Rrs, varRrs, rt_dict,
              nwalkers:int=32,
              nburn:int=1000,
              nsteps:int=20000, save_file:str=None,
              p0=None, skip_check:bool=False, ndim:int=None):
    """
    Run the emcee ensemble sampler for Bayesian inference.

    Low-level interface to emcee that handles walker initialization,
    burn-in, and production sampling. Supports optional HDF5 backend
    for saving chains.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    Rrs : np.ndarray
        Observed remote sensing reflectance [sr^-1].
    varRrs : np.ndarray
        Variance of Rrs measurements [sr^-2].
    rt_dict : dict
        Radiative transfer configuration dictionary.
    nwalkers : int, optional
        Number of ensemble walkers. Must be ≥ 2×ndim. Default is 32.
    nburn : int, optional
        Number of burn-in steps (discarded after equilibration). Default is 1000.
    nsteps : int, optional
        Number of production steps after burn-in. Default is 20000.
    save_file : str, optional
        Path to HDF5 file for saving chains via emcee backend.
        If None, chains are kept in memory only. Default is None.
    p0 : np.ndarray, optional
        Initial parameter guess (1D array). Walkers are initialized by
        replicating p0 and adding small perturbations (±1%). Required.
    skip_check : bool, optional
        Skip emcee's initial state validation. Useful when starting from
        known good positions. Default is False.
    ndim : int, optional
        Number of parameters (inferred from p0 if not provided).

    Returns
    -------
    emcee.EnsembleSampler
        The emcee sampler object containing chains and metadata.
        Access chains via sampler.get_chain().

    Raises
    ------
    ValueError
        If p0 is not provided.

    Notes
    -----
    The sampling proceeds in two phases:
    1. Burn-in: nburn steps, then sampler is reset
    2. Production: nsteps steps, chains are retained

    Walker initialization:
    - p0 is replicated nwalkers times
    - Each walker is perturbed by ±1% random noise
    - This "ball" initialization helps ensure walker diversity

    Examples
    --------
    >>> sampler = run_emcee(models, Rrs, varRrs, rt_dict,
    ...                     nwalkers=32, nburn=2000, nsteps=40000,
    ...                     p0=initial_params)
    >>> chains = sampler.get_chain()  # Shape: (nsteps, nwalkers, ndim)
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
        args=[models, Rrs, varRrs, rt_dict],
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

def fit_batch(models:list, pdict:dict, items:list, rt_dict:dict,
              n_cores:int=1, fit_method=None):
    """
    Fit multiple spectra in parallel using ProcessPoolExecutor.

    Distributes MCMC fitting across multiple CPU cores for efficient
    batch processing of large datasets.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    pdict : dict
        MCMC configuration from init_mcmc(), including Chl and Y arrays
        for all spectra to be fitted.
    items : list of tuple
        List of (Rrs, varRrs, params, idx) tuples, one per spectrum.
        Each tuple contains:
        - Rrs : np.ndarray - Observed reflectance
        - varRrs : np.ndarray - Variance
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index
    rt_dict : dict
        Radiative transfer configuration dictionary.
    n_cores : int, optional
        Number of CPU cores for parallel processing. Default is 1.
    fit_method : callable, optional
        Fitting function to use. Default is fit_one.

    Returns
    -------
    all_samples : np.ndarray
        Array of MCMC chains with shape (n_spectra, nsteps, nwalkers, nparam).
        Stored as float32 for memory efficiency.
    all_idx : np.ndarray
        Array of spectrum indices corresponding to each chain.

    Notes
    -----
    - Chains are returned as float32 to reduce memory usage
    - Progress is displayed via tqdm
    - Chunk size is automatically set to len(items) // n_cores

    Examples
    --------
    >>> items = [(Rrs[i], varRrs[i], p0[i], i) for i in range(n_spectra)]
    >>> chains, indices = fit_batch(models, pdict, items, n_cores=16)
    >>> for chain, idx in zip(chains, indices):
    ...     process_result(chain, idx)
    """
    if fit_method is None:
        fit_method = fit_one

    # Setup for parallel
    map_fn = partial(fit_one, models=models, pdict=pdict, chains_only=True, rt_dict=rt_dict)
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Need to avoid blowing up the memory!
    # Slurp
    all_idx = np.array([item[1] for item in answers])
    answers = np.array([item[0].astype(np.float32) for item in answers])

    return answers, all_idx