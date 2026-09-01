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
from bing.priors import priors as bing_priors
from bing.rt import defs as rt_defs

import emcee

from IPython import embed

#: The free-B_p prior (M3 task 2; plan choice, design §7.3): a
#: **linear-space** uniform over
#: [rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX] = [0.004, 0.05] --
#: B_p is a ratio (bb_p/b_p), like the slopes, not a log10 amplitude.
#: Built from the same UniformPrior class as the model priors so the
#: conventions match by construction: bounds inclusive (strict </>
#: comparisons), in-range log-density contribution exactly 0 (posterior
#: shape only -- this codebase's uniform priors carry no -log(width)
#: normalization).  log_prob evaluates it on the peeled B_p tail whenever
#: rt_dict['fit_Bp'] is True.
BP_PRIOR = bing_priors.UniformPrior(dict(
    flavor='uniform', pmin=rt_defs.BP_PRIOR_PMIN,
    pmax=rt_defs.BP_PRIOR_PMAX))

def log_prob(params, models:list, Rrs:np.ndarray,
             varRrs:np.ndarray, rt_dict:dict, geom=None):
    """
    Compute the log-posterior probability for given parameters.

    This is the objective function for MCMC sampling. It combines the
    log-prior (from model priors) with the log-likelihood (Gaussian,
    based on Rrs residuals and measurement variance).

    Parameters
    ----------
    params : np.ndarray
        Combined parameter vector [a_params, bb_params] in model-specific
        space (typically log10 for amplitudes, linear for slopes). When
        rt_dict['fit_Bp'] is True the vector carries one extra trailing
        element -- [a_params..., bb_params..., B_p] (design §3.3), where
        B_p is the particulate backscattering ratio bb_p/b_p in *linear*
        space. The tail is peeled off before the aparams/bparams split,
        so the model parameter layout is unchanged either way.
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
        - 'rt_backend' : str, optional - Radiative transfer backend
          ('gordon', the default when absent, or one of the robust.rt
          backends; see bing.rt.defs.RT_BACKENDS). Non-gordon values
          dispatch the forward model to
          bing.evaluate.calc_Rrs_from_models_robust.
        - 'fit_Bp' : bool, optional - When True (robust backends only;
          validate_rt_dict rejects it for 'gordon' at fit setup), B_p is
          a free/sampled parameter riding as the last element of
          ``params``; it is peeled off here, checked against its own
          prior (BP_PRIOR: a linear-space uniform over the inclusive
          [rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX] = [0.004, 0.05]
          range -- out of range returns -np.inf before any forward-model
          call; in range contributes 0, like the model uniform priors),
          and forwarded as the adapter's ``Bp`` argument. When
          False/absent (the default), ``params`` is the plain model
          vector, no B_p prior is evaluated, and the adapter falls back
          to rt_dict['Bp_value'] (Bp=None -- the fixed-B_p case).
    geom : bing.rt.geometry.ObsGeometry, optional
        Fixed per-pixel viewing/illumination geometry. Required (non-None)
        whenever rt_dict['rt_backend'] selects a robust backend; ignored
        by the default Gordon backend.

    Returns
    -------
    float
        Log-posterior probability. Returns -np.inf if parameters are
        outside prior bounds (including, under rt_dict['fit_Bp'], a B_p
        tail outside BP_PRIOR's [0.004, 0.05] range -- checked before
        the forward-model call) or if calculation produces NaN.

    Notes
    -----
    The log-likelihood is computed as:
        log(L) = -0.5 × Σ[(Rrs_model - Rrs_obs)² / varRrs]

    The total log-posterior is:
        log(P) = log(L) + log(prior_a) + log(prior_bb) [+ log(prior_B_p)]

    where the B_p term appears only under rt_dict['fit_Bp'] and, being a
    uniform prior, is either 0 (in range) or -inf (out of range).

    log_prob itself performs no configuration validation -- it sits on
    the sampling hot path. fit_one / chisq_fit.fit run
    bing.rt.defs.validate_rt_dict (and, for robust backends,
    bing.evaluate.robust_domain_check) once at fit setup instead, so an
    illegal rt_dict/geom combination raises there, before sampling. A
    *direct* robust-backend call with geom=None still fails loudly --
    with the adapter's own ValueError from
    calc_Rrs_from_models_robust -- rather than silently defaulting
    theta_s.
    """
    # B_p tail peel (M3 task 1, design §3.3): when rt_dict['fit_Bp'] is
    # True the sampled vector is [a_params..., bb_params..., B_p] -- peel
    # the tail *first* so the aparams/bparams split below is untouched.
    # Otherwise Bp stays None and the adapter falls back to
    # rt_dict['Bp_value'] (the fixed-B_p case, M2 Q1).
    if rt_dict.get('fit_Bp', False):
        Bp = params[-1]
        params = params[:-1]
    else:
        Bp = None

    # Unpack for convenience
    aparams = params[:models[0].nparam]
    bparams = params[models[0].nparam:]

    # Priors
    # TODO -- allow for more complex priors
    a_prior = models[0].priors.calc(aparams)
    b_prior = models[1].priors.calc(bparams)
    # B_p prior (M3 task 2; plan choice, design §7.3): a linear-space
    # uniform over the inclusive [BP_PRIOR_PMIN, BP_PRIOR_PMAX] range,
    # evaluated alongside the model priors -- an out-of-range tail
    # short-circuits to -inf below, *before* the forward call, exactly
    # like an out-of-range model parameter.  In range it contributes
    # exactly 0 (UniformPrior's convention).  0. when B_p is fixed
    # (fit_Bp False/absent), leaving that path's value untouched.
    Bp_prior = 0. if Bp is None else BP_PRIOR.calc(Bp)

    if np.any(np.isneginf([a_prior, b_prior, Bp_prior])):
        return -np.inf

    # Proceed -- dispatch on the RT backend (design §3.4).  The default
    # ('gordon', also the value when 'rt_backend' is absent) keeps the
    # legacy call byte-for-byte unchanged.
    if rt_dict.get('rt_backend', 'gordon') == 'gordon':
        pred = bing_eval.calc_Rrs_from_models(models[0], aparams, models[1],
            bparams, rt_dict)
    else:
        # Bp is the peeled tail when fit_Bp is True; None otherwise, which
        # the adapter documents as "fall back to rt_dict['Bp_value']" (the
        # fixed-B_p case).
        pred = bing_eval.calc_Rrs_from_models_robust(models[0], aparams,
            models[1], bparams, rt_dict, geom=geom, Bp=Bp)

    # Evaluate
    eeval = (pred-Rrs)**2 / varRrs
    # Finish
    prob = -0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob + a_prior + b_prior + Bp_prior

def init_mcmc(models:list, nsteps:int=10000, nburn:int=1000,
              rt_dict:dict=None):
    """
    Initialize MCMC configuration dictionary.

    Creates a configuration dictionary with parameters needed for emcee
    ensemble sampling. The number of walkers is automatically set based
    on the total number of parameters -- the model parameters, plus one
    for the free B_p tail when ``rt_dict['fit_Bp']`` is True (M3 task 3,
    design §3.3).

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
        Used to determine total number of parameters (ndim).
    nsteps : int, optional
        Number of MCMC steps to run after burn-in. Default is 10000.
    nburn : int, optional
        Number of burn-in steps (discarded). Default is 1000.
    rt_dict : dict, optional
        Radiative transfer configuration. Only 'fit_Bp' (default False)
        is consulted: when True, ndim -- and therefore the walker count
        -- accounts for the extra trailing B_p dimension of the sampled
        vector ``[a_params..., bb_params..., B_p]``. None (the default)
        is the fixed-B_p case and leaves both exactly as before.

    Returns
    -------
    dict
        MCMC configuration dictionary with keys:

        - 'ndim' : int - Total sampled dimensions (sum of model nparam,
          +1 when rt_dict['fit_Bp'] is True)
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
    problems. For the standard 5-parameter models, fit_Bp takes ndim
    5 -> 6 and nwalkers stays at 16.

    Examples
    --------
    >>> pdict = init_mcmc(models, nsteps=40000, nburn=2000)
    >>> print(pdict['nwalkers'])
    16
    """
    pdict = {}
    ndim = int(np.sum([model.nparam for model in models]))
    # Free B_p (M3, design §3.3): one extra trailing dimension.
    if rt_dict is not None and rt_dict.get('fit_Bp', False):
        ndim += 1
    pdict['ndim'] = ndim
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
        Tuple containing (Rrs, varRrs, params, idx[, geom]):

        - Rrs : np.ndarray - Observed remote sensing reflectance [sr^-1]
        - varRrs : np.ndarray - Variance of Rrs [sr^-2]
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index (for batch tracking and Chl/Y lookup)
        - geom : bing.rt.geometry.ObsGeometry, optional 5th element -
          fixed per-pixel viewing/illumination geometry (design §3.2).
          Required whenever rt_dict['rt_backend'] selects a robust
          backend; ignored by the default Gordon backend. Legacy
          4-tuples remain valid and imply geom=None.
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

    Raises
    ------
    ValueError
        At setup, from bing.rt.defs.validate_rt_dict -- before any
        sampling work: an unknown rt_backend; fit_Bp=True with the
        Gordon backend; a robust backend without geom (the error names
        theta_s -- it is never silently defaulted); or a robust_hybrid
        fit whose model wavelengths fall outside the emulator's
        [350, 750] nm training range.

    Warns
    -----
    robust.rt.hybrid.DomainWarning
        If the initial guess (checked before sampling) or the posterior
        median (checked after) lies outside the emulator's trained
        domain. Possible for rt_backend='robust_hybrid' only -- the
        check is a validated no-op for the other robust backends and is
        never invoked for 'gordon'. Warn-and-continue: the fit result is
        still returned.

    Notes
    -----
    The function updates model internals (Chl for Bricaud, Y for Lee)
    before running MCMC. These values are looked up from pdict using idx.

    Setup validation runs once per fit, immediately after the tuple
    unpack (bing.rt.defs.validate_rt_dict) -- the sampler itself runs
    jitted for robust backends and can never warn or validate, so both
    the configuration errors and the out-of-domain diagnostics
    (bing.evaluate.robust_domain_check on the initial guess and on the
    posterior median) live here, outside the hot loop.

    See Also
    --------
    fit_batch : Fit multiple spectra in parallel
    run_emcee : Lower-level emcee interface
    """
    # Unpack -- either the legacy 4-tuple (Rrs, varRrs, params, idx) or
    # the 5-tuple with a trailing ObsGeometry (design §3.2)
    Rrs, varRrs, params, idx = items[:4]
    geom = items[4] if len(items) > 4 else None

    # Setup validation, once per fit and before any other work (M2 task
    # 3): an unknown backend, fit_Bp with the Gordon backend, a robust
    # backend without geometry (the CQ4 error naming theta_s), or an
    # out-of-range robust_hybrid wavelength grid all raise here -- never
    # mid-sampling.  rt_dict=None (a legacy convenience some callers use)
    # validates as the default Gordon configuration.
    rt_defs.validate_rt_dict(rt_dict if rt_dict is not None else {},
                             models=models, geom=geom)
    rt_backend = (rt_dict.get('rt_backend', 'gordon')
                  if rt_dict is not None else 'gordon')

    Chl = pdict['Chl'][idx] if pdict['Chl'] is not None else None
    Y = pdict['Y'][idx] if pdict['Y'] is not None else None

    # Update the model as need be
    _ = model_utils.init_other_bits(
        models, Chl=Chl, Y=Y, Rrs=Rrs)

    # Domain check on the initial guess (robust backends only): un-jitted,
    # so robust_hybrid's DomainWarning can reach the caller before any
    # sampling begins.  Deliberately never called for 'gordon' --
    # robust_domain_check rejects a non-robust backend with ValueError by
    # construction (verified empirically; see
    # test_robust_domain_check_shares_adapter_error_paths) -- and it is a
    # validated no-op for robust_ztt/robust_baseline (no trained domain).
    if rt_backend != 'gordon':
        nap = models[0].nparam
        p0_check = np.asarray(params)
        # When B_p is free (fit_Bp, design §3.3) the caller's p0 carries
        # the B_p tail -- peel it exactly as log_prob does, so the check
        # sees the same aparams/bparams/Bp the sampler will use.  Bp=None
        # otherwise (the adapter falls back to rt_dict['Bp_value']).
        Bp_check = None
        if rt_dict.get('fit_Bp', False):
            Bp_check = float(p0_check[-1])
            p0_check = p0_check[:-1]
        bing_eval.robust_domain_check(
            models[0], p0_check[:nap], models[1], p0_check[nap:],
            rt_dict, geom=geom, Bp=Bp_check)

    # Run
    print(f"idx={idx}")
    sampler = run_emcee(
        models, Rrs, varRrs, rt_dict,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        nburn=pdict['nburn'],
        skip_check=True,
        p0=params,
        save_file=pdict['save_file'],
        geom=geom)

    # Domain check on the posterior median (robust backends only): the
    # sampling itself runs jitted and can never warn (design §4), so this
    # is where an out-of-domain landing point becomes visible.  Same
    # aparams/bparams split as log_prob's.
    if rt_backend != 'gordon':
        chain = sampler.get_chain()
        median = np.median(chain.reshape(-1, chain.shape[-1]), axis=0)
        # Same B_p tail peel as the p0 check above: with fit_Bp the chain
        # has the extra trailing column, so the flattened median does too.
        Bp_med = None
        if rt_dict.get('fit_Bp', False):
            Bp_med = float(median[-1])
            median = median[:-1]
        bing_eval.robust_domain_check(
            models[0], median[:nap], models[1], median[nap:],
            rt_dict, geom=geom, Bp=Bp_med)

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def append_Bp_seed(p0, rt_dict:dict):
    """
    Append the B_p seed to an initial-guess vector when B_p is free.

    When ``rt_dict['fit_Bp']`` is True the fitting vector carries B_p as
    its last element -- ``[a_params..., bb_params..., B_p]`` (design
    §3.3) -- so the initial guess needs a matching tail. This helper
    seeds it at ``rt_dict['Bp_value']`` (default 0.01, matching
    bing.rt.defs.rt_dict_from_p): a *linear-space* value (never log10'd
    -- B_p is a ratio, and its prior flavor is 'uniform'), inside
    BP_PRIOR's default [0.004, 0.05] range, and nonzero -- so
    init_walkers' floored perturbation gives the dimension real spread
    across walkers.

    Called by the canonical p0 builders (e.g. l23.prep_one_l23, after
    their model-parameter initial guess and log10 conversion); direct
    callers assembling p0 by hand under fit_Bp should use it too, or
    append the tail themselves.

    Parameters
    ----------
    p0 : np.ndarray or sequence
        Initial guess for the model parameters (aparams + bparams), in
        fitting space (log10 already applied where the priors say so).
    rt_dict : dict
        Radiative transfer configuration. Only 'fit_Bp' (default False)
        and 'Bp_value' (default 0.01) are consulted. None is treated as
        the fixed-B_p default.

    Returns
    -------
    np.ndarray
        ``p0`` with ``Bp_value`` appended when rt_dict['fit_Bp'] is
        True; ``np.asarray(p0)`` unchanged otherwise.
    """
    if rt_dict is not None and rt_dict.get('fit_Bp', False):
        return np.append(p0, rt_dict.get('Bp_value', 0.01))
    return np.asarray(p0)


def prior_bounds(models:list, rt_dict:dict=None):
    """
    Lower/upper parameter bounds taken from the models' priors.

    Parameters without usable bounds -- a model with no priors attached,
    or a prior flavor that has no pmin/pmax (e.g. gaussian) -- come back
    as -inf/+inf so they are simply left alone by any clipping.

    When ``rt_dict['fit_Bp']`` is True (M3 task 3, design §3.3) the
    sampled vector carries a trailing B_p element, so the bounds gain a
    matching trailing slot: BP_PRIOR's own
    [rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX] = [0.004, 0.05] --
    the same single source of truth log_prob's B_p prior and
    l23.fit_with_LM's curve_fit bounds use, so sampling, clipping, and
    optimization can never disagree on the range.

    Parameters
    ----------
    models : list
        Model objects in parameter order, typically [a_model, bb_model].
    rt_dict : dict, optional
        Radiative transfer configuration. Only 'fit_Bp' (default False)
        is consulted. None (the default) is the fixed-B_p case: bounds
        for the model parameters only, exactly as before.

    Returns
    -------
    tuple of np.ndarray
        (lower, upper), each of length sum(model.nparam), plus one when
        rt_dict['fit_Bp'] is True.
    """
    lows, highs = [], []
    for model in models:
        priors = getattr(model, 'priors', None)
        for kk in range(model.nparam):
            pmin = pmax = None
            if priors is not None and kk < len(priors.priors):
                pmin = getattr(priors.priors[kk], 'pmin', None)
                pmax = getattr(priors.priors[kk], 'pmax', None)
            lows.append(-np.inf if pmin is None else float(pmin))
            highs.append(np.inf if pmax is None else float(pmax))
    # Free-B_p tail slot (M3 task 3) -- from the defs constants, never a
    # re-typed literal.
    if rt_dict is not None and rt_dict.get('fit_Bp', False):
        lows.append(float(rt_defs.BP_PRIOR_PMIN))
        highs.append(float(rt_defs.BP_PRIOR_PMAX))
    return np.array(lows), np.array(highs)


def init_walkers(p0:np.ndarray, nwalkers:int, models:list=None,
                 frac:float=1e-2, floor:float=1e-3, rng=None,
                 rt_dict:dict=None):
    """
    Build the initial ball of walker positions for emcee.

    Each walker is ``p0`` plus a uniform perturbation whose half-width
    is ``max(abs(p0_k) * frac, floor)``, per parameter, after which
    walkers are clipped into the prior bounds.

    The floor is the important part. The perturbation used to be purely
    *multiplicative* (``p0 += p0*U(-frac, frac)``), so any parameter
    seeded at exactly 0 received **zero** spread across walkers. Because
    emcee's stretch move proposes along walker-to-walker vectors, a
    dimension with no inter-walker spread never moves for the entire
    run -- silently, with a healthy acceptance fraction and a zero-width
    credible interval. Linear parameters legitimately sit at 0 (a flat
    backscattering exponent, for instance), so this was not a corner
    case. Small-but-nonzero values were nearly as bad: a slope of 0.015
    got a half-width of 1.5e-4.

    Clipping to the priors also guarantees every walker starts with a
    finite log-probability, which the multiplicative version did not.

    Parameters
    ----------
    p0 : np.ndarray
        Starting parameter vector (1D, in fitting space).
    nwalkers : int
        Number of walkers.
    models : list, optional
        Models whose priors supply the clip bounds. If None, no clipping
        is applied.
    frac : float, optional
        Relative half-width of the perturbation. Default 1e-2, matching
        the historical 1% ball for well-scaled parameters.
    floor : float, optional
        Absolute floor on the half-width, used wherever ``abs(p0)*frac``
        falls below it. Default 1e-3.
    rng : optional
        Anything providing ``uniform(low, high, size)``. Defaults to the
        legacy ``np.random`` module, so ``np.random.seed`` still governs
        reproducibility (see bing.fitting.l23.batch_fit).
    rt_dict : dict, optional
        Radiative transfer configuration, forwarded to prior_bounds.
        When ``rt_dict['fit_Bp']`` is True (and models is not None) the
        clip bounds gain the trailing B_p slot
        ([rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX]), so a tailed p0's
        B_p column is clipped into its prior like every model parameter
        -- the perturbation floor (1e-3) is a substantial fraction of
        the [0.004, 0.05] range, so this clipping is not theoretical.
        None (the default) leaves any tail column unclipped, the pre-M3
        behavior.

    Returns
    -------
    np.ndarray
        Walker positions with shape (nwalkers, p0.size).
    """
    rng = np.random if rng is None else rng
    p0 = np.asarray(p0, dtype=float).flatten()

    # Per-parameter perturbation scale, floored so no dimension is dead
    scale = np.maximum(np.abs(p0)*frac, floor)
    walkers = np.tile(p0, (nwalkers, 1))
    walkers = walkers + rng.uniform(-1., 1., size=walkers.shape)*scale

    # Keep every walker inside the priors
    if models is not None:
        low, high = prior_bounds(models, rt_dict=rt_dict)
        nclip = min(walkers.shape[1], low.size)
        walkers[:, :nclip] = np.clip(walkers[:, :nclip],
                                     low[:nclip], high[:nclip])
    return walkers


def run_emcee(models:list, Rrs, varRrs, rt_dict,
              nwalkers:int=32,
              nburn:int=1000,
              nsteps:int=20000, save_file:str=None,
              p0=None, skip_check:bool=False, ndim:int=None,
              perturb_frac:float=1e-2, perturb_floor:float=1e-3,
              geom=None):
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
        replicating p0 and perturbing it (see init_walkers). Required.
    skip_check : bool, optional
        Skip emcee's initial state validation. Useful when starting from
        known good positions. Default is False.
    ndim : int, optional
        Number of parameters (inferred from p0 if not provided).
    perturb_frac : float, optional
        Relative half-width of the initial walker ball. Default 1e-2.
    perturb_floor : float, optional
        Absolute floor on that half-width, so parameters seeded at or
        near zero still get real spread. Default 1e-3. Widen both for
        badly degenerate models.
    geom : bing.rt.geometry.ObsGeometry, optional
        Fixed per-pixel viewing/illumination geometry, forwarded by value
        to log_prob through emcee's ``args`` (design §3.2) -- the sampler
        never sees it as a dimension. Required (non-None) whenever
        rt_dict['rt_backend'] selects a robust backend; ignored by the
        default Gordon backend.

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

    Walker initialization (see init_walkers):

    - p0 is replicated nwalkers times
    - Each walker is perturbed by ±1% of ``abs(p0)``, with an absolute
      floor so parameters at or near zero still get real spread
    - Walkers are clipped into the prior bounds, so all start with a
      finite log-probability

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
        # Replicate for nwalkers and perturb into a ball.  The scale is
        # floored and the result clipped into the priors -- see
        # init_walkers for why a purely multiplicative perturbation
        # silently freezes any parameter seeded at 0.  rt_dict rides
        # along so a fit_Bp p0's trailing B_p column is clipped into its
        # own prior slot too (M3 task 3).
        ndim = len(p0)
        p0 = init_walkers(p0, nwalkers, models=models,
                          frac=perturb_frac, floor=perturb_floor,
                          rt_dict=rt_dict)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    # Init.  emcee passes ``args`` purely positionally after the walker's
    # parameter vector, so this list must mirror log_prob's signature
    # order: (params, models, Rrs, varRrs, rt_dict, geom).  geom rides by
    # value like Rrs/varRrs -- never a sampled dimension.
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=[models, Rrs, varRrs, rt_dict, geom],
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
        List of (Rrs, varRrs, params, idx[, geom]) tuples, one per
        spectrum. Each tuple contains:

        - Rrs : np.ndarray - Observed reflectance
        - varRrs : np.ndarray - Variance
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index
        - geom : bing.rt.geometry.ObsGeometry, optional 5th element -
          fixed per-pixel viewing/illumination geometry (design §3.2),
          forwarded unchanged to fit_one. Required whenever
          rt_dict['rt_backend'] selects a robust backend; legacy
          4-tuples remain valid (geom=None) and 4-/5-tuples may be
          mixed in one list.
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
    - Each spectrum's fit runs fit_one's own setup validation
      (bing.rt.defs.validate_rt_dict) in its worker: with a robust
      rt_dict, any legacy 4-tuple in ``items`` (geom=None) raises a
      ValueError naming theta_s, which propagates out of fit_batch;
      fit_one's robust_hybrid domain checks (DomainWarning on the
      initial guess / posterior median) likewise run per spectrum,
      though warnings emitted in worker *processes* do not cross back
      into the parent (n_cores > 1).

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