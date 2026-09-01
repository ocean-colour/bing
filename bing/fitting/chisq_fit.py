"""
Least-Squares (Chi-Squared) Fitting Module for BING
====================================================

This module provides an alternative to MCMC inference using non-linear
least-squares optimization via scipy.optimize.curve_fit (Levenberg-Marquardt).

Advantages over MCMC:
- Much faster (seconds vs minutes)
- Provides point estimates and covariance matrix

Disadvantages:
- No full posterior distribution
- May not capture multi-modal solutions
- Uncertainty estimates assume Gaussian posterior

Use this module for:
- Quick initial fits to guide MCMC starting positions
- Large-scale processing where full posteriors aren't needed
- Validation and comparison with MCMC results

Examples
--------
>>> from bing.fitting import chisq_fit
>>> from bing.models import utils as model_utils
>>>
>>> models = model_utils.init(['ExpBricaud', 'Pow'], wave)
>>> items = (Rrs, varRrs, p0, idx)
>>> bounds = (lower_bounds, upper_bounds)
>>> ans, cov, idx = chisq_fit.fit(items, models, rt_dict, bounds=bounds)
>>>
>>> # Give the optimizer a larger evaluation budget (turbid spectra)
>>> ans, cov, idx = chisq_fit.fit(items, models, rt_dict, bounds=bounds,
...                               maxfev=40000)
"""
import numpy as np

from functools import partial

from scipy.optimize import curve_fit

from bing import evaluate as bing_eval
from bing.rt import defs as rt_defs

from IPython import embed

def fit(items:tuple, models:list, rt_dict:dict, bounds:tuple=None,
        maxfev:int=None):
    """
    Fit Rrs data using Levenberg-Marquardt least-squares optimization.

    Minimizes the weighted chi-squared statistic:
        χ² = Σ[(Rrs_model - Rrs_obs)² / varRrs]

    using scipy.optimize.curve_fit.

    Parameters
    ----------
    items : tuple
        Tuple containing (Rrs, varRrs, params, idx[, geom]):

        - Rrs : np.ndarray - Observed remote sensing reflectance [sr^-1]
        - varRrs : np.ndarray - Variance of Rrs [sr^-2]
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index (echoed in return for batch tracking)
        - geom : bing.rt.geometry.ObsGeometry, optional 5th element -
          fixed per-pixel viewing/illumination geometry (design §3.2),
          forwarded to fit_func on every optimizer evaluation. Required
          whenever rt_dict['rt_backend'] selects a robust backend;
          ignored by the default Gordon backend. Legacy 4-tuples remain
          valid and imply geom=None.
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    rt_dict : dict
        Radiative transfer configuration dictionary with keys:
        - 'variable_Gordon' : bool - Use wavelength-dependent Gordon coefficients
        - 'include_Raman' : bool - Include Raman scattering correction
    bounds : tuple, optional
        Parameter bounds as (lower_bounds, upper_bounds) where each is
        a 1D array matching the parameter vector. Default is (-inf, inf).
        When rt_dict['fit_Bp'] is True the parameter vector (and p0)
        carries a trailing B_p element, so finite bounds need a matching
        trailing slot -- use the default free-B_p range
        [bing.rt.defs.BP_PRIOR_PMIN, bing.rt.defs.BP_PRIOR_PMAX] =
        [0.004, 0.05] to stay consistent with the MCMC prior (see
        l23.fit_with_LM). This is the chi-squared path's only B_p range
        enforcement: fit_func performs no bound checks of its own for
        B_p, exactly as for the model parameters.
    maxfev : int, optional
        Maximum number of forward-model evaluations the optimizer may
        spend. Default None, i.e. leave scipy's own default in place.
        Raising it helps only spectra where the optimizer runs out of
        budget before converging -- it changes *whether* the fit
        returns, not how well the model can fit (on turbid GLORIA
        spectra a ~40x bump moved the convergence rate from 12.5% to
        37.5% with no change in misfit). When the budget is exhausted
        curve_fit raises RuntimeError.

    Returns
    -------
    ans : np.ndarray
        Best-fit parameters (in model space, typically log10 for amplitudes).
    cov : np.ndarray
        Estimated covariance matrix of the parameters. Diagonal elements
        give variance; sqrt of diagonal gives 1-sigma uncertainties.
    idx : int
        Input index (echoed for batch processing tracking).

    Raises
    ------
    ValueError
        At setup, from bing.rt.defs.validate_rt_dict -- before any
        optimizer work: an unknown rt_backend; fit_Bp=True with the
        Gordon backend; a robust backend without geom (the error names
        theta_s -- it is never silently defaulted); or a robust_hybrid
        fit whose model wavelengths fall outside the emulator's
        [350, 750] nm training range.

    Warns
    -----
    robust.rt.hybrid.DomainWarning
        If the initial guess lies outside the emulator's trained domain
        (rt_backend='robust_hybrid' only; checked un-jitted via
        bing.evaluate.robust_domain_check before optimization). Unlike
        fit_one there is no post-fit counterpart -- a chi-squared fit
        has no posterior median; check the returned ``ans`` yourself if
        needed. Warn-and-continue: the fit still runs.

    Notes
    -----
    The covariance matrix assumes the model is correct and residuals are
    Gaussian. For more robust uncertainty estimates, use MCMC inference.

    ``maxfev`` is the correct spelling for both of curve_fit's back ends:
    with finite ``bounds`` it uses least_squares ('trf') and renames the
    keyword to ``max_nfev`` internally, while the unbounded case passes
    it to leastsq ('lm').

    For robust backends the numerical-Jacobian step is widened to a
    float32-appropriate 1e-3 relative step (``diff_step``/``epsfcn``,
    PR #27): robust.rt runs at float32, and scipy's ~1.5e-8 default
    produces a Jacobian of exact zeros there, stalling the optimizer at
    p0. The Gordon backend (float64) keeps scipy's default.

    See Also
    --------
    bing.fitting.inference.fit_one : MCMC-based fitting
    fit_func : Forward model function used by curve_fit
    """
    if bounds is None:
        bounds = (-np.inf, np.inf)
    # Unpack -- either the legacy 4-tuple (Rrs, varRrs, params, idx) or
    # the 5-tuple with a trailing ObsGeometry (design §3.2)
    Rrs, varRrs, params, idx = items[:4]
    geom = items[4] if len(items) > 4 else None

    # Setup validation, once per fit and before any optimizer work (M2
    # task 3): an unknown backend, fit_Bp with the Gordon backend, a
    # robust backend without geometry (the CQ4 error naming theta_s), or
    # an out-of-range robust_hybrid wavelength grid all raise here --
    # never mid-optimization.
    rt_defs.validate_rt_dict(rt_dict if rt_dict is not None else {},
                             models=models, geom=geom)

    # Domain check on the initial guess (robust backends only; there is
    # no posterior in a chi-squared fit, hence no post-fit counterpart):
    # un-jitted, so robust_hybrid's DomainWarning can reach the caller.
    # Deliberately never called for 'gordon' -- robust_domain_check
    # rejects a non-robust backend with ValueError by construction
    # (verified empirically) -- and it is a validated no-op for
    # robust_ztt/robust_baseline (no trained domain).
    if (rt_dict or {}).get('rt_backend', 'gordon') != 'gordon':
        nap = models[0].nparam
        p0_check = np.asarray(params)
        # When B_p is free (fit_Bp, design §3.3) the caller's p0 carries
        # the B_p tail -- peel it exactly as fit_func does, so the check
        # sees the same aparams/bparams/Bp the optimizer will use.
        # Bp=None otherwise (the adapter falls back to
        # rt_dict['Bp_value']).
        Bp_check = None
        if rt_dict.get('fit_Bp', False):
            Bp_check = float(p0_check[-1])
            p0_check = p0_check[:-1]
        bing_eval.robust_domain_check(
            models[0], p0_check[:nap], models[1], p0_check[nap:],
            rt_dict, geom=geom, Bp=Bp_check)

    # Only pass maxfev when asked, so scipy's default is untouched
    kwargs = {} if maxfev is None else dict(maxfev=maxfev)

    # Finite-difference step for the robust backends (PR #27). robust.rt
    # runs at float32 (jax_enable_x64 is never enabled -- CQ1), but
    # curve_fit's default relative step for the numerical Jacobian is
    # ~sqrt(float64 eps) ~ 1.5e-8 -- far below float32 resolution, so
    # every column of the differenced Jacobian is *exactly zero* and the
    # optimizer declares convergence at p0 without moving (measured: a
    # parameter step of 1.5e-8 changes no Rrs value at all; 1e-3 changes
    # them smoothly, and with it a noiseless synthetic refit recovers the
    # generating parameters to ~4e-7). Use a float32-appropriate relative
    # step: `diff_step` for the bounded case (curve_fit dispatches to
    # least_squares/'trf') and the equivalent `epsfcn` (step =
    # sqrt(epsfcn) * |x|) for the unbounded case ('lm' via leastsq).
    # The Gordon backend is float64 end-to-end and keeps scipy's default.
    if (rt_dict or {}).get('rt_backend', 'gordon') != 'gordon':
        unbounded = (np.all(np.isneginf(np.asarray(bounds[0]))) and
                     np.all(np.isposinf(np.asarray(bounds[1]))))
        if unbounded:
            kwargs.setdefault('epsfcn', 1e-6)
        else:
            kwargs.setdefault('diff_step', 1e-3)

    partial_func = partial(fit_func, models=models, rt_dict=rt_dict,
                           geom=geom)
    ans, cov =  curve_fit(partial_func, None,
                          Rrs, p0=params, sigma=np.sqrt(varRrs),
                          full_output=False, bounds=bounds, **kwargs)
    # Return
    return ans, cov, idx

def fit_func(wave:np.ndarray, *params, models:list=None,
             return_full:bool=False, rt_dict:dict=None, geom=None):
    """
    Forward model function for curve_fit optimization.

    Computes model Rrs from absorption and backscattering parameters
    using the Gordon radiative transfer approximation, optionally
    including Raman scattering correction.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array [nm]. Note: This parameter is required by
        scipy.curve_fit interface but is not used directly (wavelengths
        come from model objects).
    *params : float
        Variable-length parameter tuple. First models[0].nparam values
        are absorption parameters, remainder are backscattering
        parameters. When rt_dict['fit_Bp'] is True the tuple carries one
        extra trailing element -- (a_params..., bb_params..., B_p)
        (design §3.3), where B_p is the particulate backscattering ratio
        bb_p/b_p in *linear* space; it is peeled off before the
        aparams/bparams split, so the model parameter layout is
        unchanged either way.
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    return_full : bool, optional
        If True, returns (Rrs, a, bb) instead of just Rrs.
        Useful for diagnostics. Default is False.
    rt_dict : dict
        Radiative transfer configuration dictionary. The optional
        'rt_backend' key (default 'gordon' when absent; see
        bing.rt.defs.RT_BACKENDS) selects the forward model: 'gordon'
        keeps the legacy calc_Rrs_from_models call, any robust value
        dispatches to bing.evaluate.calc_Rrs_from_models_robust. The
        optional 'fit_Bp' key (robust backends only; validate_rt_dict
        rejects it for 'gordon' at fit setup): when True, B_p is a free
        parameter riding as the last element of ``params``; it is peeled
        off here and forwarded as the adapter's ``Bp`` argument. When
        False/absent (the default), ``params`` is the plain model vector
        and the adapter falls back to rt_dict['Bp_value'] (Bp=None --
        the fixed-B_p case). Unlike inference.log_prob (which evaluates
        the B_p prior and returns -np.inf out of range), fit_func never
        range-checks the peeled value -- a curve_fit model function must
        return a prediction, and bounds are the optimizer's job (pass
        them via ``fit``'s ``bounds`` argument; the model parameters are
        handled identically).
    geom : bing.rt.geometry.ObsGeometry, optional
        Fixed per-pixel viewing/illumination geometry. Required (non-None)
        whenever rt_dict['rt_backend'] selects a robust backend; ignored
        by the default Gordon backend.

    Returns
    -------
    np.ndarray or tuple
        If return_full=False:
            Rrs : np.ndarray - Predicted remote sensing reflectance [sr^-1]
        If return_full=True:
            Rrs : np.ndarray - Predicted Rrs
            a : np.ndarray - Total absorption coefficient [m^-1]
            bb : np.ndarray - Total backscattering coefficient [m^-1]

    Notes
    -----
    Parameters are split between absorption and backscattering models:
        aparams = params[:models[0].nparam]
        bparams = params[models[0].nparam:]

    The function delegates to evaluate.calc_Rrs_from_models() for the
    actual radiative transfer calculation.
    """

    # B_p tail peel (M3 task 1, design §3.3): when rt_dict['fit_Bp'] is
    # True the incoming tuple is (a_params..., bb_params..., B_p) -- peel
    # the tail *first* so the aparams/bparams split below is untouched.
    # Otherwise Bp stays None and the adapter falls back to
    # rt_dict['Bp_value'] (the fixed-B_p case, M2 Q1).
    if rt_dict is not None and rt_dict.get('fit_Bp', False):
        Bp = params[-1]
        params = params[:-1]
    else:
        Bp = None

    # Unpack for convenience
    aparams = np.array(params[:models[0].nparam])
    bparams = np.array(params[models[0].nparam:])

    # Calculate -- dispatch on the RT backend (design §3.4).  The default
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

    if return_full:
        a = models[0].eval_a(aparams)
        bb = models[1].eval_bb(bparams)
        return pred.flatten(), a.flatten(), bb.flatten()
    else:
        return pred.flatten()