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
"""
import numpy as np

from functools import partial

from scipy.optimize import curve_fit

from bing import evaluate as bing_eval

from IPython import embed

def fit(items:tuple, models:list, rt_dict:dict, bounds:tuple=None):
    """
    Fit Rrs data using Levenberg-Marquardt least-squares optimization.

    Minimizes the weighted chi-squared statistic:
        χ² = Σ[(Rrs_model - Rrs_obs)² / varRrs]

    using scipy.optimize.curve_fit.

    Parameters
    ----------
    items : tuple
        Tuple containing (Rrs, varRrs, params, idx):
        - Rrs : np.ndarray - Observed remote sensing reflectance [sr^-1]
        - varRrs : np.ndarray - Variance of Rrs [sr^-2]
        - params : np.ndarray - Initial parameter guess
        - idx : int - Spectrum index (echoed in return for batch tracking)
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    rt_dict : dict
        Radiative transfer configuration dictionary with keys:
        - 'variable_Gordon' : bool - Use wavelength-dependent Gordon coefficients
        - 'include_Raman' : bool - Include Raman scattering correction
    bounds : tuple, optional
        Parameter bounds as (lower_bounds, upper_bounds) where each is
        a 1D array matching the parameter vector. Default is (-inf, inf).

    Returns
    -------
    ans : np.ndarray
        Best-fit parameters (in model space, typically log10 for amplitudes).
    cov : np.ndarray
        Estimated covariance matrix of the parameters. Diagonal elements
        give variance; sqrt of diagonal gives 1-sigma uncertainties.
    idx : int
        Input index (echoed for batch processing tracking).

    Notes
    -----
    The covariance matrix assumes the model is correct and residuals are
    Gaussian. For more robust uncertainty estimates, use MCMC inference.

    See Also
    --------
    bing.fitting.inference.fit_one : MCMC-based fitting
    fit_func : Forward model function used by curve_fit
    """
    if bounds is None:
        bounds = (-np.inf, np.inf)
    # Unpack
    Rrs, varRrs, params, idx = items

    partial_func = partial(fit_func, models=models, rt_dict=rt_dict)
    ans, cov =  curve_fit(partial_func, None, 
                          Rrs, p0=params, sigma=np.sqrt(varRrs),
                          full_output=False, bounds=bounds)
    # Return
    return ans, cov, idx

def fit_func(wave:np.ndarray, *params, models:list=None,
             return_full:bool=False, rt_dict:dict=None):
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
        are absorption parameters, remainder are backscattering parameters.
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    return_full : bool, optional
        If True, returns (Rrs, a, bb) instead of just Rrs.
        Useful for diagnostics. Default is False.
    rt_dict : dict
        Radiative transfer configuration dictionary.

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

    # Unpack for convenience
    aparams = np.array(params[:models[0].nparam])
    bparams = np.array(params[models[0].nparam:])

    # Calculate
    pred = bing_eval.calc_Rrs_from_models(models[0], aparams, models[1],
        bparams, rt_dict)

    if return_full:
        a = models[0].eval_a(aparams)
        bb = models[1].eval_bb(bparams)
        return pred.flatten(), a.flatten(), bb.flatten()
    else:
        return pred.flatten()