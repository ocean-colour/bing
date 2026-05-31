"""
Post-Fitting Evaluation Module for BING
========================================

This module provides functions for analyzing and evaluating bio-optical
parameter retrievals from MCMC chains or least-squares fits.

Key functionality:
- Computing statistics (median, percentiles) from MCMC chains
- Reconstructing IOPs (absorption, backscattering) from parameters
- Computing model Rrs with uncertainties
- Processing chain arrays (burn-in removal, thinning)

The module bridges the gap between raw fitting outputs and scientific
analysis, providing properly formatted results with uncertainties.

Examples
--------
>>> from bing import evaluate
>>> from bing.models import utils as model_utils
>>>
>>> # Reconstruct IOPs from MCMC chains
>>> a_mean, bb_mean, a_lo, a_hi, bb_lo, bb_hi, Rrs, sigRrs = \\
...     evaluate.reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))
>>>
>>> # Get parameter statistics
>>> stats = evaluate.calc_stats(chains, names=['Adg', 'Sdg', 'Aph', 'Bnw', 'beta'])
"""

import numpy as np

from bing.rt import rrs as bing_rrs
from bing.fitting import chisq_fit

from IPython import embed

def calc_stats(chains, names:list=None,
               perc=(14, 86)):
    """
    Compute summary statistics from MCMC chains.

    Calculates median and percentile bounds for each parameter after
    applying burn-in removal and thinning.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains with shape (nsteps, nwalkers, nparam).
    names : list of str, optional
        Parameter names. If None, uses generic names ['p0', 'p1', ...].
    perc : tuple, optional
        Lower and upper percentiles for credible interval. Default is (14, 86),
        corresponding to approximately 1-sigma for a Gaussian.

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - 'names' : list of str - Parameter names
        - 'med' : np.ndarray - Median values for each parameter
        - 'pXX' : np.ndarray - Lower percentile values (e.g., 'p14')
        - 'pYY' : np.ndarray - Upper percentile values (e.g., 'p86')

    Notes
    -----
    Chains are automatically processed with thin_burn_chains() which
    removes burn-in (default 7000 steps) and flattens walker dimension.

    Examples
    --------
    >>> stats = calc_stats(chains, names=['Adg', 'Sdg', 'Aph', 'Bnw', 'beta'])
    >>> print(f"Adg = {10**stats['med'][0]:.4f} "
    ...       f"[{10**stats['p14'][0]:.4f}, {10**stats['p86'][0]:.4f}]")
    """
    # Thin/burn
    chains = thin_burn_chains(chains)

    # Names
    if names is None:
        names = [f'p{ii}' for ii in range(chains.shape[1])]

    # Simple stats
    stats = {}
    stats['names'] = names

    stats['med'] = np.median(chains, axis=0)
    stats[f'p{perc[0]:02d}'] = np.percentile(chains, perc[0], axis=0)
    stats[f'p{perc[1]:02d}'] = np.percentile(chains, perc[1], axis=0)

    return stats

def calc_Rrs_from_models(a_model, a_params, bb_model, bb_params, 
        rt_dict:dict):
    """
    Calculate Rrs from model parameters using Gordon radiative transfer.

    This is the central forward model function that computes remote sensing
    reflectance from absorption and backscattering model parameters, optionally
    including wavelength-dependent Gordon coefficients and Raman correction.

    Parameters
    ----------
    a_model : aNWModel
        Absorption model object (e.g., aNWExpBricaud).
    a_params : np.ndarray
        Absorption model parameters. Shape can be (nparam,) for single
        evaluation or (nsamples, nparam) for batch evaluation.
    bb_model : bbNWModel
        Backscattering model object (e.g., bbNWPow).
    bb_params : np.ndarray
        Backscattering model parameters. Shape matches a_params.
    rt_dict : dict
        Radiative transfer configuration with keys:
        - 'variable_Gordon' : bool - Use wavelength-dependent G1, G2
        - 'include_Raman' : bool - Apply Raman scattering correction

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs [sr^-1]. Shape is (nwave,) for
        single evaluation or (nsamples, nwave) for batch.

    Raises
    ------
    ValueError
        If variable_Gordon=True but model G1/G2 are not set.

    Notes
    -----
    When include_Raman=True, the function computes IOPs at both emission
    and excitation wavelengths to calculate the Raman correction factor.

    See Also
    --------
    bing.rt.rrs.calc_Rrs : Low-level Rrs calculation
    reconstruct_from_chains : Higher-level function using this internally
    """
    # IOPs for model wave
    a = a_model.eval_a(a_params)
    bb = bb_model.eval_bb(bb_params)

    # Elastic
    if rt_dict['variable_Gordon'] and a_model.G1 is None:
        raise ValueError("Need to set model G1, G2 for variable Gordon")

    # Raman?
    if rt_dict['include_Raman']:
        # a_ex, bb_ex
        a_ex = a_model.eval_a_ex(a_params)
        bb_ex = bb_model.eval_bb_ex(bb_params)
        bb_R = bb_model.bb_R
    else:
        a_ex, bb_ex, bb_R = None, None, None

    # Call me
    Rrs = bing_rrs.calc_Rrs(a, bb,
                            in_G1=a_model.G1, in_G2=a_model.G2,
                            in_G0=getattr(a_model, 'G0', None),
                            a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

    # Fluorescence? Accept rt_dicts that don't specify the key (ad-hoc dicts
    # built by tests / notebooks pre-date the include_Chl_fl field).
    if rt_dict.get('include_Chl_fl', False):
        # a_ph
        aph = (10**a_params[...,-1:]) * a_model.a_ph
        aph_ex = aph[a_model.i_Chl_ex]

        # Call me
        Rrs_fl = bing_rrs.calc_Rrs_fluorescence(
            a_model.wave, a, bb,
            a[:,a_model.i_Chl_ex],
            bb[:,a_model.i_Chl_ex],
            aph_ex,
            a_model.wave[a_model.i_Chl_ex],
            a_model.Ed_ex,
            a_model.Ed_em,
            phi_C=rt_dict['phi_C'],
            double_gaussian=rt_dict['double_gaussian'])
        # Add
        Rrs += Rrs_fl

    # Return
    return Rrs

def reconstruct_from_chains(models:list, chains:np.ndarray, rt_dict:dict,
                            perc=(5,95)):
    """
    Reconstruct IOPs and Rrs with uncertainties from MCMC chains.

    Evaluates the absorption and backscattering models for all chain samples
    to compute posterior distributions of IOPs and Rrs, then summarizes with
    median and percentile statistics.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    chains : np.ndarray
        MCMC chains with shape (nsteps, nwalkers, nparam).
    rt_dict : dict
        Radiative transfer configuration dictionary.
    perc : tuple, optional
        Percentiles for credible interval bounds. Default is (5, 95),
        giving a 90% credible interval.

    Returns
    -------
    a_mean : np.ndarray
        Median total absorption coefficient at each wavelength [m^-1].
    bb_mean : np.ndarray
        Median total backscattering coefficient at each wavelength [m^-1].
    a_low : np.ndarray
        Lower percentile of absorption [m^-1].
    a_high : np.ndarray
        Upper percentile of absorption [m^-1].
    bb_low : np.ndarray
        Lower percentile of backscattering [m^-1].
    bb_high : np.ndarray
        Upper percentile of backscattering [m^-1].
    Rrs : np.ndarray
        Median model Rrs at each wavelength [sr^-1].
    sigRrs : np.ndarray
        Standard deviation of Rrs at each wavelength [sr^-1].

    Notes
    -----
    Chains are processed with thin_burn_chains() before evaluation.
    This removes burn-in and flattens the walker dimension.

    The function handles Raman correction if rt_dict['include_Raman']=True,
    computing IOPs at both emission and excitation wavelengths.

    Memory usage can be significant for long chains since all samples
    are evaluated simultaneously.

    Examples
    --------
    >>> a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs, sigRrs = \\
    ...     reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))
    >>> plt.fill_between(wave, a_lo, a_hi, alpha=0.3)
    >>> plt.plot(wave, a_med)
    """
    # Burn/thin the chains
    chains = thin_burn_chains(chains)

    # Calc
    a = models[0].eval_a(chains[..., :models[0].nparam])
    bb = models[1].eval_bb(chains[..., models[0].nparam:])
    if rt_dict['include_Raman']:
        a_ex = models[0].eval_a_ex(chains[..., :models[0].nparam])
        bb_ex = models[1].eval_bb_ex(chains[..., models[0].nparam:])
        bb_R = np.outer(np.ones(chains.shape[0]), models[1].bb_R)
    else:
        a_ex, bb_ex, bb_R = None, None, None

    # Make a_ph before deleting chains
    if rt_dict.get('include_Chl_fl', False):
        aph = (10**chains[...,models[0].nparam-1:models[0].nparam]) * models[0].a_ph
        aph_ex = aph[...,models[0].i_Chl_ex]

    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_low, a_high = np.percentile(a, perc, axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_low, bb_high = np.percentile(bb, perc, axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    Rrs = bing_rrs.calc_Rrs(a, bb,
            in_G1=models[0].G1, in_G2=models[0].G2,
            in_G0=getattr(models[0], 'G0', None),
            a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

    if rt_dict.get('include_Chl_fl', False):
        #embed(header='268 of evaluate.py')
        # Call me
        Rrs_fl = bing_rrs.calc_Rrs_fluorescence(
            models[0].wave, a, bb,
            a[:,models[0].i_Chl_ex],
            bb[:,models[0].i_Chl_ex],
            aph_ex, 
            np.outer(np.ones(a.shape[0]), models[0].wave[models[0].i_Chl_ex]),
            np.outer(np.ones(a.shape[0]), models[0].Ed_ex),
            models[0].Ed_em,
            phi_C=rt_dict['phi_C'],
            double_gaussian=rt_dict['double_gaussian'])
        # Add
        Rrs += Rrs_fl

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_low, a_high, bb_low, bb_high, Rrs, sigRs 


def reconstruct_chisq_fits(models:list, params:np.ndarray, rt_dict:dict,
                           Chl:np.ndarray=None,
                           bb_basis_params:np.ndarray=None):
    """
    Reconstructs the parameters and calculates statistics from chisq fits.

    Parameters:
        - models (list): A list of model objects.
        - params (ndarray): An array of the best-fit paramerers
            if ndim==1, then it is one fit
            if ndim==2, then it is an (nfits, nparams) array of fits
        - rt_dict (dict): dict describing the Radiative transfer
        - Chl (ndarray): The chlorophyll values to use for the fits. Default is None.
        - bb_basis_params (ndarray): The basis parameters to use for the fits. Default is None.
            (nspec, nparams)


    Returns:
        - a_mean (ndarray): The mean of the parameter 'a' across the fits.
        - bb_mean (ndarray): The mean of the parameter 'bb' across the fits.
        - a_5 (ndarray): The 5th percentile of the parameter 'a' across the fits.
        - a_95 (ndarray): The 95th percentile of the parameter 'a' across the fits.
        - bb_5 (ndarray): The 5th percentile of the parameter 'bb' across the fits.
        - bb_95 (ndarray): The 95th percentile of the parameter 'bb' across the fits.
        - Rrs (ndarray): The calculated model Rrs.
        - sigRs (ndarray): The standard deviation of Rrs.

    """
    all_Rrs = []
    all_a = []
    all_bb = []
    # Fit
    in_ndim = params.ndim
    if params.ndim == 1:
        params = params.reshape(1, -1)

    for ss, param in enumerate(params):
        # Chl?
        if models[0].uses_Chl:
            models[0].set_aph(np.atleast_1d(Chl)[ss])
        # Lee?
        if models[1].uses_basis_params:
            models[1].set_basis_func(np.atleast_1d(bb_basis_params)[ss])
        model_Rrs, a_mean, bb_mean = chisq_fit.fit_func(
            models[0].wave, *param, models=models, return_full=True,
            rt_dict=rt_dict)
        # Save
        all_Rrs.append(model_Rrs)
        all_a.append(a_mean)
        all_bb.append(bb_mean)

    # Flatten?
    if in_ndim == 1:
        all_Rrs = all_Rrs[0]
        all_a = all_a[0]
        all_bb = all_bb[0]

    # Return
    return np.array(all_Rrs), np.array(all_a), np.array(all_bb)


def thin_burn_chains(chains:np.ndarray,
                     burn:int=7000, thin:int=1):
    """
    Remove burn-in and thin MCMC chains.

    Processes raw MCMC chains by removing initial burn-in samples,
    applying optional thinning, and flattening the walker dimension.

    Parameters
    ----------
    chains : np.ndarray
        Raw MCMC chains with shape (nsteps, nwalkers, nparam).
    burn : int, optional
        Number of initial steps to discard as burn-in. Default is 7000.
    thin : int, optional
        Thinning factor (keep every thin-th sample). Default is 1 (no thinning).

    Returns
    -------
    np.ndarray
        Processed chains with shape (nsamples, nparam), where
        nsamples = (nsteps - burn) // thin * nwalkers.

    Notes
    -----
    The walker dimension is flattened, treating all walkers as independent
    samples from the posterior. This is valid after burn-in when walkers
    have converged to sampling the same distribution.

    Examples
    --------
    >>> chains.shape
    (40000, 16, 5)
    >>> processed = thin_burn_chains(chains, burn=7000, thin=1)
    >>> processed.shape
    (528000, 5)  # (40000-7000) * 16
    """
    # Burn/thin the chains
    return chains[burn::thin, :, :].reshape(-1, chains.shape[-1])