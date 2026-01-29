""" Evaluate the model fits """

import numpy as np

from bing.rt import rrs as bing_rrs
from bing.fitting import chisq_fit

from IPython import embed

def calc_stats(chains, names:list=None, 
               perc=(14, 86)):
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


def reconstruct_from_chains(models:list, chains:np.ndarray, 
                            perc=(5,95)):
    """
    Reconstructs the parameters and calculates statistics from chains of model parameters.

    Parameters:
        - models (list): A list of model objects.
        - chains (ndarray): An array of shape (n_samples, n_chains, n_params) containing the chains of model parameters.
        - perc (tuple): The percentiles to calculate. Default is (5, 95).

    Returns:
        - a_mean (ndarray): The mean of the parameter 'a' across the chains.
        - bb_mean (ndarray): The mean of the parameter 'bb' across the chains.
        - a_low (ndarray): The Xth percentile of the parameter 'a' across the chains.
        - a_high (ndarray): The XXth percentile of the parameter 'a' across the chains.
        - bb_low (ndarray): The Xth percentile of the parameter 'bb' across the chains.
        - bb_high (ndarray): The XXth percentile of the parameter 'bb' across the chains.
        - Rrs (ndarray): The calculated model Rrs.
        - sigRs (ndarray): The standard deviation of Rrs.

    """
    # Burn/thin the chains
    chains = thin_burn_chains(chains)

    # Calc
    a = models[0].eval_a(chains[..., :models[0].nparam])
    bb = models[1].eval_bb(chains[..., models[0].nparam:])
    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_low, a_high = np.percentile(a, perc, axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_low, bb_high = np.percentile(bb, perc, axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    Rrs = bing_rrs.calc_Rrs(a, bb)

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_low, a_high, bb_low, bb_high, Rrs, sigRs 


def reconstruct_chisq_fits(models:list, params:np.ndarray,
                           Chl:np.ndarray=None,
                           bb_basis_params:np.ndarray=None):
    """
    Reconstructs the parameters and calculates statistics from chisq fits.

    Parameters:
        - models (list): A list of model objects.
        - params (ndarray): An array of the best-fit paramerers
            if ndim==1, then it is one fit
            if ndim==2, then it is an (nfits, nparams) array of fits
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
            models[0].wave, *param, models=models, return_full=True)
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
    # Burn/thin the chains
    return chains[burn::thin, :, :].reshape(-1, chains.shape[-1])