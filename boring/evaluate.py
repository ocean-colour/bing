""" Evaluate the model fits """

import numpy as np

from big import rt as big_rt
from big import chisq_fit


def reconstruct_from_chains(models:list, chains, burn=7000, thin=1):
    """
    Reconstructs the parameters and calculates statistics from chains of model parameters.

    Parameters:
        - models (list): A list of model objects.
        - chains (ndarray): An array of shape (n_samples, n_chains, n_params) containing the chains of model parameters.
        - burn (int): The number of burn-in samples to discard from the chains. Default is 7000.
        - thin (int): The thinning factor to apply to the chains. Default is 1.

    Returns:
        - a_mean (ndarray): The mean of the parameter 'a' across the chains.
        - bb_mean (ndarray): The mean of the parameter 'bb' across the chains.
        - a_5 (ndarray): The 5th percentile of the parameter 'a' across the chains.
        - a_95 (ndarray): The 95th percentile of the parameter 'a' across the chains.
        - bb_5 (ndarray): The 5th percentile of the parameter 'bb' across the chains.
        - bb_95 (ndarray): The 95th percentile of the parameter 'bb' across the chains.
        - Rrs (ndarray): The calculated model Rrs.
        - sigRs (ndarray): The standard deviation of Rrs.

    """
    # Burn the chains
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    # Calc
    a = models[0].eval_a(chains[..., :models[0].nparam])
    bb = models[1].eval_bb(chains[..., models[0].nparam:])
    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_5, a_95 = np.percentile(a, [5, 95], axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_5, bb_95 = np.percentile(bb, [5, 95], axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    Rrs = big_rt.calc_Rrs(a, bb)

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_5, a_95, bb_5, bb_95, Rrs, sigRs 


def reconstruct_chisq_fits(models:list, params:np.ndarray,
                           Chl:np.ndarray=None):
    """
    Reconstructs the parameters and calculates statistics from chisq fits.

    Parameters:
        - models (list): A list of model objects.
        - params (ndarray): An array of the best-fit paramerers
            if ndim==1, then it is one fit
            if ndim==2, then it is an (nfits, nparams) array of fits
        - Chl (ndarray): The chlorophyll values to use for the fits. Default is None.


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
        if models[0].name == 'ExpBricaud':
            models[0].set_aph(Chl[ss])
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