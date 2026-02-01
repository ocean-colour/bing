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

def calc_Rrs_from_models(a_model, a_params, bb_model, bb_params, rt_dict:dict):

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
    Rrs = bing_rrs.calc_Rrs(a, bb, in_G1=a_model.G1, in_G2=a_model.G2,
                            a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)
    # Return
    return Rrs

def reconstruct_from_chains(models:list, chains:np.ndarray, rt_dict:dict,
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
    if rt_dict['include_Raman']:
        a_ex = models[0].eval_a_ex(chains[..., :models[0].nparam])
        bb_ex = models[1].eval_bb_ex(chains[..., models[0].nparam:])
        bb_R = np.outer(np.ones(chains.shape[0]), models[1].bb_R)
    else:
        a_ex, bb_ex, bb_R = None, None, None

    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_low, a_high = np.percentile(a, perc, axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_low, bb_high = np.percentile(bb, perc, axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    '''
    from importlib import reload
    from bing.rt import raman

    mu_d = raman.MU_D_DEFAULT
    mu_u = raman.MU_U_DEFAULT
    mu_R = raman.MU_R_DEFAULT
    s_E= 1.
    R_E = raman.calc_R_elastic(a, bb, s_E, mu_d, mu_u)

    R_raman = raman.calc_R_raman_total(
        a, bb, a_ex, bb_ex, bb_R, 1.,
        s_E, mu_d, mu_u, mu_R, True,
    )

    embed(header='81 of evaluate')
    '''
    Rrs = bing_rrs.calc_Rrs(a, bb, in_G1=models[0].G1, in_G2=models[0].G2,
            a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

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
    # Burn/thin the chains
    return chains[burn::thin, :, :].reshape(-1, chains.shape[-1])