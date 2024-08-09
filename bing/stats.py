""" Run stats on the MCMC chains or LM fits """
import numpy as np

from bing import evaluate as bing_eval
#from oceancolor.satellites import modis as sat_modis
#from oceancolor.satellites import seawifs as sat_seawifs

from IPython import embed

def calc_chisq(model_Rrs:np.ndarray, gordon_Rrs:np.ndarray, scl_noise:float,
               noise_term:np.ndarray=None):
    """
    Calculate the chi-square statistic for comparing model Rrs to Gordon Rrs.

    Parameters:
        model_Rrs (np.ndarray): Array of model Rrs values. (nsample, nwave)
        gordon_Rrs (np.ndarray): Array of Gordon Rrs values.
        scl_noise (float): Scaling factor for the noise in Gordon Rrs.

    Returns:
        np.ndarray: Array of chi-square values. one per sample

    """
    if noise_term is None:
        if scl_noise in ['MODIS_Aqua']:
            noise_term = sat_modis.modis_aqua_error
        elif scl_noise == 'SeaWiFS':
            noise_term = sat_seawifs.seawifs_error
        else:
            noise_term = scl_noise*gordon_Rrs
    # Generate the model Rrs
    ichi2 = ((model_Rrs - gordon_Rrs) / noise_term)**2

    # Return
    if model_Rrs.ndim == 1:
        return np.sum(ichi2)
    else:
        return np.sum(ichi2, axis=1)


def calc_ICs(gordon_Rrs:np.ndarray, models:list, params:np.ndarray, 
              scl_noise:float, use_LM:bool=False,
              debug:bool=False, Chl:np.ndarray=None,
              noise_vector:np.ndarray=None,
              bb_basis_params:np.ndarray=None):
    """ Calculate the Akaike and Bayesian Information Criterion
    
    Args:
        gordon_Rrs (np.ndarray): Array of Gordon's remote sensing reflectance values.
        models (list): List of models used for fitting.
        params (np.ndarray): Array of model parameters.
        scl_noise (float): Scaling factor for noise.
        use_LM (bool, optional): Flag indicating whether to use Levenberg-Marquardt algorithm. Defaults to False.
        debug (bool, optional): Flag indicating whether to enable debugging. Defaults to False.
        Chl (np.ndarray, optional): Array of chlorophyll-a concentration values. Defaults to None.
        bb_basis_params (ndarray, optional): The basis parameters to use for the fits. Default is None.
            (nspec, nparams)
    
    Returns:
        float: Bayesian Information Criterion (BIC) value.
    """
    
    if use_LM:
        model_Rrs, _, _ = bing_eval.reconstruct_chisq_fits(
            models, params, Chl=Chl, bb_basis_params=bb_basis_params)
    else:
        raise ValueError("Not ready for MCMC yet")

    # Calculate the chisq
    chi2 = calc_chisq(model_Rrs, gordon_Rrs, scl_noise, noise_term=noise_vector)
            
    nparm = np.sum([model.nparam for model in models])

    # BIC
    BICs = nparm * np.log(model_Rrs.shape[1]) + chi2 

    # AIC
    AICs = 2. * nparm + chi2

    if debug and np.isclose(scl_noise, 0.5):
        embed(header='calc_ICs 38')

    # Return
    return AICs, BICs