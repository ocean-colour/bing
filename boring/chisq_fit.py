""" Perform a chi-squared fitting instead of Bayesian inference """
import numpy as np

from functools import partial

from scipy.optimize import curve_fit

from boring import rt as boring_rt

from IPython import embed

def fit(items:tuple, models:list):
    """
    Fits the given Rrs data to the specified models using curve fitting.

    Parameters:
        items: A tuple containing the Rrs data, variance of Rrs data, initial parameters, and index.
        models (list): The models to fit the data to.

    Returns:
        ans (np.ndarray): The optimized parameters for the curve fitting.
        cov (np.ndarray): The estimated covariance of ans.
        idx (int): The index of the fitted data. (for book-keeping)

    """
    # Unpack
    Rrs, varRrs, params, idx = items

    partial_func = partial(fit_func, models=models)
    ans, cov =  curve_fit(partial_func, None, 
                          Rrs, p0=params, sigma=np.sqrt(varRrs),
                          full_output=False)
    # Return
    return ans, cov, idx

def fit_func(wave:np.ndarray, *params, models:list=None,
             return_full:bool=False):
    """
    Calculate the predicted values of Rrs based on the given wave array and parameters.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        *params: Variable number of parameters.
        models (list): List of models.
        return_full (bool): Whether to return the full predicted values.

    Returns:
        np.ndarray: Predicted values of Rrs.
    """

    # Unpack for convenience
    aparams = np.array(params[:models[0].nparam])
    bparams = np.array(params[models[0].nparam:])

    # Calculate
    a = models[0].eval_a(aparams)
    bb = models[1].eval_bb(bparams)

    pred = boring_rt.calc_Rrs(a, bb)
    #embed(header='fit_func 33')

    if return_full:
        return pred.flatten(), a.flatten(), bb.flatten()
    else:
        return pred.flatten()