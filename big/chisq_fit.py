""" Perform a chi-squared fitting instead of Bayesian inference """
import numpy as np

from functools import partial

from scipy.optimize import curve_fit

from big import rt as big_rt

from IPython import embed

def fit(items, models):
    # Unpack
    Rrs, varRrs, params, idx = items

    partial_func = partial(fit_func, models=models)
    ans, cov =  curve_fit(partial_func, None, 
                          Rrs, p0=params, sigma=np.sqrt(varRrs),
                          full_output=False)
    # Return
    return ans, cov, idx

def fit_func(wave, *params, models=None):

    # Unpack for convenience
    aparams = np.array(params[:models[0].nparam])
    bparams = np.array(params[models[0].nparam:])

    # Calculate
    a = models[0].eval_a(aparams)
    bb = models[1].eval_bb(bparams)

    pred = big_rt.calc_Rrs(a, bb) 
    #embed(header='fit_func 33')

    return pred.flatten()