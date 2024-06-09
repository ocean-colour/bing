""" Generic functions for a and bb """

import numpy as np

from IPython import embed

def constant(wave:np.ndarray, params:np.ndarray):
    return np.outer(10**params[...,0], np.ones_like(wave))

def exponential(wave:np.ndarray, params:np.ndarray, pivot:float=400., S:float=None):
    Amp = np.outer(10**params[...,0], np.ones_like(wave)) 
    if S is None:
        return Amp * np.exp(np.outer(-10**params[...,1], wave-pivot))
    else:
        return Amp * np.exp(-S*(wave-pivot))

def powerlaw(wave:np.ndarray, params:np.ndarray, pivot:float=600.):
    return np.outer(10**params[...,0], np.ones_like(wave)) *\
                    (pivot/wave)**(10**params[...,1]).reshape(-1,1)

def gen_basis(params:np.ndarray, basis_func_list:list):
    # Loop on the basis functions
    for ss, basis_func in enumerate(basis_func_list):
        tmp = np.outer(10**params[...,ss], basis_func)
        if ss == 0:
            ans = tmp
        else:
            ans += tmp
    # Return
    return ans