""" Generic functions for a and bb """

import numpy as np

def constant(wave:np.ndarray, params:np.ndarray):
    return np.outer(10**params[...,0], np.ones_like(wave))

def exponential(wave:np.ndarray, params:np.ndarray, pivot:float=400.):
    return np.outer(10**params[...,0], np.ones_like(wave)) *\
        np.exp(np.outer(-10**params[...,1], wave-pivot))


def powerlaw(wave:np.ndarray, params:np.ndarray, pivot:float=600.):
    return np.outer(10**params[...,0], np.ones_like(wave)) *\
                    (pivot/wave)**(10**params[...,1]).reshape(-1,1)