""" Generic functions for a and bb """

import numpy as np

from IPython import embed

def constant(wave:np.ndarray, params:np.ndarray):
    """
    Returns a constant value multiplied by an array of ones, with the same shape as the input wave array.

    Parameters:
    wave (np.ndarray): Input array of wave values.
    params (np.ndarray): Input array of parameters.

    Returns:
    np.ndarray: Array of constant values with the same shape as the input wave array.
    """
    return np.outer(10**params[...,0], np.ones_like(wave))

def exponential(wave:np.ndarray, params:np.ndarray, pivot:float=400., S:float=None):
    """
    Calculate the exponential function for a given wave array and parameters.

    Parameters:
        wave (np.ndarray): Array of wave values.
        params (np.ndarray): Array of parameters.
        pivot (float, optional): Pivot value for the exponential function. Default is 400.
        S (float, optional): Scaling factor for the exponential function. Default is None.
            linear, not log10

    Returns:
        np.ndarray: Result of the exponential function calculation.
    """
    Amp = np.outer(10**params[...,0], np.ones_like(wave)) 
    if S is None:
        return Amp * np.exp(np.outer(params[...,1], wave-pivot))
    else:
        return Amp * np.exp(-S*(wave-pivot))

def gaussian(wave:np.ndarray, params:np.ndarray):
    """
    Calculate the Gaussian function for a given wave array and parameters.

    Parameters:
        wave (np.ndarray): Array of wave values.
        params (np.ndarray): Array of parameters.
        pivot (float, optional): Pivot value for the Gaussian function. Default is 400.

    Returns:
        np.ndarray: Result of the Gaussian function calculation.
    """
    #embed(header='gaussia 51')
    Amp = np.outer(10**params[...,0], np.ones_like(wave))
    numer = np.outer(np.ones(params[...,1].size), wave) - np.outer(10**params[...,2], np.ones_like(wave))
    denom = np.outer(10**params[...,1]**2, np.ones_like(wave))
    # Finish
    return Amp * np.exp(-0.5 * numer**2 / denom)

def powerlaw(wave:np.ndarray, params:np.ndarray, pivot:float=600.):
    """
    Calculate the power law function for a given wavelength.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        params (np.ndarray): Array of parameters for the power law function.
        pivot (float, optional): Pivot wavelength. Default is 600.

    Returns:
        np.ndarray: Array of calculated values for the power law function.
    """
    return np.outer(10**params[...,0], np.ones_like(wave)) *\
                    (pivot/wave)**(params[...,1]).reshape(-1,1)

def gen_basis(params: np.ndarray, basis_func_list: list):
    """
    Generate a basis matrix by applying basis functions to the given parameters.

    Parameters:
    - params: numpy.ndarray
        An array of parameters.
    - basis_func_list: list
        A list of basis functions.

    Returns:
    - ans: numpy.ndarray
        The generated basis matrix.
    """
    # Loop on the basis functions
    for ss, basis_func in enumerate(basis_func_list):
        tmp = np.outer(10**params[..., ss], basis_func)
        if ss == 0:
            ans = tmp
        else:
            ans += tmp
    # Return
    return ans