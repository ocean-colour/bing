""" Methods useful to the satellites module """

import numpy as np
from scipy.interpolate import interp1d


def convert_to_satwave(wave:np.ndarray, spec:np.ndarray,
                     sat_wave:np.ndarray):
    """
    Convert the spectrum to MODIS wavelengths

    Parameters:
        wave (np.ndarray): Wavelengths of the input Rrs
        spec (np.ndarray): Spectrum. a, b, Rrs, etc. 
        sat_wave (np.ndarray): Wavelengths of the satellite

    Returns:
        np.ndarray: Rrs at MODIS wavelengths
    """
    # Interpolate
    f = interp1d(wave, spec, kind='linear', fill_value='extrapolate')
    new_spec = f(sat_wave)

    # Return
    return new_spec