

# RT dict

def rt_dict_from_p(p):
    """
    Prepare data and models for L23 fitting.
    This function initializes the necessary data, models, priors, and MCMC 
    parameters for fitting L23 data. It also handles wavelength conversions, 
    noise scaling, and initial guesses for the fitting process.

    Args:
        p (object): Parameter object containing configuration 
            radiative transfer options
    """

    rt_dict = {}
    for key in ['variable_Gordon', # Enable wavelength-dependent Gordon coefficients
                'include_Raman', # Enable Raman scattering correction
                'variable_Gordon_G0', # Turn on G0
                'include_Chl_fl', # Enable chlorophyll fluorescence
                'phi_C', # Fluorescence quantum yield
                'double_gaussian', # Enable double-Gaussian emission model
            ]:
        if hasattr(p,key):
            rt_dict[key] = getattr(p, key)
        else:
            rt_dict[key] = None

    # Return
    return rt_dict