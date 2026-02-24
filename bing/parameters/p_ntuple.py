""" Define and generate parameter tuple for BING 2.0 """

from collections import namedtuple

def_dict = dict(model_names=[], # Name of models for a and bb, list
                scl_noise=None,  # Scale noise, float or str
                # Data
                wv_min=400.,       # Minimum wavelength, float
                wv_max=700.,       # Maximum wavelength, float
                satellite='PACE',    # Satellite name, str
                add_noise=False,    # Add noise flag, bool
                # Radiative Transfer
                variable_Gordon=True, # Wavelength dependent Gordon coefficients?
                include_Raman=False, # Include Raman corrections
                include_Chl_fl=False, # Include chlorophyll fluorescence corrections
                # IOPs
                apriors=None,       # Priors for a params, list of dict
                bpriors=None,       # Priors for bb params, list of dict
                othera_priors=None,  # Other a priors, list of dict
                set_Sdg=None,       # Set Sdg flag, bool
                sSdg=None,         # Sdg value, float
                beta=None,         # Beta value, float
                # MCMC
                nMC=None,            # Number of Monte Carlo simulations, int
                nsteps=40000,        # MCMC steps, int
                nburn=1000           # MCMC burn-in, int
    )

def gen(**kwargs):
    """
    Generate a named tuple with parameters for BING 2.0.
    Args:
        **kwargs: Keyword arguments for parameters.
    Returns:
        namedtuple: Named tuple with parameters.
    """
    # Merge default and user-defined parameters
    params = def_dict.copy()
    params.update(kwargs)

    # Scale noise
    if params['scl_noise'] is None:
        params['scl_noise'] = params['satellite']
    #
    MyNamedTuple = namedtuple('BING20_tuple', params.keys())
    p = MyNamedTuple(**params)

    # Return
    return p
