""" Standard parameter sets for the Bing model."""

from bing.parameters import p_ntuple

from IPython import embed


def expb_pow(**kwargs):
    """ Parameters for ExpBricaud and Pow models.
    """


    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Uniform for beta from 0. - 2. (positive here means negative slope)
    bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

    # Add kwargs to priors
    params = dict(model_names=['ExpBricaud', 'Pow'],
        apriors=apriors, bpriors=bpriors, 
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def expb_powflex(**kwargs):
    """ Parameters for ExpBricaud and Pow with a FREE (signed) bb slope.

    Identical to :func:`expb_pow` except that the power-law exponent
    ``beta`` may be **negative**, i.e. bb_nw is allowed to be flat or to
    *rise* toward the red instead of strictly decreasing.  That is the
    "white"/mineral limit measured in turbid, mineral-dominated water,
    where the particulate backscattering slope flattens toward
    wavelength independence (Snyder et al. 2008, Appl. Opt. 47, 666;
    Gordon et al. 2009, Opt. Express 17, 16192; Doxaran et al. 2009,
    Limnol. Oceanogr. 54, 1257).

    This widens the *range* of the existing single power law without
    changing its functional *form*, so it serves as the control for the
    turbid-water two-component models: a fit that still fails here
    implicates the form rather than the priors.

    Args:
        **kwargs: Any p_ntuple.gen() field to override, e.g.
            satellite, wv_min, wv_max, nsteps.

    Returns:
        namedtuple: The BING parameter tuple built by p_ntuple.gen.
    """

    # Priors -- as expb_pow, except for beta (below)
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Uniform for beta from -1. - 2.  In (pivot/wave)**beta, beta > 0 is
    # a decreasing bb_nw; beta < 0 lets it rise with wavelength, which
    # expb_pow's floor at 0. forbids.
    bpriors[1]=dict(flavor='uniform', pmin=-1., pmax=2.)

    # Add kwargs to priors
    params = dict(model_names=['ExpBricaud', 'Pow'],
        apriors=apriors, bpriors=bpriors,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def expb_pow2(**kwargs):
    """ Parameters for ExpBricaud and the two-component Pow2 model.

    Pow2 splits particulate backscattering into a near-flat *mineral*
    term (pivot 700 nm) and a steeper *organic* term (pivot 600 nm):

        bb_nw = Bmin*(700/wave)**eta_min + Borg*(600/wave)**eta_org

    which is what turbid, mineral-dominated water requires -- larger
    backscatter magnitude in the red, and a flatter (or rising) spectral
    shape than a single decreasing power law can make (Snyder et al.
    2008; Doxaran et al. 2009; Neukermans et al. 2012).

    The two exponent priors are deliberately kept on **disjoint** ranges
    (mineral <= 0.5 <= organic).  The terms are otherwise exchangeable,
    and a symmetric prior would leave the posterior with a
    label-switching degeneracy.

    Args:
        **kwargs: Any p_ntuple.gen() field to override, e.g.
            satellite, wv_min, wv_max, nsteps.

    Returns:
        namedtuple: The BING parameter tuple built by p_ntuple.gen.
    """

    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*4

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Mineral exponent: flat/"white" limit, allowed to rise (negative)
    bpriors[1]=dict(flavor='uniform', pmin=-0.5, pmax=0.5)
    # Organic exponent: the open-ocean particle slope
    bpriors[3]=dict(flavor='uniform', pmin=0.5, pmax=2.)

    # Add kwargs to priors
    params = dict(model_names=['ExpBricaud', 'Pow2'],
        apriors=apriors, bpriors=bpriors,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def expb_pow2flat(**kwargs):
    """ Parameters for ExpBricaud and the 3-parameter Pow2Flat model.

    As :func:`expb_pow2` but with the mineral exponent fixed at 0, i.e.
    a spectrally flat mineral term:

        bb_nw = Bmin + Borg*(600/wave)**eta_org

    One fewer parameter, and no amplitude/slope degeneracy within the
    mineral term -- the identifiability-safe arm for comparison against
    Pow2.

    Args:
        **kwargs: Any p_ntuple.gen() field to override, e.g.
            satellite, wv_min, wv_max, nsteps.

    Returns:
        namedtuple: The BING parameter tuple built by p_ntuple.gen.
    """

    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Organic exponent: the open-ocean particle slope
    bpriors[2]=dict(flavor='uniform', pmin=0.5, pmax=2.)

    # Add kwargs to priors
    params = dict(model_names=['ExpBricaud', 'Pow2Flat'],
        apriors=apriors, bpriors=bpriors,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def expbf_pow(**kwargs):
    """ Parameters for ExpBricaudFree and Pow models.
    """


    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*4
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Uniform for Sdg from 0.01 - 0.02
    apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

    # Uniform for beta from 0. - 2. (positive here means negative slope)
    bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

    # Add kwargs to priors
    params = dict(model_names=['ExpBricaudFree', 'Pow'],
        apriors=apriors, bpriors=bpriors, 
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)


def giop(**kwargs):
    """ Parameters for GIOP model
    """
    
    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

    # Add kwargs to priors
    params = dict(model_names=['GIOP', 'Lee'],
        apriors=apriors, bpriors=bpriors, 
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def gsm(**kwargs):
    """ Parameters for GSM model
    """

    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

    # Add kwargs to priors
    params = dict(model_names=['GSM', 'GSM'],
        apriors=apriors, bpriors=bpriors, add_noise=True)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

def k2b(**kwargs):
    """ Parameters for K2B model
    """
    params = dict(model_names=['Bricaud', 'Cst'], nsteps=500000,
        apriors=None, bpriors=None, add_noise=True,
                  sSdg=0.002, set_Sdg=False)
    params.update(kwargs)

    # Generate and return parameters
    return p_ntuple.gen(**params)

#def every_every(nsteps:int=500000):
#
#    p = p_ntuple.gen(['Every', 'Every'],
#        set_Sdg=False, sSdg=0.002, apriors=None, bpriors=None,
#        scl_noise=0.02, nsteps=nsteps,
#        add_noise=False, wv_min=400., wv_max=700.)
#
#    return p
