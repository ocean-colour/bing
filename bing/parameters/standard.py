""" Standard parameter sets for the Bing model."""

from bing.parameters import p_ntuple

from IPython import embed


def expb_pow(**kwargs):

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

def giop(**kwargs):
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
    # Priors
    apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*1

    # Add kwargs to priors
    params = dict(model_names=['GSM', 'GSM'],
        apriors=apriors, bpriors=bpriors, add_noise=True)
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
