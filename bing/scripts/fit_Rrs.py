""" Script to use BING to fit an input Table of Rrs values """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Fit Rrs')
    parser.add_argument("table_file", type=str, help="Table of input values.  Required: [wave,Rrs] Optional: [sigRrs,anw,bbnw] (.csv)")
    parser.add_argument("models", type=str, help="Comma separate list of the a, bb models.  e.g. Exp,Cst")
    parser.add_argument("--outroot", type=str, help="Save outputs to this root")
    parser.add_argument("--satellite", type=str, help="Simulate as if observed by the chosen satellite [Aqua, PACE]")
    parser.add_argument("--fit_method", type=str, default='mcmc', help="Method for fitting [mcmc, chisq]")
    #parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")


    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import pandas
    from scipy.interpolate import interp1d

    from matplotlib import pyplot as plt

    from ocpy.satellites import pace as sat_pace

    from bing.models import utils as model_utils
    from bing import chisq_fit
    from bing import plotting as bing_plot
    from bing import priors as bing_priors
    from bing import inference as bing_inf

    # Load up the input table
    df_input = pandas.read_csv(pargs.table_file)

    # Checks
    if 'wave' not in df_input.columns:
        raise ValueError('Input table must contain a "wave" column.')
    if 'Rrs' not in df_input.columns:
        raise ValueError('Input table must contain an "Rrs" column.')
    if pargs.satellite is not None and pargs.satellite not in ['Aqua', 'PACE']:
        raise ValueError('Satellite must be either "Aqua" or "PACE".')
    else:
        if 'sigRrs' not in df_input.columns: 
            raise ValueError('Input table must contain a "sigRrs" column if not simulating satellite observations.')

    # Process input

    # Simulate satellite observations?
    if pargs.satellite is not None:
        if pargs.satellite == 'Aqua':
            # Load Aqua error
            raise ValueError('Not ready for Aqua')
        elif pargs.satellite == 'PACE':
            fit_wave = np.arange(400, 701, 5)
            fit_sigRrs = sat_pace.gen_noise_vector(fit_wave)

        # Interpolate to fit_wave
        f = interp1d(df_input['wave'], df_input['Rrs'])
        fit_Rrs = f(fit_wave)
    else:
        fit_wave = df_input['wave'].values
        fit_Rrs = df_input['Rrs'].values
        fit_sigRrs = df_input['sigRrs'].values

    # Optional
    if 'anw' in df_input.columns:
        f = interp1d(df_input['wave'], df_input['anw'])
        fit_anw = f(fit_wave)
    if 'bbnw' in df_input.columns:
        f = interp1d(df_input['wave'], df_input['bbnw'])
        fit_bbnw = f(fit_wave)
        

    # Initialize the models
    model_names = pargs.models.split(',')
    models = model_utils.init(model_names, fit_wave)

    # Internals
    if models[0].uses_Chl:
        #models[0].set_aph(odict['Chl'])
        raise ValueError('Not ready for fitting with Chl')
    if models[1].uses_basis_params:  # Lee
        raise ValueError('Not ready for fitting with Y')
        #models[1].set_basis_func(odict['Y'])

    # Set priors
    if pargs.fit_method == 'mcmc':
        # Default
        prior_dict = bing_priors.default
        for jj in range(2):
            prior_dicts = [prior_dict]*models[jj].nparam
            models[jj].priors = bing_priors.Priors(prior_dicts)
        # Initialize the MCMC
        nsteps:int=10000 
        nburn:int=1000 
        pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)
                

    # Initial guess
    if fit_anw is not None and fit_bbnw is not None:
        p0_a = models[0].init_guess(fit_anw)
        p0_b = models[1].init_guess(fit_bbnw)
        p0 = np.concatenate((np.log10(np.atleast_1d(p0_a)), 
                            np.log10(np.atleast_1d(p0_b))))
    else:
        raise ValueError('Not ready for fitting without anw and bbnw for the initial guess')

    # Fit
    items = [(fit_Rrs, fit_sigRrs**2, p0, None)]
    if pargs.fit_method == 'mcmc':
        # Fit
        chains, _ = bing_inf.fit_one(items[0], models=models, pdict=pdict, chains_only=True)
        ans = None
    elif pargs.fit_method == 'chisq':
        ans, cov, _ = chisq_fit.fit(items[0], models)
    

    # Plot
    anw_dict = {} if fit_anw is None else dict(wave=fit_wave, spec=fit_anw)
    bbnw_dict = {} if fit_bbnw is None else dict(wave=fit_wave, spec=fit_bbnw)
    bing_plot.show_fits(models, ans if ans is not None else chains, 
                       None, None,
                figsize=(9,4),
                fontsize=17.,
                Rrs_true=dict(wave=models[0].wave, spec=fit_Rrs, var=fit_sigRrs**2),
                   #Rrs_true=dict(wave=df_lee.wave, spec=gordon_Rrs_1),
                anw_true=anw_dict, bbnw_true=bbnw_dict,
                )
    plt.show()
    #embed(header='Done fitting')

    # Save
    if pargs.outroot is not None:
        outfile = pargs.outroot + '.npz'

        # Chisq
        outdict = {}
        if pargs.fit_method == 'mcmc':
            outdict['chains'] = chains
        elif pargs.fit_method == 'chisq':
            outdict['ans'] = ans
            outdict['cov'] = cov    

        # Extras
        outdict['wave'] = models[0].wave
        outdict['Rrs'] = fit_Rrs
        outdict['sigRrs'] = fit_sigRrs
        if fit_anw is not None:
            outdict['anw'] = fit_anw
        if fit_bbnw is not None:
            outdict['bbnw'] = fit_bbnw

        # Do it
        np.savez(outfile, **outdict)
        print("Wrote: {:s}".format(outfile))