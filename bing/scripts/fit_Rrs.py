""" Script to use BING to fit an input Table of Rrs values """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Fit Rrs')
    parser.add_argument("table_file", type=str, help="Table of input values.  Required: [wave,Rrs] Optional: [sigRrs,anw,bbnw] (.csv)")
    parser.add_argument("models", type=str, help="Comma separate list of the a, bb models.  e.g. Exp,Cst")
    parser.add_argument("--satellite", type=str, help="Simulate as if observed by the chosen satellite [Aqua, PACE]")
    parser.add_argument("--fit_method", type=str, default='mcmc', help="Simulate as if observed by the chosen satellite [Aqua, PACE]")
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

    from matplotlib import pyplot as plt

    from bing.models import utils as model_utils
    from bing import chisq_fit
    from bing import plotting as bing_plot

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
    if pargs.satellite is not None:
        # Simulate satellite observations
        pass
    else:
        fit_wave = df_input['wave'].values
        fit_Rrs = df_input['Rrs'].values
        fit_sigRrs = df_input['sigRrs'].values
        # Optional
        if 'anw' in df_input.columns:
            fit_anw = df_input['anw'].values
        if 'bbnw' in df_input.columns:
            fit_bbnw = df_input['bbnw'].values
        


    # Initialize the models
    model_names = pargs.models.split(',')
    models = model_utils.init(model_names, fit_wave)

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
        embed(header='Fitting with MCMC 78')
        from bing.fit import mcmc
        fit_results = mcmc.fit(fit_Rrs, fit_sigRrs, p0, models)
    elif pargs.fit_method == 'chisq':
        ans, cov, _ = chisq_fit.fit(items[0], models)
    

    # Plot
    anw_dict = {} if fit_anw is None else dict(wave=fit_wave, spec=fit_anw)
    bbnw_dict = {} if fit_bbnw is None else dict(wave=fit_wave, spec=fit_bbnw)
    bing_plot.show_fit(models, ans, None, None,
                figsize=(9,4),
                fontsize=17.,
                Rrs_true=dict(wave=models[0].wave, spec=fit_Rrs, var=fit_sigRrs**2),
                   #Rrs_true=dict(wave=df_lee.wave, spec=gordon_Rrs_1),
                anw_true=anw_dict, bbnw_true=bbnw_dict,
                )
    plt.show()
    #embed(header='Done fitting')

    # Save