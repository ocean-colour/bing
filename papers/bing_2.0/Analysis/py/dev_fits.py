""" Fits for development of BING 2.0 """

import os

from collections import namedtuple
import numpy as np

from matplotlib import pyplot as plt
import corner

from bing import inference as bing_inf
from bing import plotting as bing_plot

#from xqaa.params import XQAAParams
#from xqaa import retrieve

from IPython import embed

import anly_utils_20 
import param
import prep_for_fits

def fit(p:namedtuple, idx:int, 
        show:bool=False,
        show_xqaa:bool=False,
        burn:int=7000, thin:int=1,
        seed:int=None,
        debug:bool=False):
    """
    Fits a model to the data for a given index.

    Args:
        model_names (list): List of model names.
        idx (int): Index of the data.
        n_cores (int, optional): Number of cores to use. Defaults to 20.
        nsteps (int, optional): Number of steps for MCMC. Defaults to 10000.
        nburn (int, optional): Number of burn-in steps for MCMC. Defaults to 1000.
        use_chisq (bool, optional): Flag to use chi-square fitting. Defaults to False.
        add_noise (bool, optional): Flag to add noise to the data. Defaults to False.
        max_wave (float, optional): Maximum wavelength. Defaults to None.
        show (bool, optional): Flag to show the fit. Defaults to False.
        MODIS (bool, optional): Flag for MODIS data. Defaults to False.
        SeaWiFS (bool, optional): Flag for SeaWiFS data. Defaults to False.
        PACE (bool, optional): Flag for PACE data. Defaults to False.
        show_xqaa (bool, optional): Flag to show xqaa data. Defaults to False.
        set_Sdg (float, optional): Set the Sdg parameter by fitting to a_dg first. 
            Use this value as the uncertainty for the Prior. Defaults to None.
    Returns:
        tuple: Tuple containing the fitted parameters and covariance matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    # Prep and unpack
    prep_dict = prep_for_fits.one_l23(p, idx)
    odict = prep_dict['odict']
    pdict = prep_dict['pdict']
    models = prep_dict['models']
    model_Rrs = prep_dict['model_Rrs']
    model_varRrs = prep_dict['model_varRrs']
    model_wave = models[0].wave
    p0 = prep_dict['p0']
    l23_wave = odict['true_wave']

    # pdict -- this is a hack for a single run
    pdict['Chl'] = np.zeros(idx+1)
    pdict['Chl'][idx] = odict['Chl']
    pdict['Y'] = np.zeros(idx+1)
    pdict['Y'][idx] = odict['Y']

    # Set the items
    #p0 -= 1
    items = [(model_Rrs, model_varRrs, p0, idx)]

    outfile = anly_utils_20.chain_filename(p, idx=idx)

    # Fit
    chains, idx = bing_inf.fit_one(
            items[0], models=models, pdict=pdict, chains_only=True)
    # Save
    anly_utils_20.save_fits(chains, idx, outfile, 
                            extras=dict(wave=model_wave, 
                                        obs_Rrs=model_Rrs, 
                                        varRrs=model_varRrs, 
                                        Chl=odict['Chl'], 
                                        Y=odict['Y']))
    # Show?
    if show:
        if show_xqaa:
            xqaaParams = XQAAParams()
            xq_anw, xq_bbnw, _ = retrieve.iops_from_Rrs(
                l23_wave, gordon_Rrs, xqaaParams)
            # NaN out the anw extremes
            keep = (l23_wave > xqaaParams.amin) & (l23_wave < 600.)
            xq_anw[~keep] = np.nan
            #
            xq_dict = dict(wave=l23_wave, anw=xq_anw, bbnw=xq_bbnw)
        else:
            xq_dict = None

        bing_plot.show_fits(
            models, chains, 
            odict['Chl'], odict['Y'],
            Rrs_true=dict(wave=model_wave, spec=model_Rrs),
            anw_true=dict(wave=l23_wave, spec=odict['anw']),
            bbnw_true=dict(wave=l23_wave, spec=odict['bbnw']),
            xqaa=xq_dict, perc=(16, 84),
            )
        plt.show()

        burn = 7000
        if burn > chains.shape[0]:
            embed(header='210 of dev_fits')
        thin = 1
        coeff = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])

        # Corner plot
        # Labels
        clbls = models[0].pnames + models[1].pnames
        # Add log 10
        clbls = [r'$\log_{10}('+f'{clbl}'+r'$)' for clbl in clbls]
        fig = corner.corner(
            coeff, labels=clbls,
            label_kwargs={'fontsize':17},
            color='k',
            #axes_scale='log',
            truths=None, #truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            )
        # Add 90%
        ss = 0
        for ax in fig.get_axes():
            if len(ax.get_title()) > 0:
                # Calculate the percntile
                p_5, p_95 = np.percentile(coeff[:,ss], [5, 95], axis=0)
                # Plot a vertical line
                ax.axvline(p_5, color='b', linestyle=':')
                ax.axvline(p_95, color='b', linestyle=':')
                ss += 1
        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.show()

        # a_nw
        bing_plot.show_anw_fits(
            models, coeff,
            anw_true=dict(
                wave=l23_wave, a_dg=odict['adg'],
                a_ph=odict['aph']),
            perc=(16, 84))

        if debug:
            embed(header='268 of dev')
            

def main(flg):
    flg = int(flg)

    # NMF
    if flg == 1:
        #fit_one(['ExpNMF', 'Pow'], idx=170, use_chisq=True,
        #        show=True)
        #fit_one(['ExpNMF', 'Lee'], idx=170, use_chisq=True,
        #        show=True, add_noise=True, PACE=True,
        #        scl_noise='PACE', show_xqaa=True)

        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*4
        # Gaussian for Sdg
        #apriors[1] = dict(flavor='gaussian', mean=0.015, sigma=0.001)
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        fit(['ExpNMF', 'Pow'], idx=170, use_chisq=False,
                show=True, add_noise=True, PACE=True,
                scl_noise='PACE', show_xqaa=True,
                apriors=apriors)#, nsteps=50000, nburn=5000)

    # Single fit
    if flg == 2:
        #p = param.p_ntuple(['ExpBricaud', 'Pow'], 
        #    set_Sdg=True, sSdg=0.002, beta=1., 
        #    add_noise=True, wv_min=400.)

        p = param.p_ntuple(['GIOP', 'Lee'], 
            set_Sdg=False, sSdg=0.002, beta=1., 
            add_noise=True, wv_min=400.)
        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3

        # Uniform for Sdg
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        #fit(p, 170, show=True, apriors=apriors) 
        #fit(p, 2532, show=True, apriors=apriors) 
        fit(p, 2773, show=True, apriors=apriors, nsteps=40000) 
            

    # Bricaud + UV (100 trials)
    #  - 100 trials
    #  - Strict prior on Sdg and beta
    if flg == 3:
        p = param.p_ntuple(['ExpBricaud', 'Pow'], 
            set_Sdg=True, sSdg=0.002, beta=1., nMC=100,
            add_noise=True, wv_min=375.)
            #add_noise=True, wv_min=350.)
            #add_noise=True, wv_min=400.)

        # Do it
        fit(p, 170, show=False, 
            nsteps=20000, nburn=2000,
            debug=False)

    # Bricaud + UV (100 trials)
    #  - 100 trials
    #  - Loos prior on Sdg
    if flg == 4:
        #for wv_min in [375., 400]:
        for wv_min in [350.]:#, 375, 400]:
            p = param.p_ntuple(['ExpBricaud', 'Pow'], 
                set_Sdg=False, beta=1., nMC=100,
                add_noise=True, wv_min=wv_min)
                #add_noise=True, wv_min=350.)
                #add_noise=True, wv_min=400.)

            # Priors
            apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
            # Uniform for Sdg
            apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

            # Do it
            fit(p, 170, show=False, 
                nsteps=20000, nburn=2000,
                apriors=apriors, debug=False)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Testing
        #flg += 2 ** 1  # 2 -- No priors
        #flg += 2 ** 2  # 4 -- bb_water

    else:
        flg = sys.argv[1]

    main(flg)
