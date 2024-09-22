""" Fits for development of BING 2.0 """

import os

import numpy as np

from matplotlib import pyplot as plt
import corner

from bing.models import utils as model_utils
from bing import inference as bing_inf
from bing import rt as bing_rt
from bing import chisq_fit
from bing import plotting as bing_plot
from bing import priors as bing_priors

from ocpy.satellites import modis as sat_modis
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs

from xqaa.params import XQAAParams
from xqaa import retrieve

from IPython import embed

import anly_utils_20 

def fit(model_names:list, idx:int, 
        nsteps:int=10000, nburn:int=1000, 
        scl_noise:float=0.02, use_chisq:bool=False,
        add_noise:bool=False,
        min_wave:float=400.,
        max_wave:float=700.,
        show:bool=False,
        MODIS:bool=False,
        SeaWiFS:bool=False,
        PACE:bool=False,
        bbnw_pow:float=None,
        show_xqaa:bool=False,
        apriors:list=None,
        set_Sdg:float=None, 
        set_beta:float=None,
        nMC:int=None,
        debug:bool=False):
    """
    Fits a model to the data for a given index.

    Args:
        model_names (list): List of model names.
        idx (int): Index of the data.
        n_cores (int, optional): Number of cores to use. Defaults to 20.
        nsteps (int, optional): Number of steps for MCMC. Defaults to 10000.
        nburn (int, optional): Number of burn-in steps for MCMC. Defaults to 1000.
        scl_noise (float, optional): Scaling factor for noise. Defaults to 0.02.
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
    odict = anly_utils_20.prep_l23_data(
        idx, scl_noise=scl_noise, min_wave=min_wave, max_wave=max_wave)

    # Set power-law
    if bbnw_pow is not None:
        odict['Y'] = bbnw_pow

    # Unpack
    wave = odict['wave']
    l23_wave = odict['true_wave']

    # Wavelenegths
    if MODIS:
        model_wave = sat_modis.modis_wave
    elif PACE:
        model_wave = anly_utils_20.pace_wave(wv_min=min_wave,
                                             wv_max=max_wave)
    elif SeaWiFS:
        model_wave = sat_seawifs.seawifs_wave
    else:
        model_wave = wave

    # Priors
    if model_names[0] == 'ExpB':
        use_model_names = ['Exp', model_names[1]]
    else:
        use_model_names = model_names.copy()

    # Models
    models = model_utils.init(use_model_names, model_wave)

    # Set priors
    if not use_chisq:
        prior_dict = dict(flavor='log_uniform', pmin=-6, pmax=5)
        for jj in range(2):
            prior_dicts = [prior_dict]*models[jj].nparam
            # Special cases
            if jj == 0 and apriors is not None:
                prior_dicts = apriors
            elif jj == 0 and model_names[0] == 'ExpBricaud':
                prior_dicts[1] = dict(flavor='log_uniform', 
                                    pmin=np.log10(0.007), 
                                    pmax=np.log10(0.02))
            elif jj == 1 and model_names[1] == 'Pow' and set_beta is not None:
                prior_dicts[1] = dict(flavor='gaussian', 
                                    mean=set_beta, sigma=0.1)

            # Sdg
            if set_Sdg is not None and jj==0:
                print(f"Using Sdg = {odict['Sdg']}")
                # Find Sdg
                ii = models[0].pnames.index('Sdg')
                prior_dicts[ii] = dict(flavor='gaussian', 
                                    mean=odict['Sdg'], sigma=set_Sdg)
            # Finish
            models[jj].priors = bing_priors.Priors(prior_dicts)
                    
    # Initialize the MCMC
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn)
    
    # Gordon Rrs
    gordon_Rrs = bing_rt.calc_Rrs(odict['a'], odict['bb'])

    # Internals
    if models[0].uses_Chl:
        models[0].set_aph(odict['Chl'])
    if models[1].uses_basis_params:  # Lee
        models[1].set_basis_func(odict['Y'])

    # Bricaud?
    # Interpolate
    model_Rrs = anly_utils_20.convert_to_satwave(l23_wave, gordon_Rrs, model_wave)
    model_anw = anly_utils_20.convert_to_satwave(l23_wave, odict['anw'], model_wave)
    model_bbnw = anly_utils_20.convert_to_satwave(l23_wave, odict['bbnw'], model_wave)

    model_varRrs = anly_utils_20.scale_noise(scl_noise, model_Rrs, model_wave)

    orig_model_Rrs = model_Rrs.copy()
    if add_noise:
        model_Rrs = anly_utils_20.add_noise(
                orig_model_Rrs, abs_sig=np.sqrt(model_varRrs))

    # Initial guess
    p0_a = models[0].init_guess(model_anw)
    p0_b = models[1].init_guess(model_bbnw)
    p0 = np.concatenate((np.atleast_1d(p0_a), 
                         np.atleast_1d(p0_b)))

    # Log 10
    if use_chisq:
        p0 = np.log10(p0)
    else:
        cnt = 0
        for ss in [0,1]:
            for prior in models[ss].priors.priors:
                if prior.flavor[0:3] == 'log':
                    p0[cnt] = np.log10(p0[cnt])
                cnt += 1

    # Chk initial guess
    ca = models[0].eval_a(p0[0:models[0].nparam])
    cbb = models[1].eval_bb(p0[models[0].nparam:])
    pRrs = bing_rt.calc_Rrs(ca, cbb)
    print(f'Initial Rrs guess: {np.mean((model_Rrs-pRrs)/model_Rrs)}')
    #embed(header='159 of fit one')
    

    # Set the items
    #p0 -= 1
    items = [(model_Rrs, model_varRrs, p0, idx)]

    outfile = anly_utils_20.chain_filename(
        model_names, scl_noise, add_noise, idx=idx,
        MODIS=MODIS, PACE=PACE, SeaWiFS=SeaWiFS,
        beta=set_beta, Sdg=set_Sdg, 
        wv_min=min_wave, nMC=nMC)

    # Bayes
    if not use_chisq:
        # Fit
        if nMC is None:
            chains, idx = bing_inf.fit_one(
                items[0], models=models, pdict=pdict, chains_only=True)
        else:
            chains = []
            for ss in range(nMC):
                print(f'Running {ss} of {nMC}')
                # Error
                if add_noise:
                    model_Rrs = anly_utils_20.add_noise(
                        orig_model_Rrs, abs_sig=np.sqrt(model_varRrs))
                # Run
                ichains, idx = bing_inf.fit_one(
                    items[0], models=models, pdict=pdict, chains_only=True)
                # Save
                chains.append(ichains) 
            chains = np.array(chains)

        # Save
        anly_utils_20.save_fits(chains, idx, outfile, 
                             extras=dict(wave=model_wave, 
                                         obs_Rrs=model_Rrs, 
                                         varRrs=model_varRrs, 
                                         Chl=odict['Chl'], 
                                         Y=odict['Y']))
        ans = None
    else: # chi^2
        # Fit
        #embed(header='ADD BOUNDS FROM PRIORS: fit_one 153')
        ans, cov, idx = chisq_fit.fit(items[0], models)
            
        # Save
        outfile = outfile.replace('BING', 'BING_LM')
        np.savez(outfile, ans=ans, cov=cov,
              wave=wave, obs_Rrs=gordon_Rrs, varRrs=model_varRrs,
              Chl=odict['Chl'], Y=odict['Y'])
        print(f"Saved: {outfile}")

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
            models, 
            ans if ans is not None else chains, 
            odict['Chl'], odict['Y'],
            Rrs_true=dict(wave=model_wave, spec=model_Rrs),
            anw_true=dict(wave=l23_wave, spec=odict['anw']),
            bbnw_true=dict(wave=l23_wave, spec=odict['bbnw']),
            xqaa=xq_dict,
            )
        plt.show()

        if not use_chisq:
            burn = 7000
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
            # Add 95%
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
                    a_ph=odict['aph']))

            if debug:
                embed(header='268 of dev')
            

    if use_chisq:
        return ans, cov


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
        fit_one(['ExpNMF', 'Pow'], idx=170, use_chisq=False,
                show=True, add_noise=True, PACE=True,
                scl_noise='PACE', show_xqaa=True,
                apriors=apriors)#, nsteps=50000, nburn=5000)

    # Bricaud
    if flg == 2:
        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
        # Gaussian for Sdg
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        fit(['ExpBricaud', 'Pow'], idx=170, use_chisq=False,
                show=True, add_noise=True, PACE=True,
                scl_noise='PACE', show_xqaa=True, 
                set_Sdg=0.002,
                set_beta=1.,
                apriors=apriors, debug=True,
                min_wave=350.)#, nsteps=50000, nburn=5000)

    # Bricaud + UV (100 trials)
    if flg == 3:
        min_wave = 400.

        # Priors
        apriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*3
        # Gaussian for Sdg
        apriors[1]=dict(flavor='uniform', pmin=0.01, pmax=0.02)

        # Do it
        fit(
            ['ExpBricaud', 'Pow'], 
            idx=170, use_chisq=False,
            show=False, add_noise=True, PACE=True,
            scl_noise='PACE', 
            set_Sdg=0.002,
            set_beta=1.,
            nMC=100,
            nsteps=20000, nburn=2000,
            apriors=apriors, debug=True,
            min_wave=min_wave)

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