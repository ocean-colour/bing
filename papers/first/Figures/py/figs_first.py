""" Figs for Gordon Analyses """
import os, sys

import numpy as np

from scipy.optimize import curve_fit
import pandas


from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import corner

from oceancolor.utils import plotting 
from oceancolor.water import absorption
from oceancolor.hydrolight import loisel23


from big import rt as big_rt
from big.models import anw as big_anw
from big.models import bbnw as big_bbnw
from big import chisq_fit
from big import stats as big_stats

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import anly_utils

from IPython import embed


def get_chain_file(model_names, scl_noise, add_noise, idx,
                   use_LM=False):
    scl_noise = 0.02 if scl_noise is None else scl_noise
    noises = f'{int(100*scl_noise):02d}'
    noise_lbl = 'N' if add_noise else 'n'
    chain_file = f'../Analysis/Fits/BIG_{model_names[0]}{model_names[1]}_{idx}_{noise_lbl}{noises}.npz'
    # LM
    if use_LM:
        chain_file = chain_file.replace('BIG', 'BIG_LM')
    return chain_file, noises, noise_lbl

from IPython import embed


def fig_u(outfile='fig_u.png'):
    # Load
    ds = loisel23.load_ds(4,0)
    # Unpack
    wave = ds.Lambda.data
    Rrs = ds.Rrs.data
    a = ds.a.data
    bb = ds.bb.data
    # u
    u = bb / (a+bb)
    # rrs
    A, B = 0.52, 1.17
    rrs = Rrs / (A + B*Rrs)
    # Select wavelengths
    i370 = np.argmin(np.abs(wave-370.))
    i440 = np.argmin(np.abs(wave-440.))
    i500 = np.argmin(np.abs(wave-500.))
    i600 = np.argmin(np.abs(wave-600.))

    # Gordon
    G1, G2 = 0.0949, 0.0794  # Gordon

    def rrs_func(uval, G1, G2):
        rrs = G1*uval + G2*uval**2
        return rrs

    # GIOP
    uval = np.linspace(0., 0.25, 1000)
    rrs_GIOP = rrs_func(uval, G1, G2)
    Rrs_GIOP = A*rrs_GIOP / (1 - B*rrs_GIOP)

    # Fit
    save_ans = []
    for ii in [i370, i440, i500, i600]:
        ans, cov = curve_fit(rrs_func, u[:,ii], rrs[:,ii], p0=[0.1, 0.1], sigma=np.ones_like(u[:,ii])*0.0003)
        save_ans.append(ans)

    #
    fig = plt.figure(figsize=(9,5))

    plt.clf()
    ax = plt.gca()
    for lbl, clr, idx, ans in zip(['370nm', '440nm', '500nm', '600nm'],
                                ['purple', 'b','g', 'r'],
                                [i370, i440, i500, i600],
                                save_ans):
        ax.scatter(u[:,idx], rrs[:,idx], color=clr, s=1., label=lbl)
        irrs = rrs_func(u[:,idx], ans[0], ans[1])
        usrt = np.argsort(u[:,idx])
        ax.plot(u[usrt,idx], irrs[usrt], '-', color=clr, label=f'Fit: G0={ans[0]:0.2f}, G1={ans[1]:0.2f}')
    # GIOP
    ax.plot(uval, rrs_GIOP, 'k--', label='Gordon')
    #
    ax.set_xlabel(r'$u_\lambda$')
    ax.set_ylabel(r'$r_{\rm rs}$')
    ax.legend(fontsize=12)
    plotting.set_fontsize(ax, 13.)
    #
    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_mcmc_fit(model_names:list, idx:int=170, chain_file=None,
                 outroot='fig_BIG_fit', show_bbnw:bool=True,
                 add_noise:bool=False, log_Rrs:bool=False,
                 show_trueRrs:bool=False,
                 wstep:int=1, use_LM:bool=False,
                 set_abblim:bool=True, scl_noise:float=None): 

    chain_file, noises, noise_lbl = get_chain_file(
        model_names, scl_noise, add_noise, idx, use_LM=use_LM)
    d_chains = np.load(chain_file)


    # Load the data
    odict = anly_utils.prep_l23_data(idx, step=wstep)
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a_true = odict['a']
    bb_true = odict['bb']
    aw = odict['aw']
    adg = odict['adg']
    aph = odict['aph']
    bbw = odict['bbw']
    bbnw = bb_true - bbw
    wave_true = odict['true_wave']
    Rrs_true = odict['true_Rrs']

    gordon_Rrs = big_rt.calc_Rrs(odict['a'][::wstep], odict['bb'][::wstep])

    # Init the models
    anw_model = big_anw.init_model(model_names[0], wave, 'log')
    bbnw_model = big_bbnw.init_model(model_names[1], wave, 'log')
    models = [anw_model, bbnw_model]

    # Interpolate
    aw_interp = np.interp(wave, wave_true, aw)
    bbw_interp = np.interp(wave, wave_true, bbw)

    # Reconstruc
    if use_LM:
        model_Rrs, a_mean, bb_mean = chisq_fit.fit_func(
            wave, *d_chains['ans'], models=models, return_full=True)
    else:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = anly_utils.reconstruct(
            models, d_chains['chains']) 

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')

    # Outfile
    outfile = outroot + f'_{model_names[0]}{model_names[1]}_{idx}_{noise_lbl}{noises}.png'
    if use_LM:
        outfile = outfile.replace('BIG', 'BIG_LM')

    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,3)
    

    '''
    # #########################################################
    # a with water

    ax_aw = plt.subplot(gs[0])
    ax_aw.plot(wave_true, a_true, 'ko', label='True', zorder=1)
    ax_aw.plot(wave, a_mean, 'r-', label='Retreival')
    ax_aw.fill_between(wave, a_5, a_95,
            color='r', alpha=0.5, label='Uncertainty') 
    #ax_a.set_xlabel('Wavelength (nm)')
    ax_aw.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')
    #else:
    #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    ax_aw.legend(fontsize=lgsz)
    ax_aw.set_ylim(bottom=0., top=2*a_true.max())
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels

    # Add model, index as text
    ax_aw.text(0.1, 0.9, f'Index: {idx}', fontsize=15, transform=ax_aw.transAxes,
            ha='left')
    ax_aw.text(0.1, 0.8, f'Model: {model_names[0]}{model_names[1]}', fontsize=15, transform=ax_aw.transAxes,
            ha='left')
    '''


    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[1])
    ax_anw.plot(wave_true, a_true-aw, 'ko', label='True', zorder=1)
    ax_anw.plot(wave, a_mean-aw_interp, 'r-', label='Retreival')
    if not use_LM:
        ax_anw.fill_between(wave, a_5-aw_interp, a_95-aw_interp, 
            color='r', alpha=0.5, label='Uncertainty') 
    
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    ax_anw.plot(wave_true, adg, '-', color='brown', label=r'$a_{\rm dg}$')
    ax_anw.plot(wave_true, aph, 'b-', label=r'$a_{\rm ph}$')

    #else:
    #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    ax_anw.legend(fontsize=10.)
    if set_abblim:
        ax_anw.set_ylim(bottom=0., top=2*(a_true-aw).max())
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels


    # #########################################################
    # b
    ax_bb = plt.subplot(gs[2])
    if show_bbnw:
        use_bbw = bbw[::wstep]
        show_bb = bbnw
    else:
        use_bbw = 0.
        show_bb = bbnw
    ax_bb.plot(wave_true, show_bb, 'ko', label='True')
    ax_bb.plot(wave, bb_mean-use_bbw, 'g-', label='Retrieval')
    if not use_LM:
        ax_bb.fill_between(wave, bb_5-use_bbw, bb_95-use_bbw,
            color='g', alpha=0.5, label='Uncertainty') 

    #ax_bb.set_xlabel('Wavelength (nm)')
    if show_bbnw:
        ax_bb.set_ylabel(r'$b_bnw(\lambda) \; [{\rm m}^{-1}]$')
    else:
        ax_bb.set_ylabel(r'$b_b(\lambda) \; [{\rm m}^{-1}]$')

    ax_bb.legend(fontsize=lgsz)
    if set_abblim:
        ax_bb.set_ylim(bottom=0., top=2*show_bb.max())


    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0])
    if show_trueRrs:
        ax_R.plot(wave_true, Rrs_true, 'kx', label='True L23')
    ax_R.plot(wave, gordon_Rrs, 'k+', label='L23 + Gordon')
    ax_R.plot(wave, model_Rrs, 'r-', label='Fit', zorder=10)
    if not use_LM:
        ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
            color='r', alpha=0.5, zorder=10) 

    if add_noise:
        ax_R.plot(d_chains['wave'], d_chains['obs_Rrs'], 'bs', label='Observed')

    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')

    ax_R.legend(fontsize=lgsz)
    
    # Log scale y-axis
    if log_Rrs:
        ax_R.set_yscale('log')
    else:
        ax_R.set_ylim(bottom=0., top=1.1*Rrs_true.max())
    
    # axes
    for ss, ax in enumerate([ax_anw, ax_R, ax_bb]):
        plotting.set_fontsize(ax, 14)
        if ss > 1:
            ax.set_xlabel('Wavelength (nm)')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_multi_fits(models:list=None, indices:list=None, 
                 outroot='fig_multi_fits', show_bbnw:bool=False,
                 add_noise:bool=False, log_Rrs:bool=False,
                 show_trueRrs:bool=False,
                 set_abblim:bool=True, scl_noise:float=None): 

    if models is None:
        models = ['cstcst', 'expcst', 'exppow', 'giop+']
    if indices is None:
        indices = [170, 1032]
    outfile = outroot + f'_{indices[0]}_{indices[1]}.png'

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,3)

    compare_models(models, indices[0], 
                   [plt.subplot(gs[0]), plt.subplot(gs[1]), 
                    plt.subplot(gs[2])],
                   lbl_wavelengths=False)
    compare_models(models, indices[1], 
                   [plt.subplot(gs[3]), plt.subplot(gs[4]), 
                    plt.subplot(gs[5])])

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def compare_models(models:list, idx:int, axes:list,
                   add_noise:bool=False, scl_noise:float=None,
                   log_Rrs:bool=True, lbl_wavelengths:bool=True):

    # Loop on models
    for ss, clr, model in zip(range(len(models)), ['r', 'g', 'b', 'orange'], models):
        nparm = fgordon.grab_priors(model).shape[0]
        chain_file, noises, noise_lbl = get_chain_file(model, scl_noise, add_noise, idx)
        d_chains = inf_io.load_chains(chain_file)

        # Load the data
        odict = gordon.prep_data(idx)
        wave = odict['wave']
        Rrs = odict['Rrs']
        varRrs = odict['varRrs']
        a_true = odict['a']
        bb_true = odict['bb']
        aw = odict['aw']
        adg = odict['adg']
        aph = odict['aph']
        bbw = odict['bbw']
        bbnw = bb_true - bbw
        wave_true = odict['true_wave']
        Rrs_true = odict['true_Rrs']

        gordon_Rrs = fgordon.calc_Rrs(odict['a'][::2], odict['bb'][::2])

        # Reconstruc
        pdict = fgordon.init_mcmc(model, d_chains['chains'].shape[-1], 
                                wave, Y=odict['Y'], Chl=odict['Chl'])
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = gordon.reconstruct(
            model, d_chains['chains'], pdict) 


        # #########################################################
        # a without water

        ax_anw = axes[1]
        if ss == 0:
            ax_anw.plot(wave_true, a_true-aw, 'ko', label='True', zorder=1)
            ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

        ax_anw.plot(wave, a_mean-aw[::2], clr, label='Retreival')
        #ax_anw.fill_between(wave, a_5-aw_interp, a_95-aw_interp, 
        #        color='r', alpha=0.5, label='Uncertainty') 
        

        #ax_anw.plot(wave_true, adg, '-', color='brown', label=r'$a_{\rm dg}$')
        #ax_anw.plot(wave_true, aph, 'b-', label=r'$a_{\rm ph}$')

        #else:
        #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

        #ax_anw.legend(fontsize=10.)
        #if set_abblim:
        #    ax_anw.set_ylim(bottom=0., top=2*(a_true-aw).max())
        #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels


        # #########################################################
        # b
        use_bbw = bbw[::2]
        ax_bb = axes[2]
        if ss == 0:
            ax_bb.plot(wave_true, bbnw, 'ko', label='True')
            ax_bb.set_ylabel(r'$b_{b,nw} (\lambda) \; [{\rm m}^{-1}]$')
        ax_bb.plot(wave, bb_mean-use_bbw, '-', color=clr, label='Retrieval')
        #ax_bb.fill_between(wave, bb_5-use_bbw, bb_95-use_bbw,
        #        color='g', alpha=0.5, label='Uncertainty') 


        #ax_bb.legend(fontsize=lgsz)
        #if set_abblim:
        #    ax_bb.set_ylim(bottom=0., top=2*bb_true.max())


        # #########################################################
        # Rs
        ax_R = axes[0]
        if ss == 0:
            ax_R.plot(wave, gordon_Rrs, 'k+', label='Observed')
            ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')
            lgsz = 12.
            if log_Rrs:
                ax_R.set_yscale('log')
            else:
                ax_R.set_ylim(bottom=0., top=1.1*Rrs_true.max())
        ax_R.plot(wave, model_Rrs, '-', color=clr, label=f'n={nparm}', zorder=10)
        #ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
        #        color='r', alpha=0.5, zorder=10) 
        ax_R.legend(fontsize=lgsz, loc='lower left')

        
    # axes
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, 14)
        if lbl_wavelengths:
            ax.set_xlabel('Wavelength (nm)')
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels



def fig_corner(model, outroot:str='fig_gordon_corner', idx:int=170,
        scl_noise:float=None,
        add_noise:bool=False): 

    chain_file, noises, noise_lbl = get_chain_file(model, scl_noise, add_noise, idx)
    d_chains = inf_io.load_chains(chain_file)

    # Outfile
    outfile = outroot + f'_{model}_{idx}_{noise_lbl}{noises}.png'

    burn = 7000
    thin = 1
    chains = d_chains['chains']
    coeff = 10**(chains[burn::thin, :, :].reshape(-1, chains.shape[-1]))

    if model == 'hybpow':
        clbls = ['H0', 'g', 'H1', 'H2', 'B1', 'b']
    elif model == 'exppow':
        clbls = ['Adg', 'g', 'Bnw', 'bnw']
    elif model == 'hybnmf':
        clbls = ['H0', 'g', 'H1', 'H2', 'B1', 'B2']
    elif model == 'giop+':
        clbls = ['Adg', 'Sdg', 'Aph', 'Bnw', 'beta']
    else:
        clbls = None

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        #truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_chi2_model(model:str, idx:int=170, chain_file=None, 
                   low_wv=500., 
                   outroot='fig_chi2_model', show_bbnw:bool=False,
                 set_abblim:bool=True, scl_noise:float=None,
                 add_noise:bool=False): 

    # Outfile
    outfile = outroot + f'_{model}_{idx}.png'

    chain_file, noises, noise_lbl = get_chain_file(model, scl_noise, add_noise, idx)
    d_chains = inf_io.load_chains(chain_file)

    # Load the data
    odict = gordon.prep_data(idx)
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a_true = odict['a']
    bb_true = odict['bb']
    aw = odict['aw']
    bbw = odict['bbw']
    bbnw = bb_true - bbw
    wave_true = odict['true_wave']
    Rrs_true = odict['true_Rrs']

    gordon_Rrs = fgordon.calc_Rrs(odict['a'][::2], odict['bb'][::2])

    # Interpolate
    aw_interp = np.interp(wave, wave_true, aw)
    bbw_interp = np.interp(wave, wave_true, bbw)

    # Reconstruc
    pdict = fgordon.init_mcmc(model, d_chains['chains'].shape[-1], 
                              wave, Y=odict['Y'], Chl=odict['Chl'])
    a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
        model_Rrs, sigRs = gordon.reconstruct(
        model, d_chains['chains'], pdict) 

    # Low wave
    ilow = np.argmin(np.abs(wave - low_wv))

    # Calcualte chi^2
    nparm = fgordon.grab_priors(model).shape[0]
    red_chi2s = []
    red_chi2s_low = []
    sigs = [1, 2., 3, 5, 7, 10, 15, 20, 30]
    for scl_sig in sigs:
        chi2 = ((model_Rrs - gordon_Rrs) / ((scl_sig/100.) * gordon_Rrs))**2
        reduced_chi2 = np.sum(chi2) / (len(gordon_Rrs) - nparm)
        red_chi2s.append(reduced_chi2)
        # Low
        reduced_chi2_low = np.sum(chi2[:ilow]) / (ilow - nparm)
        red_chi2s_low.append(reduced_chi2_low)
        

    fig = plt.figure(figsize=(8,8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    ax.plot(sigs, red_chi2s, 'ko-', label='Full')

    ax.plot(sigs, red_chi2s_low, 'bo-', label=r'$\lambda < '+f'{int(low_wv)}'+r'$ nm')

    ax.set_xlabel(r'$100 \, \sigma_{R_{rs}} / R_{rs}$')
    ax.set_ylabel(r'$\chi^2_{\nu}$')

    # Horizontal line at 1.
    ax.axhline(1, color='r', linestyle='--')

    # Add model as text
    ax.text(0.1, 0.1, model+f': idx={idx}', fontsize=15, transform=ax.transAxes,
            ha='left')

    # Log scale y-axis
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Vertical line at 5%
    ax.axvline(5, color='k', linestyle=':')

    # Grid me
    ax.grid(True)
    ax.legend(fontsize=14)

    plotting.set_fontsize(ax, 15)

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_spectra(idx:int, 
                 outroot='fig_spectra_', show_bbnw:bool=False,
                 add_noise:bool=False, log_Rrs:bool=False,
                 show_trueRrs:bool=False,
                 show_acomps:bool=False,
                 bbscl:float=20,
                 set_abblim:bool=True, scl_noise:float=None): 

    # Outfile
    outfile = outroot + f'{idx}.png'

    # 
    odict = anly_utils.prep_l23_data(idx)
    wave = odict['true_wave']
    a = odict['a']
    aw = odict['aw']
    anw = odict['anw']
    aph = odict['aph']
    adg = odict['adg']
    bb = odict['bb']
    bbw = odict['bbw']
    bnw = odict['bb'] - bbw

    #
    fig = plt.figure(figsize=(9,5))
    ax = plt.gca()
    # a
    ax.plot(wave, a, 'k-', label=r'$a$', zorder=1)

    ax.plot(wave, aw, 'b-', label=r'$a_w$', zorder=1)
    ax.plot(wave, anw, 'r-', label=r'$a_{nw}$', zorder=1)
    if show_acomps:
        ax.plot(wave, aph, 'g-', label=r'$a_{ph}$', zorder=1)
        ax.plot(wave, adg, '-', color='brown', label=r'$a_{dg}$', zorder=1)

    # bb
    ax.plot(wave, bb, ':', color='k', label=r'$b_{b}$', zorder=1)

    ax.plot(wave, bbscl*bbw, ':', color='blue', label=f'{bbscl}*'+r'$b_{b,w}$', zorder=1)
    ax.plot(wave, bbscl*bnw, ':', color='red', label=f'{bbscl}*'+r'$b_{b,nw}$', zorder=1)

    #
    # Legend filled white
    ax.legend(fontsize=13., loc='upper right', 
              frameon=True, facecolor='white')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a, b_b \; [{\rm m}^{-1}]$')
    ymax = 0.08
    ax.set_xlim(350., 750.)
    ax.set_ylim(0., ymax)

    plotting.set_fontsize(ax, 15)

    # Fill between
    alpha=0.3
    ax.fill_between([500., 750.], 0, ymax, color='red', alpha=alpha)
    ax.fill_between([350., 450.], 0, ymax, color='blue', alpha=alpha)

    # Add text
    yl1, yl2 = 0.075, 0.07
    x1, x2 = 505., 445.
    # Red
    ax.text(x1, yl1, r'$a_w \gg a_{nw}$', fontsize=15, ha='left')
    ax.text(x1, yl2, r'$b_{b,nw} \approx b_{b,w}$', fontsize=15, ha='left')

    ax.text(x2, yl1, r'$b_{b,w} \gg b_{b,nw}$', fontsize=15, ha='right')
    ax.text(x2, yl2, r'$a_{nw} > a_{w}$', fontsize=15, ha='right')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_all_bic(use_LM:bool=True, wstep:int=1,
                outfile:str='fig_all_bic.png'):

    Bdict = {}
    
    # Loop on the models
    for k in [3,4]:#,5]:
        Bdict[k] = []

        # Model names
        if k == 3:
            model_names = ['Exp', 'Cst']
        elif k == 4:
            model_names = ['Exp', 'Pow']
        else:
            raise ValueError("Bad k")

        chain_file, noises, noise_lbl = get_chain_file(
            model_names, 0.02, False, 'L23', use_LM=use_LM)
        d_chains = np.load(chain_file)

        # Load data for wave
        odict = anly_utils.prep_l23_data(170, step=wstep)
        wave = odict['wave']

        # Init the models
        anw_model = big_anw.init_model(model_names[0], wave, 'log')
        bbnw_model = big_bbnw.init_model(model_names[1], wave, 'log')
        models = [anw_model, bbnw_model]

        # Loop on S/N
        if k == 3:
            sv_s2n = []
        for s2n in [0.02, 0.03, 0.05, 0.07, 0.10]:
            # Calculate BIC
            BICs = big_stats.calc_BICs(d_chains['obs_Rrs'], models, d_chains['ans'],
                                s2n, use_LM=use_LM)
            Bdict[k].append(BICs)
            # 
            if k == 3:
                sv_s2n += [s2n]*BICs.size
        Bdict[k] = np.array(Bdict[k]).T
        # Concatenate
        Bdict[k] = np.concatenate(Bdict[k])
                            
    # Generate a pandas table
    D_BIC_34 = Bdict[3] - Bdict[4]
    df = pandas.DataFrame()
    df['D_BIC'] = D_BIC_34
    df['s2n'] = sv_s2n
    

    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    sns.kdeplot(data=df, x="D_BIC", hue="s2n", ax=plt.subplot(gs[0]))

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")





# ############################################################
def fig_one_bic(models:list=None, idx:int=170, 
            scl_noises:list=None,
            low_wv=500., 
            outroot='fig_bic_', show_bbnw:bool=False,
            set_abblim:bool=True, 
            add_noise:bool=False): 


    # Outfile
    outfile = outroot + f'{idx}.png'

    if scl_noises is None:
        scl_noises = [0.02, 0.03, 0.05, 0.07, 0.10]

    if models is None:
        models = ['expcst', 'exppow', 'giop+']

    # Load the data
    odict = gordon.prep_data(idx)
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a_true = odict['a']
    bb_true = odict['bb']
    aw = odict['aw']
    bbw = odict['bbw']
    bbnw = bb_true - bbw
    wave_true = odict['true_wave']
    Rrs_true = odict['true_Rrs']

    gordon_Rrs = fgordon.calc_Rrs(odict['a'][::2], odict['bb'][::2])

    # Calculate BIC
    BICs = {}
    nparms = []
    for model in models:
        nparm = fgordon.grab_priors(model).shape[0]
        nparms.append(nparm)
        if model not in BICs.keys():
            BICs[model] = []
        # Load noiseless (should not matter) 
        chain_file, noises, noise_lbl = get_chain_file(
            model, 0.02, False, idx)
        d_chains = inf_io.load_chains(chain_file)

        # Reconstruct
        pdict = fgordon.init_mcmc(model, d_chains['chains'].shape[-1], 
                                wave, Y=odict['Y'], Chl=odict['Chl'])
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = gordon.reconstruct(
            model, d_chains['chains'], pdict) 

        for scl_noise in scl_noises:
            # Calcualte chi^2
            chi2 = ((model_Rrs - gordon_Rrs) / ((scl_noise) * gordon_Rrs))**2
            Bic = nparm * np.log(len(model_Rrs)) + np.sum(chi2) 
            # Save
            BICs[model].append(Bic)
        
    # Plot em

    fig = plt.figure(figsize=(8,8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    for kk, scl_noise in enumerate(scl_noises):
        these_BICs = []
        for model in models:
            these_BICs.append(BICs[model][kk])
        ax.plot(nparms, these_BICs, '-', label=f'{int(100*scl_noise):02d}')

    ax.set_xlabel('N parameters')
    ax.set_ylabel('BIC')

    # Add model as text
    ax.text(0.1, 0.1, f'idx={idx}', fontsize=15, transform=ax.transAxes,
            ha='left')

    # Log scale y-axis
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.set_ylim(0., 100.)

    # Grid me
    ax.grid(True)
    ax.legend(fontsize=14)

    plotting.set_fontsize(ax, 15)

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Indiv
    if flg == 1:
        fig_u()

    # Spectra
    if flg == 2:
        fig_spectra(170, bbscl=20)

    # BIC
    if flg == 4:
        fig_all_bic()

    # LM fits
    if flg == 10:
        #fig_mcmc_fit(['Exp', 'Pow'], idx=170, log_Rrs=True)
        fig_mcmc_fit(['Exp', 'Pow'], idx=170, log_Rrs=True, use_LM=True)
        fig_mcmc_fit(['Exp', 'Cst'], idx=170, log_Rrs=True, use_LM=True)
        fig_mcmc_fit(['Cst', 'Cst'], idx=170, log_Rrs=True, use_LM=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)