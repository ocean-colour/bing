""" Figs for Gordon Analyses """
import os, sys
from importlib.resources import files

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d
import pandas


from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import corner

from ocpy.water import absorption
from ocpy.utils import plotting 
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import modis as sat_modis
from ocpy.water import absorption

from bing import plotting as bing_plot
from bing.models import utils as model_utils
from bing.models import functions
from bing import evaluate

#from bing.models import anw as bing_anw
#from bing.models import bbnw as bing_bbnw
#from bing import chisq_fit
#from bing import stats as bing_stats

# Local
sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20

sys.path.append(os.path.abspath("../Analysis/py"))
import anly_utils

from IPython import embed

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)

def fig_u(outfile='fig_u.png', log_log:bool=False):
    """
    Generate a figure showing the relationship between u (backscattering ratio) and rrs (remote sensing reflectance).

    Parameters:
        outfile (str): The filename of the output figure (default: 'fig_u.png')

    """
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
    uval = np.linspace(0., 0.40, 1000)
    rrs_GIOP = rrs_func(uval, G1, G2)
    Rrs_GIOP = A*rrs_GIOP / (1 - B*rrs_GIOP)

    # Fit
    save_ans = []
    for ii in [i370, i440, i500, i600]:
        ans, cov = curve_fit(rrs_func, u[:,ii], rrs[:,ii], p0=[0.1, 0.1], sigma=np.ones_like(u[:,ii])*0.0003)
        save_ans.append(ans)
        #embed(header='figs 167')

    #
    fig = plt.figure(figsize=(8,5))

    plt.clf()
    ax = plt.gca()
    for lbl, clr, idx, ans in zip(['370nm', '440nm', '500nm', '600nm'],
                                ['purple', 'b','g', 'r'],
                                [i370, i440, i500, i600],
                                save_ans):
        ax.scatter(u[:,idx], rrs[:,idx], color=clr, s=1., label=r'$\lambda = $'+lbl)
        irrs = rrs_func(u[:,idx], ans[0], ans[1])
        usrt = np.argsort(u[:,idx])
        ax.plot(u[usrt,idx], irrs[usrt], '-', color=clr, 
                label=r'Fit: $G_1='+f'{ans[0]:0.2f},'+r'G_2='+f'{ans[1]:0.2f}'+r'$')
        # Stats
        if idx == i370:
            uv = 0.35
            ss = np.argmin(np.abs(uv - u[:,idx]))
            rrsv = rrs_func(uv, G1, G2)
            print(f"Perecent error: {100.*(rrsv-rrs[ss,idx])/rrs[ss,idx]:0.2f}%")
            #embed(header='figs 167')
        # RMS of fit
        rms = np.sqrt(np.mean((rrs[usrt,idx] - irrs)**2/(irrs**2)))
        print(f"wv={lbl}, rRMS={10*rms:0.4f}")

    # GIOP
    ax.plot(uval, rrs_GIOP, 'k--', label=f'Gordon: '+r'$G_1='+f'{G1}, '+r'$G_2=$'+f'{G2}'+r'$')
    ax.grid()
    #
    ax.set_xlabel(r'$u(\lambda)$')
    ax.set_ylabel(r'$r_{\rm rs} (\lambda)$')
    ax.legend(fontsize=10)
    plotting.set_fontsize(ax, 15.)

    if log_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        #
        ax.set_xlim(2e-3,None)
        ax.set_ylim(1e-4,None)
    
    #
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_Kd(outfile='fig_Kd.png'):
    """
    Generate a figure showing the relationship between Kd
    and IOPs

    Parameters:
        outfile (str): The filename of the output figure (default: 'fig_u.png')

    """
    def lee2002_func(a, bb, thetas=0.):
        Kd_lee = (1+0.005*thetas)*a + 4.18 * (1-0.52*np.exp(-10.8*a))*bb
        return Kd_lee

    # Load
    ds = loisel23.load_ds(4,0)
    ds_profile = loisel23.load_ds(4,0, profile=True)

    # Unpack
    wave = ds.Lambda.data
    Rrs = ds.Rrs.data
    a = ds.a.data
    bb = ds.bb.data
    aph = ds.aph.data

    Kd = ds_profile.KEd_z[1,:,:]
    #xscat = a[:,idx] + 4.18 * (1-0.52*np.exp(-10.8*a[:,idx]))*bb[:,idx]
    xscat = a + 4.18 * (1-0.52*np.exp(-10.8*a))*bb
    sclr = np.outer(np.ones(Rrs.shape[0]), wave)


    # Select wavelengths
    i370 = np.argmin(np.abs(wave-370.))
    i440 = np.argmin(np.abs(wave-440.))
    i500 = np.argmin(np.abs(wave-500.))
    i600 = np.argmin(np.abs(wave-600.))

    Chl = aph[:,i440] / 0.05582

    # Calculate Kd

    #
    fig = plt.figure(figsize=(7,5))

    plt.clf()
    ax = plt.gca()

    sc = ax.scatter(xscat, Kd, c=sclr, s=1., cmap='jet')
    gen_cb(sc, 'Wavelength (nm)')

    #
    ax.set_xlabel(r'Lee+2002 $K_d(a,b_b)$ ordinate')
    ax.set_ylabel(r'$K_d$')
    #ax.legend(fontsize=12)

    # Add a 1-1 line using the axis limits
    axlim = ax.get_xlim()
    ax.plot(axlim, axlim, 'k--')


    plotting.set_fontsize(ax, 15.)
    #
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_mcmc_fit(model_names:list, idx:int=170, chain_file=None,
                 outroot='fig_fit_', 
                 add_noise:bool=False, 
                 full_LM:bool=True,
                 MODIS:bool=False, 
                 PACE:bool=False, 
                 SeaWiFS:bool=False,
                 max_wave:float=None,
                 use_LM:bool=False,
                 scl_noise:float=0.02): 

    # Load the fits
    chain_file = anly_utils.chain_filename(
        model_names, scl_noise, add_noise, idx=idx, 
        MODIS=MODIS, PACE=PACE, SeaWiFS=SeaWiFS)
    print(f'Loading: {chain_file}')
    d = np.load(chain_file)

    # Data
    odict = anly_utils.prep_l23_data(idx, scl_noise=scl_noise,
                                     max_wave=max_wave)
    #embed(header='figs 167')

    # Prep 
    model_wave = d['wave']
    models = model_utils.init(model_names, model_wave)

    # Outfile
    bcfile = os.path.basename(chain_file)
    outfile = outroot + bcfile.replace('npz', 'png')
    if use_LM:
        outfile = outfile.replace('BING', 'BING_LM')

    # Inputs
    params = d['ans'] if use_LM else d['chains']
    # Set up the basis functions, etc.
    a_params = d['Chl']
    bb_params = d['Y']
    if models[0].uses_Chl:
        models[0].set_aph(float(a_params))
    if models[1].uses_basis_params:  # Lee
        models[1].set_basis_func(float(bb_params))
    if full_LM:
        params = params[idx]
        a_params = a_params[idx]
        bb_params = bb_params[idx]

    #embed(header='237 of figs')
    axes = bing_plot.show_fit(
        models, params,
        ex_a_params=a_params, ex_bb_params=bb_params,
        Rrs_true=dict(wave=d['wave'], spec=d['obs_Rrs'],
                      var=d['varRrs']),
        anw_true=dict(wave=odict['true_wave'], spec=odict['anw']),
        bbnw_true=dict(wave=odict['true_wave'], spec=odict['bbnw']),
        fontsize=15.,
        )
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_degenerate_fits(model_names:list, idx:int=170, chain_file=None,
                 outroot='fig_deg_', 
                 add_noise:bool=False, 
                 full_LM:bool=True,
                 MODIS:bool=False, 
                 PACE:bool=False, 
                 SeaWiFS:bool=False,
                 max_wave:float=None,
                 use_LM:bool=False,
                 scl_noise:float=0.02): 

    # Load the fits
    #chain_file = '../Analysis/Fits/BING_LM_ExpBricaudPow_170_nP.npz'
    chain_file = '../Analysis/Fits/BING20_EveryEvery_170_P_n02_UV400_SdgU.npz'
    #chain_file = anly_utils.chain_filename(
    #    model_names, scl_noise, add_noise, idx=idx, 
    #    MODIS=MODIS, PACE=PACE, SeaWiFS=SeaWiFS)
    #print(f'Loading: {chain_file}')
    d = np.load(chain_file)
    wave = d['wave']
    Rrs_true=dict(wave=d['wave'], spec=d['obs_Rrs'],
                      var=d['varRrs'])

    # Data
    odict = anly_utils.prep_l23_data(idx, scl_noise=scl_noise,
                                     max_wave=max_wave)

    gd_wave = (odict['true_wave'] >= 400.) & (odict['true_wave'] <= 700.) 
    anw_true=dict(wave=odict['true_wave'][gd_wave], spec=odict['anw'][gd_wave])
    bbnw_true=dict(wave=odict['true_wave'][gd_wave], spec=odict['bbnw'][gd_wave])

    # Outfile
    outfile = outroot + f'{idx}.png'

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    # TODO -- FIX THIS!
    # THIS IS A HACK UNTIL I CAN RESOLVE bbw
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    idx = 170 # Random choie
    l23_bb = ds.bb.data[idx] 
    l23_bbnw = ds.bbnw.data[idx] 
    l23_bbw = l23_bb - l23_bbnw
    # Interpolate
    bb_w = np.interp(wave, l23_wave, l23_bbw)
    #embed(header='figs 324')
    bb_true = bb_w + bbnw_true['spec']#[gd_wave]

    a_true = anw_true['spec'] + a_w

    # #########################################################
    # Plot the solution
    lgsz = 14.
    figsize:tuple=(14,6)

    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,3)
    

    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[1])
    ax_anw.plot(anw_true['wave'], anw_true['spec'], 'ko', label='True', zorder=1)

    sv_anws = []
    sv_bbnws = []
    #scales = [0.01, 0.1, 0.3, 1., 3., 10., 100]
    lw = 3
    scales = [0.9, 1., 3., 10., 100]
    for ss, scale in enumerate(scales):
        scaled_anw = anw_true['spec'] * scale
        #lbl = 'Retrieval' if ss == 0 else None
        lbl = f'{scale:0.1f}'
        ax_anw.plot(anw_true['wave'], scaled_anw, ':', label=lbl, lw=lw)
        # Calculate bbnw
        scaled_a = a_true - anw_true['spec'] + scaled_anw
        sv_anws.append(scaled_anw)
        bbnw = scaled_a * (bb_true/a_true) - bb_w
        sv_bbnws.append(bbnw)
        
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    ax_anw.set_ylim(2e-4,3.)


    # #########################################################
    # bb nw
    ax_bb = plt.subplot(gs[2])
    ax_bb.plot(bbnw_true['wave'], bbnw_true['spec'], 'ko', label='True', zorder=1)
    for ss, scale in enumerate(scales):
        lbl = 'Retrieval' if ss == 0 else None
        ax_bb.plot(wave, sv_bbnws[ss], ':', label=lbl, lw=lw)
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')
    #ax_bb.set_ylim(0., 0.001)

    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0])
    ax_R.plot(Rrs_true['wave'], Rrs_true['spec'], 'k+', label='True', zorder=1)
    ax_R.plot(Rrs_true['wave'], Rrs_true['spec'], 'b-', label='Retrieval')
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')

    # Log scale y-axis
    #ax_R.set_yscale('log')
    
    # axes
    fontsize:float=12.
    axes = [ax_anw, ax_R, ax_bb]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)
        if ss < 3:
            ax.set_yscale('log')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_multi_fits(models:list=None, 
                   indices:list=None, 
                   min_wave:float=400.,
                   max_wave:float=700.,
                 outroot='fig_multi_fits'): 

    if models is None:
        models = [('Cst','Cst'), ('Exp','Cst'), ('Exp','Pow'), ('ExpBricaud','Pow')]
    if indices is None:
        indices = [170, 1032]
    outfile = outroot + f'_{indices[0]}_{indices[1]}.png'

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,3)

    compare_models(models, indices[0], 
                   [plt.subplot(gs[0]), plt.subplot(gs[1]), 
                    plt.subplot(gs[2])],
                   lbl_wavelengths=False,
                   min_wave=min_wave, max_wave=max_wave)
    compare_models(models, indices[1], 
                   [plt.subplot(gs[3]), plt.subplot(gs[4]), 
                    plt.subplot(gs[5])],
                    min_wave=min_wave, max_wave=max_wave)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def compare_models(models:list, idx:int, axes:list, 
                   min_wave:float=None, max_wave:float=None,
                   add_noise:bool=False, scl_noise:float=None,
                   log_Rrs:bool=True, lbl_wavelengths:bool=True,
                   use_LM:bool=True, full_LM:bool=True):

    # Loop on models
    for ss, clr, model_names in zip(
        range(len(models)), ['r', 'g', 'b', 'orange'], models):


        rdict = anly_utils.recon_one(
            model_names, idx, 
            scl_noise=scl_noise, add_noise=add_noise, use_LM=use_LM,
            full_LM=full_LM, min_wave=min_wave, max_wave=max_wave)
        # Unpack what we need
        #noise_lbl = rdict['noise_lbl']
        #noises = rdict['noises']
        wave_true = rdict['wave_true']
        Rrs_true = rdict['Rrs_true']
        a_true = rdict['a_true']
        a_mean = rdict['a_mean']
        bb_true = rdict['bb_true']
        aw = rdict['aw']
        adg = rdict['adg']
        aph = rdict['aph']
        aw_interp = rdict['aw_interp']
        wave = rdict['wave']
        bbw = rdict['bbw']
        bbnw = rdict['bbnw']
        bb_mean = rdict['bb_mean']
        gordon_Rrs = rdict['gordon_Rrs']
        model_Rrs = rdict['model_Rrs']
        models = [rdict['anw_model'], rdict['bbnw_model']]

        nparm = models[0].nparam + models[1].nparam

        # #########################################################
        # a without water

        ax_anw = axes[1]
        if ss == 0:
            ax_anw.plot(wave_true, a_true-aw, 'ko', label='True', zorder=1)
            ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

        ax_anw.plot(wave, a_mean-aw, clr, label='Retreival')


        # #########################################################
        # b
        use_bbw = bbw
        ax_bb = axes[2]
        if ss == 0:
            ax_bb.plot(wave_true, bbnw, 'ko', label='True')
            ax_bb.set_ylabel(r'$b_{b,nw} (\lambda) \; [{\rm m}^{-1}]$')
        ax_bb.plot(wave, bb_mean-use_bbw, '-', color=clr, label='Retrieval')

        # #########################################################
        # Rs
        ax_R = axes[0]
        if ss == 0:
            ax_R.plot(wave, gordon_Rrs, 'k+', label='True')
            ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [{\rm sr}^{-1}$]')
            lgsz = 11.
            if log_Rrs:
                ax_R.set_yscale('log')
            else:
                ax_R.set_ylim(bottom=0., top=1.1*Rrs_true.max())
        ax_R.plot(wave, model_Rrs, '-', color=clr, label=f'[k={nparm}]', zorder=10)
        ax_R.legend(fontsize=lgsz, loc='lower left')

        rel_err = np.abs(model_Rrs - gordon_Rrs) / gordon_Rrs
        print(f"Model {model_names}: {rel_err.mean():0.3f} {rel_err.max():0.3f}")
        
    # axes
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, 14)
        if lbl_wavelengths:
            ax.set_xlabel('Wavelength (nm)')
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels


def fig_corner(model_names:list, outroot:str='fig_corner_', idx:int=170,
                 full_LM:bool=True, scl_noise:float=None,
                 MODIS:bool=False, PACE:bool=False,
                 SeaWiFS:bool=False, show_log:bool=False,
                 use_LM:bool=False, add_noise:bool=False): 

    # Load the fits
    chain_file = anly_utils.chain_filename(
        model_names, scl_noise, add_noise, idx=idx, 
        MODIS=MODIS, PACE=PACE, SeaWiFS=SeaWiFS)
    print(f'Loading: {chain_file}')
    d_chains = np.load(chain_file)

    # Init the models
    models = model_utils.init(model_names, d_chains['wave'])

    # Right answer
    ds = loisel23.load_ds(4,0)
    i440 = np.argmin(np.abs(ds.Lambda.data-440.))
    i443 = np.argmin(np.abs(ds.Lambda.data-443.))
    i600 = np.argmin(np.abs(ds.Lambda.data-600.))

    aph_440 = ds.aph.data[idx,i440]
    true_Chl = aph_440 / 0.05582
    aph_443 = ds.aph.data[idx,i443]
    adg_440 = ds.ag.data[idx,i440] + ds.ad.data[idx,i440]
    adg_443 = ds.ag.data[idx,i443] + ds.ad.data[idx,i443]
    bbnw_443 = ds.bbnw.data[idx,i443]
    bbnw_600 = ds.bbnw.data[idx,i600]

    # Outfile
    bcfile = os.path.basename(chain_file)
    outfile = outroot + bcfile.replace('npz', 'png')

    burn = 7000
    thin = 1
    chains = d_chains['chains']
    coeff = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    if not show_log:
        coeff = 10**coeff
    

    if model_names[0] == 'GIOP':
        truths = [adg_440, aph_440]
    elif model_names[0] == 'GSM':
        truths = [adg_443, true_Chl]

    if model_names[1] == 'Lee':
        truths += [bbnw_600]
    elif model_names[1] == 'GSM':
        truths += [bbnw_443]

    # Labels
    clbls = models[0].pnames + models[1].pnames
    # Add log 10
    clbls = [r'$\log_{10}('+f'{clbl}'+r'$)' for clbl in clbls]
    #embed(header='figs 407')

    if show_log and truths is not None:
        truths = np.log10(truths)

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        truths=truths,
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
                 outroot='fig_spectra_', 
                 show_acomps:bool=False,
                 xmax:float=700.,
                 use_ylog:bool=True,
                 show_total:bool=False,
                 bbscl:float=1):

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
    bbnw = odict['bb'] - bbw

    #
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()

    # Colors
    ctotal ='gray'
    cwater ='black'
    cnw ='orange'
    #embed(header='559 of figs')

    # a
    # Total
    if show_total:
        ax.plot(wave, a, '-', color=ctotal, label=r'$a$', zorder=1)

    ax.plot(wave, aw, '-', color=cwater, label=r'$a_w$', zorder=1)
    ax.plot(wave, anw, '-', color=cnw, label=r'$a_{nw}$', zorder=1)
    if show_acomps:
        ax.plot(wave, aph, 'b-', label=r'$a_{ph}$', zorder=1)
        ax.plot(wave, adg, '-', color='brown', label=r'$a_{dg}$', zorder=1)

    # bb
    bb_ls = '--'
    if show_total:
        ax.plot(wave, bb, bb_ls, color=ctotal, label=r'$b_{b}$', zorder=1)

    if bbscl != 1.:
        ax.plot(wave, bbscl*bbw, bb_ls, color=cwater, label=f'{bbscl}*'+r'$b_{b,w}$', zorder=1)
        ax.plot(wave, bbscl*bbnw, bb_ls, color=cnw, label=f'{bbscl}*'+r'$b_{b,nw}$', zorder=1)
    else:
        ax.plot(wave, bbscl*bbw, bb_ls, color=cwater, label=r'$b_{b,w}$', zorder=1)
        ax.plot(wave, bbscl*bbnw, bb_ls, color=cnw, label=r'$b_{b,nw}$', zorder=1)

    #
    # Legend filled white
    ax.legend(fontsize=13., loc='upper right', 
              frameon=True, facecolor='white')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a, b_b \; [{\rm m}^{-1}]$')
    ax.set_xlim(350., xmax)
    if use_ylog:
        ax.set_yscale('log')
        ymax = 1.0
        ax.set_ylim(1.e-4, ymax)
    else:
        ymax = 0.08
        ax.set_ylim(0., ymax)

    plotting.set_fontsize(ax, 17)

    # Fill between
    aw_to_anw = aw/anw
    bbw_to_bbnw = bbw/bbnw
    ratio = 5.
    red_idx = np.argmin(np.abs(aw_to_anw - ratio))
    blue_idx = np.argmin(np.abs(bbw_to_bbnw - ratio))

    alpha=0.3
    ax.fill_between([wave[red_idx], xmax], 0, ymax, color='red', alpha=alpha)
    ax.fill_between([350., wave[blue_idx]], 0, ymax, color='blue', alpha=alpha)

    # Text
    buff = 5.
    if not use_ylog:
        ax.text(wave[red_idx]+buff, 0.025, r'$a_w > '+f'{int(ratio)}'+r'a_{nw}$', fontsize=15, ha='left')
        ax.text(wave[red_idx]+buff, 0.02, r'$b_{b,nw} \approx b_{b,w}$', fontsize=15, ha='left')

        ax.text(wave[blue_idx]-buff, 0.07, r'$b_{b,w} >'+f'{int(ratio)}'+r'b_{b,nw}$', fontsize=15, ha='right')
        ax.text(wave[blue_idx]-buff, 0.075, r'$a_{nw} > a_{w}$', fontsize=15, ha='right')
    else:
        yscl = 1.5
        tfsz = 14
        #ax.text(wave[red_idx]+buff, 0.025, r'$a_w > '+f'{int(ratio)}'+r'a_{nw}$', fontsize=tfsz, ha='left')
        #ax.text(wave[red_idx]+buff, 0.025*yscl, r'$b_{b,nw} \approx b_{b,w}$', fontsize=tfsz, ha='left')

        #ax.text(wave[blue_idx]-buff, 0.07, r'$b_{b,w} >'+f'{int(ratio)}'+r'b_{b,nw}$', fontsize=tfsz, ha='right')
        #ax.text(wave[blue_idx]-buff, 0.07*yscl, r'$a_{nw} > a_{w}$', fontsize=tfsz, ha='right')

        yscl = 0.90
        tfsz = 17
        ax.text(wave[blue_idx]-buff, ymax*yscl, 
                'Water dominates\n back-scattering\n (retrieve '+r'$a_{\rm nw}$)', fontsize=tfsz, ha='right',
                va='top')
        ax.text(wave[red_idx]+buff, ymax*yscl, 
                'Water dominates absorption\n' +r'(retrieve $b_{b,nw}$)', 
                fontsize=tfsz, ha='left',
                va='top')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_satellite_noise(satellite:str, wave:int, min_Rrs:float=-0.03):

    # Load up the data
    if satellite == 'MODIS_Aqua':
        # Load
        #sat_file = files('boring').joinpath(os.path.join('data', 'MODIS', 'MODIS_matchups_rrs.csv'))
        sat_key = 'aqua_rrs'
        insitu_key = 'insitu_rrs'
        matchups = sat_modis.load_matchups()
    elif satellite == 'SeaWiFS':
        sat_key = 'seawifs_rrs'
        insitu_key = 'insitu_rrs'
        matchups = sat_seawifs.load_matchups()
    else:
        raise ValueError("Not ready for this satellite yet")

    outfile = f'fig_noise_{satellite}_{wave}.png'
    cut = np.isfinite(matchups[f'{sat_key}{wave}']) & (matchups[f'{sat_key}{wave}'] > min_Rrs) & (
        matchups[f'{insitu_key}{wave}'] > min_Rrs)

    matchups = matchups[cut].copy()


    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Compare in-situ with 
    ax_c = plt.subplot(gs[0])

    #embed(header='figs 167')
    ax_c.plot(matchups[f'{insitu_key}{wave}'], matchups[f'{sat_key}{wave}'], 
              'ko', markersize=0.5)
    # 1-1 line
    mxval = np.concatenate([matchups[f'{insitu_key}{wave}'], matchups[f'{sat_key}{wave}']]).max()
    ax_c.plot([0., mxval], [0., mxval], 'r--')

    # Labels
    ax_c.set_xlabel(f'In-situ '+r'$R_{\rm rs}$'+f'({wave} nm)'+r' [sr$^{-1}$]')
    ax_c.set_ylabel(f'{satellite} '+r'$R_{\rm rs}$'+f'({wave} nm)'+r' [sr$^{-1}$]')

    ax_c.text(0.1, 0.9, f'{satellite}', fontsize=17, transform=ax_c.transAxes, ha='left')
    ax_c.grid()

    # ###########################################3
    # Histogram the diff
    ax_h = plt.subplot(gs[1])
    diff = matchups[f'{insitu_key}{wave}'] - matchups[f'{sat_key}{wave}']

    ax_h.hist(diff, bins=100, histtype='step', color='k', linewidth=2)
    _, low, high = sigmaclip(diff, low=4., high=4.)

    # Show clipped regions
    ax_h.axvline(low, color='r', linestyle='--')
    ax_h.axvline(high, color='r', linestyle='--')

    # Stats
    sig_cut = (diff > low) & (diff < high)
    std = np.std(diff[sig_cut])

    # Text me
    ax_h.text(0.95, 0.9, f'RMS={std:0.4f}'+r' [sr$^{-1}$]', fontsize=17, 
              transform=ax_h.transAxes, ha='right')

    # Labels
    ax_h.set_xlabel(r'$\Delta R_{\rm rs}$'+f'({wave}) '+r'[sr$^{-1}$]')
    ax_h.set_ylabel('N')

    # axes
    for ax in [ax_c, ax_h]:
        plotting.set_fontsize(ax, 19)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



def fig_pace_noise(outfile:str='fig_pace_noise.png'):

    # Load up the data
    pace_file = files('ocpy').joinpath(os.path.join(
        'data', 'satellites', 'PACE_error.csv'))
    actual_PACE_error = pandas.read_csv(pace_file)
    acut = (actual_PACE_error['wave'] < 700.) & (actual_PACE_error['wave'] > 400.)

    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    l23_PACE_error = sat_pace.gen_noise_vector(l23_wave)
    lcut = (l23_wave < 700.) & (l23_wave > 400.)

    # Load a random Rrs
    idx = 170 # Random choice
    Rrs = ds.Rrs.data[idx]
    Rrs = Rrs[lcut]

    # S/N
    s2n = Rrs / l23_PACE_error[lcut]

    #embed(header='fig_all_bic 660')

    fig = plt.figure(figsize=(10,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Compare in-situ with 
    ax_c = plt.subplot(gs[0])

    #embed(header='figs 167')
    ax_c.plot(actual_PACE_error['wave'][acut], actual_PACE_error['PACE_sig'][acut], 'b-',
              label='Median PACE Noise')
    ax_c.plot(l23_wave[lcut], l23_PACE_error[lcut], 'ko', label='Re-sampled PACE Noise')


    # Labels
    ax_c.set_xlabel('Wavelength (nm)')
    ax_c.set_ylabel('Noise [sr$^{-1}$]')
    ax_c.set_ylim(None, 1e-3)
    ax_c.grid()

    # Log
    ax_c.set_yscale('log')

    ax_c.legend(fontsize=15)

    # New axis for S/N
    ax_s2n = ax_c.twinx()
    ax_s2n.plot(l23_wave[lcut], s2n, 'r*', label='Example S/N')
    ax_s2n.set_ylabel('S/N', color='red')
    ax_s2n.set_ylim(0., None)
    # Color the axis font
    for label in ax_s2n.get_yticklabels():
        label.set_color('red')
    
    # axes
    for ax in [ax_c, ax_s2n]:
        plotting.set_fontsize(ax, 19)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_all_ic(use_LM:bool=True, show_AIC:bool=False,
                outfile:str='fig_all_bic.png', MODIS:bool=False,
                comp_ks:tuple=((3,4),(4,5)),
                SeaWiFS:bool=False, xmax:float=None,
                PACE:bool=False, log_x:bool=True):

    Bdict = {}

    s2ns = [0.05, 0.10, 0.2]
    ks = [comp_ks[0][0], comp_ks[0][1], comp_ks[1][0], comp_ks[1][1]]

    if MODIS:
        s2ns += ['MODIS_Aqua']
    elif PACE:
        s2ns += ['PACE']
    elif SeaWiFS:
        s2ns += ['SeaWiFS']

    #embed(header='fig_all_ic 571')
    Adict, Bdict = anly_utils.calc_ICs(
        ks, s2ns, use_LM=use_LM, MODIS=MODIS, PACE=PACE,
        SeaWiFS=SeaWiFS)
        
    # Generate a pandas table
    D_BIC_A = Bdict[comp_ks[0][0]] - Bdict[comp_ks[0][1]]
    D_BIC_B = Bdict[comp_ks[1][0]] - Bdict[comp_ks[1][1]]

    # Trim junk in MODIS
    D_BIC_A = np.maximum(D_BIC_A, -5.)
    D_BIC_B = np.maximum(D_BIC_B, -5.)
    #embed(header='690 of fig_all_bic')

    #embed(header='fig_all_bic 660')

    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    nbins = 100
    # 34
    xlbl = 'AIC' if show_AIC else 'BIC'
    save_axes = []
    for ss in range(2):
        ax = plt.subplot(gs[ss])
        D_BIC = D_BIC_A if ss == 0 else D_BIC_B
        subset = f'{comp_ks[ss][0]},{comp_ks[ss][1]}'
        for ss, s2n in enumerate(s2ns):
            if log_x:
                xvals = np.log10(D_BIC[ss] + 6.)
            else:
                xvals = D_BIC[ss]
            # CDF
            srt = np.sort(xvals)
            yvals = np.arange(srt.size) / srt.size
            try:
                fs2n = int(1./float(s2n))
                color = None
                ls = ':'
                lw = 2
            except ValueError:
                fs2n = s2n
                color = 'k'
                ls = '-'
                lw = 3
            ax.plot(srt, yvals, label=f'S/N={fs2n}', color=color, 
                    linewidth=lw, ls=ls)
            # PDF
            #ax34.hist(xvals, bins=nbins,
            #        histtype='step', 
            #        fill=None, label=f's2n={s2n}',
            #        linewidth=3)
            # Stats
            #print(f'{s2n}: {np.sum(D_BIC_34[ss] < 0)/D_BIC_34[ss].size}')
        if log_x:
            ax.set_xlabel(r'$\log_{10}(\Delta \, \rm '+xlbl+'_{'+f'{subset}'+r'} + 6)$')
        else:
            ax.set_xlabel(r'$\Delta \, \rm '+xlbl+'_{'+f'{subset}'+r'}$')

        # Make it pretty
        # Title
        title = r'Include $\beta_{\rm nw}$?' if ss == 0 else 'Include phytoplankton?' 
        #ax.text(0.5, 1.05, title, ha='center', va='top', 
        #        fontsize=15, transform=ax.transAxes)

        ax.set_ylabel('CDF')
        ax.grid(True)
        plotting.set_fontsize(ax, 17)
        #
        if xmax is None:
            xmax = 30. if (MODIS or SeaWiFS) else 50.
        if not log_x:
            ax.set_xlim(-5., xmax)
        else:
            ax.set_xlim(0.25, None)
        ax.legend(fontsize=17)

        # Vertical line at 0
        vline = 5. if show_AIC else 0.
        if log_x:
            vline = np.log10(vline + 6.)
        ax.axvline(vline, color='r', linestyle='--', lw=2)
        # Grab ylimits
        #xl = ax.get_xlim()
        #yl = ax.get_ylim()
        #ax.text(5, 0.6, 'Complex model favored', fontsize=18, ha='left')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



# ################################
def fig_bic_modis_pace(use_LM:bool=True, 
                outfile:str='fig_bic_modis_pace.png', 
                log_x:bool=False):


    r_s2ns = [0.05, 0.10, 0.2]

    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    xlbl = 'BIC'
    for ss in range(2):

        if ss==0:
            s2ns = r_s2ns + ['MODIS/Aqua']
            MODIS = True
            PACE = False
            dataset = '(a) Multi-spectral'
            ks = [3,5]
        else:
            s2ns = r_s2ns + ['OCI/PACE']
            MODIS = False
            PACE = True
            dataset = '(b) Hyperspectral'
            ks = [4,5]

        #embed(header='fig_all_ic 571')
        Adict, Bdict = anly_utils.calc_ICs(
            ks, s2ns, use_LM=use_LM, MODIS=MODIS, PACE=PACE)
            
        # Generate a pandas table
        D_BIC_A = Bdict[ks[0]] - Bdict[ks[1]]


    #embed(header='fig_all_bic 660')

        ax = plt.subplot(gs[ss])
        D_BIC = D_BIC_A #if ss == 0 else D_BIC_B
        subset = f'{ks[0]},{ks[1]}'
        for ss, s2n in enumerate(s2ns):
            if log_x:
                xvals = np.log10(D_BIC[ss] + 6.)
            else:
                xvals = D_BIC[ss]
            # CDF
            srt = np.sort(xvals)
            yvals = np.arange(srt.size) / srt.size
            try:
                fs2n = int(1./float(s2n))
                color = None
                ls = ':'
                lw = 2
            except ValueError:
                fs2n = s2n
                color = 'k'
                ls = '-'
                lw = 3
            ax.plot(srt, yvals, label=f'S/N={fs2n}', color=color, 
                    linewidth=lw, ls=ls)
            # Stats
            print(f'{subset}, {fs2n} -------------')
            print(f'% with BIC > 0: {100*np.sum(srt > 0)/srt.size}')
        if log_x:
            ax.set_xlabel(r'$\log_{10}(\Delta \, \rm '+xlbl+'_{'+f'{subset}'+r'} + 6)$')
        else:
            ax.set_xlabel(r'$\Delta \, \rm '+xlbl+'_{'+f'{subset}'+r'}$')

        # Make it pretty
        # Title
        ax.text(0.9, 0.7, dataset, ha='right', va='top', 
                fontsize=19, transform=ax.transAxes)

        ax.set_ylabel('CDF')
        ax.grid(True)
        plotting.set_fontsize(ax, 17)
        #
        xmax = 30. #if MODIS else 50.
        if not log_x:
            ax.set_xlim(-5., xmax)
        else:
            ax.set_xlim(0.25, None)
        ax.set_ylim(0.,1)
        ax.legend(fontsize=17)

        # Vertical line at 0
        vline = 0.
        if log_x:
            vline = np.log10(vline + 6.)
        ax.axvline(vline, color='r', linestyle='--', lw=2)
        # Grab ylimits
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        #ax.text(5, 0.6, 'Complex model favored', fontsize=18, ha='left')

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


def fig_Sexp(outfile='fig_Sexp.png', kmodel:int=4):


    # Load
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    aph = ds.aph.data
    anw = ds.anw.data

    ks = [3,4,5]
    pdict = {}
    for k in ks:
        pdict[k] = {}
        # Model names
        if k == 3:
            model_names = ['Exp', 'Cst']
        elif k == 4:
            model_names = ['Exp', 'Pow']
        elif k == 5:
            model_names = ['ExpBricaud', 'Pow']
        else:
            raise ValueError("Bad k")

        chain_file = anly_utils.chain_filename(
            model_names, 0.02, False, 'L23', use_LM=True,
            PACE=True)
        # Load up
        d = np.load(chain_file)
        # Parse
        pdict[k]['params'] = d['ans']
        if k == ks[0]:
            pdict['Rrs'] = d['obs_Rrs']
            pdict['idx'] = d['idx']

    Sexp = pdict[kmodel]['params'][:,1]
    i440 = np.argmin(np.abs(l23_wave-440.))
    aph_anw = aph[:,i440]/anw[:,i440]

    xmin, xmax = 0.08, 0.9
    #
    cut = Sexp > -6.
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    #
    ax.scatter(aph_anw[cut], 10**Sexp[cut], s=1, color='k')
    # Sg
    #ax.fill_between([xmin, xmax], [np.log10(0.01)]*2, [np.log10(0.02)]*2, color='cyan', alpha=0.3, label=r'$S_g$')
    ax.fill_between([xmin, xmax], [0.01]*2, [0.02]*2, color='cyan', alpha=0.3, label=r'$S_g$')
    # Sd
    ax.fill_between([xmin, xmax], [0.007]*2, [0.015]*2, color='yellow', alpha=0.3, label=r'$S_d$')
    #ax.fill_between([xmin, xmax], [np.log10(0.007)]*2, [np.log10(0.015)]*2, color='brown', alpha=0.3, label=r'$S_d$')
    # Fit to a_dg
    #ax.fill_between([xmin, xmax], [np.log10(adg_fits[:,1].min())]*2, [np.log10(adg_fits[:,1].max())]*2, color='yellow', alpha=0.3, label=r'$S_{dg}$')
    # Werdell2013
    #ax.axhline(np.log10(0.018), color='k', ls='--', label='GIOP')
    #ax.axhline(np.log10(0.0206), color='k', ls=':', label='GSM')
    ax.axhline(0.018, color='k', ls='--', label='GIOP')
    ax.axhline(0.0206, color='k', ls=':', label='GSM')
    # Tara extreme
    #ax.axhline(np.log10(0.004746), color='r', ls='-', label='Tara')
    ax.axhline(0.004746, color='r', ls='-', label='Tara')
    #
    #ax.set_ylabel(r'$\log_{10} \, S_{\rm exp}$')
    ax.set_ylabel(r'$S_{\rm exp} \rm \; [nm^{-1}]$')
    ax.set_xlabel(r'$[a_{\rm ph}/a_{\rm nw}] (440)$')
    ax.set_xlim(xmin, xmax)
    #
    #ax.set_ylim(-3., None)
    ax.set_ylim(0, 0.022)
    ax.legend(fontsize=15.)
    #
    plotting.set_fontsize(ax, 17.)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_aph_vs_aph(model:str, outroot='fig_aph_vs_aph',
                   no_errorbars:bool=True):
    xmin, xmax = 1e-3, 1

    # Outfile
    outfile = outroot + f'_{model}.png'

    MODIS = False
    SeaWiFS = False

    # Init
    add_noises = [False, True, True]
    error_lbls = ['No RT error\n No data error', 
                  'No RT error\n Data with error',
                  'No RT error\n No data error']
    #
    if model == 'GIOP':
        clr = 'b'
        mlbl = 'GIOP/MODIS'
        model_names = ['GIOP', 'Lee']
        MODIS = True
        scl_noises = [0.02, 'MODIS_Aqua', 'MODIS_Aqua']
    elif model == 'GSM':
        mlbl = 'GSM/SeaWiFS'
        clr = 'g'
        model_names = ['GSM', 'GSM']
        SeaWiFS = True
        scl_noises = [0.02, 'SeaWiFS', 'SeaWiFS']
    else:
        raise ValueError("Not ready for this model")

    # Load
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    aph = ds.aph.data
    i440_l23 = np.argmin(np.abs(l23_wave-440.))
    l23_a440 = aph[:,i440_l23]

    k_g = model

    all_ga440 = []
    all_sig_ga440 = []
    for ss, scl_noise in enumerate(scl_noises):
        # Load
        chain_file = anly_utils.chain_filename(
            model_names, scl_noise, add_noises[ss], 
            MODIS=MODIS, SeaWiFS=SeaWiFS)
        chain_file = chain_file.replace('BING', 'BING_LM')
        # Load up
        print(f'Loading {chain_file}')
        d = np.load(chain_file)

        # Flags
        #embed(header='fig_aph_vs_aph 1206')

        # Load models
        models = model_utils.init(model_names, d['wave'])

        # Calculate
        perrs = [np.sqrt(np.diag(item)) for item in d['cov']]
        perrs = np.array(perrs)

        g_a440, sig_a440 = anly_utils.calc_aph440(
            models, d['Chl'], d['ans'], perrs, 1)
        # Save
        all_ga440.append(g_a440)
        all_sig_ga440.append(sig_a440)

    fig = plt.figure(figsize=(7,10))
    gs = gridspec.GridSpec(2,1)


    naxes = 2
    for ss in range(naxes):
        ax = plt.subplot(gs[ss])

        if ss == 0 or no_errorbars:
            ax.scatter(l23_a440, all_ga440[ss], s=1, 
                   color=clr)#, label=model)
        else:
            ax.errorbar(l23_a440, all_ga440[ss], yerr=all_sig_ga440[ss], 
                color=clr, fmt='o', markersize=1)
        #
        ax.plot([xmin, xmax], [xmin, xmax], 'k--', label='1 to 1')
        if ss == 0:
            ax.plot([xmin, xmax], [2*xmin, 2*xmax], 'k:', label='2 to 2')
            ax.plot([xmin, xmax], [xmin/2, xmax/2], 'k-.', label='0.5 to 0.5')
        # Log
        #
        ax.set_ylim(1e-3, 0.99)
        ax.grid()

        # Errors
        efsz = 19.
        ax.text(0.95, 0.05, mlbl+'\n\n'+error_lbls[ss], fontsize=efsz, 
                transform=ax.transAxes, ha='right')

        plotting.set_fontsize(ax, 17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #
        if ss == naxes-1:
            ax.set_xlabel(r'$a_{\rm ph}^{\rm L23} (440)$')
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels
        ax.set_ylabel(r'$a_{\rm ph}^{\rm '+f'{model}'+r'} (440)$')

        if ss == 0:
            ax.legend(fontsize=15.)

    # Write
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_aph_and_bbnw(model_names:list, outroot='fig_aph_and_bbnw',
                scl_noise:float=0.02, add_noise:bool=False, 
                SeaWiFS:bool=False, MODIS:bool=False,
                bb_wv:int=440, # Wave for bbnw
                aph_wv:int=440, # Wave for bbnw
                BING_file:str=None,
                PACE:bool=False,
                outfile:str=None):


    # Outfile
    if outfile is None:
        outfile = outroot + f'_{model_names[0]}{model_names[1]}.png'

    # Load
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    aph = ds.aph.data
    iawv_l23 = np.argmin(np.abs(l23_wave-aph_wv))
    ibwv_l23 = np.argmin(np.abs(l23_wave-bb_wv))
    l23_aph = aph[:,iawv_l23]
    l23_bbnw = ds.bbnw.data
    l23_bbnw = l23_bbnw[:,ibwv_l23]

    if add_noise:
        error_text = 'Observational error'
        if PACE:
            scl = 3.
        else:
            scl = 10.
    else:
        error_text = 'No observational error'
        scl = 2.

    if MODIS:
        sat = 'MODIS'
    elif SeaWiFS:
        sat = 'SeaWiFS'
    elif PACE:
        sat = 'PACE'
    else:
        raise IOError("Bad satellite")
        

    # Load
    if BING_file is None:
        chain_file = anly_utils.chain_filename(
            model_names, scl_noise, add_noise,
            MODIS=MODIS, SeaWiFS=SeaWiFS, PACE=PACE)
        chain_file = chain_file.replace('BING', 'BING_LM')
        # Load up
        print(f'Loading {chain_file}')
        d = np.load(chain_file)

        models = model_utils.init(model_names, d['wave'])

        # More
        ibbnw = np.argmin(np.abs(d['wave']-models[1].pivot))
        l23_bbnw = l23_bbnw[:,ibbnw]

        # Specifics
        if model_names[1] == 'Lee':
            Y = d['Y']
        else:
            Y = None

        # aph
        perrs = [np.sqrt(np.diag(item)) for item in d['cov']]
        perrs = np.array(perrs)

        if models[0].name in ['ExpBricaud', 'ExpBricaudFix']:
            aph_idx = 2
        else:
            aph_idx = 1
        g_aph, sig_aph = anly_utils.calc_aph(
            models, d['Chl'], d['ans'], perrs, aph_idx,
            wave=aph_wv)

        # bbnw
        if models[1].name == 'Pow':
            bbnw_idx = d['ans'].shape[1]-2
            nbbnw = 2
        else:
            bbnw_idx = d['ans'].shape[1]-1
            nbbnw = 2

        bbnw = anly_utils.calc_bbnw(
            models, d['ans'], perrs, bbnw_idx, nbbnw, bb_wv, Y=Y)
    else:
        # Load
        df_bing = pandas.read_csv(BING_file)
        # Extract
        g_aph = df_bing['aph_440'].values
        sig_aph = df_bing['sig_aph_440'].values

        bbnw = df_bing['bbp_440'].values
        sig_bbnw = df_bing['sig_bbp_440'].values

        # REMOVE THIS!
        #ds = loisel23.load_ds(4,0)
        #iwave = np.argmin(np.abs(ds.Lambda.data - 440))
        #bbw_440=ds.bb.data[0,iwave]-ds.bbnw.data[0,iwave]
        #bbnw -= bbw_440


    def plot_lines(ax, xmin, xmax, scl):
        ax.plot([xmin, xmax], [xmin, xmax], 'k--', label='1 to 1')
        ax.plot([xmin, xmax], [scl*xmin, scl*xmax], 'k:', label=f'{scl} to 1')
        ax.plot([xmin, xmax], [xmin/scl, xmax/scl], 'k-.', label=f'{1./scl:0.1f} to 1')

    # Figures
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # ########################################################
    # aph
    ax_ph = plt.subplot(gs[0])

    # Non detections
    non_d = g_aph < 3*sig_aph
    ax_ph.scatter(l23_aph[~non_d], g_aph[~non_d], s=1, color='b')#, label=model)
    ax_ph.scatter(l23_aph[non_d], g_aph[non_d], s=1, edgecolors='b',
                    facecolors='none', alpha=0.3)#, label=model)


    xmin_aph, xmax_aph = 5e-4, 1
    ymin_aph, ymax_aph = 2e-5, 1
    plot_lines(ax_ph, xmin_aph, xmax_aph, scl)
    ax_ph.set_xlim(xmin_aph, xmax_aph)
    ax_ph.set_ylim(ymin_aph, ymax_aph)
    ax_ph.grid()

    aph_lbl = model_names[0] if model_names[0] != 'ExpBricaud' else '[k=5]'
    ax_ph.set_xlabel(r'$a_{\rm ph}^{\rm L23}$'+f'({int(aph_wv)})')
    ax_ph.set_ylabel(r'$a_{\rm ph}^{\rm '+f'{aph_lbl}'+r'}'+f'({int(aph_wv)})'+r'$')

    def calc_stats(x, y, sigy):
        bias = np.nanmedian(y/x)
        diff = x - y
        std = np.nanstd(diff/x)
        mae = np.nanmean(np.abs(diff)/x)
        #
        return std, bias, np.median(sigy/y), mae

    # Stats 
    std, bias, err, mae = calc_stats(l23_aph, g_aph, sig_aph)
    #embed(header='fig_aph_and_bbnw 1463')
    print(f'aph stats: bias={bias:0.2f}, std={std:0.2f}, mae={mae:0.2f}')

    high_aph = l23_aph > 0.01
    std2, bias2, _, mae2 = calc_stats(l23_aph[high_aph], 
                                      g_aph[high_aph], 
                                      sig_aph[high_aph])
    print(f'aph stats with l23_aph>0.01: bias={bias2:0.2f}, mae={mae2:0.2f}')

    # Text
    ax_ph.text(0.95, 0.10, 
               f'{sat}\n bias={int(100*bias)-100}%\n MAE={int(100*mae)}%\nRMS={int(100*std)}%',
               fontsize=17,
               transform=ax_ph.transAxes, ha='right')

    # Label the top x-axis with Chl
    ax2 = ax_ph.twiny()
    Chl_lim = np.array([xmin_aph, xmax_aph])/0.05582
    ax2.set_xlim(Chl_lim)
    #ax2.set_xticks([0.01, 0.1, 1])
    #ax2.set_xticklabels([0.01, 0.1, 1])
    ax2.set_xlabel('Chl [mg '+r'$\rm m^{-3}]$')
    

    
    # #####################################################################
    # bbnw
    ax_bb = plt.subplot(gs[1])

    ax_bb.scatter(l23_bbnw, bbnw, s=1, color='r')#, label=model)
    xmin_bb, xmax_bb = 4e-5, 3e-2
    plot_lines(ax_bb, xmin_bb, xmax_bb, scl)
    ax_bb.set_ylim(xmin_bb, xmax_bb)
    ax_bb.grid()

    ax_bb.set_xlabel(r'$b_{\rm b,nw}^{\rm L23} '+f'({int(bb_wv)})'+r'$')
    ax_bb.set_ylabel(r'$b_{\rm b,nw}^{\rm '+f'{aph_lbl}'+r'}'+f' ({int(bb_wv)})'+r'$')

    std, bias, err, mae = calc_stats(l23_bbnw, bbnw, sig_bbnw)
    print(f'bb stats: bias={bias:0.2f}, std={std:0.2f}')

    ax_bb.text(0.95, 0.10, 
               f'\n\n bias={int(100*bias)-100}%\n MAE={int(100*mae)}%\nRMS={int(100*std)}%',
               fontsize=17,
               transform=ax_bb.transAxes, ha='right')

    
    for ss, ax in enumerate([ax_ph, ax_bb, ax2]):
        plotting.set_fontsize(ax, 17)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #
        if ss == 0:
            ax.legend(fontsize=15.)

    # Write
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_bing_figs(p, outroot:str, idx:int=2773, 
                         make_fit:bool=True,
                         make_corner:bool=True,
                         make_anw:bool=True,
                         ):

    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=p.wv_min, wv_max=p.wv_max)
    l23_wave = odict['true_wave']

    model_wave = anly_utils_20.pace_wave(
        wv_min=p.wv_min, wv_max=p.wv_max)
    use_model_names = p.model_names.copy()
    models = model_utils.init(use_model_names, model_wave)

    # Load chains
    chain_file = anly_utils_20.chain_filename(p, idx=idx)
                                              #path='../../bing_2.0/Analysis/Fits')
    d = np.load(chain_file)

    # Init the other stuff..
    _ = model_utils.init_other_bits(models, Chl=d['Chl'], 
                                    Y=d['Y'])

    # Fit
    if make_fit:
        outfile1 = f'fig_bing_fit_{outroot}.png'
        bing_plot.show_fits(
            models, d['chains'], d['Chl'], d['Y'],
            Rrs_true=dict(wave=model_wave, spec=d['obs_Rrs'], var=d['varRrs']),
            anw_true=dict(wave=l23_wave, spec=odict['anw']),
            bbnw_true=dict(wave=l23_wave, spec=odict['bbnw']),
            perc=(16, 84), outfile=outfile1,
            )

    chains = d['chains']
    burn = 7000
    thin = 1
    coeff = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    print(f'There are {coeff.shape[0]} samples in the corner plot')

    # Corner plot
    if make_corner:
        # Labels
        clbls = models[0].pnames + models[1].pnames
        # Add log 10
        clbls = [r'$\log_{10}('+f'{clbl}'+r'$)' for clbl in clbls]
        # Fix Sedg
        clbls[1] = f'{clbls[1]}'

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
                p_16, p_84 = np.percentile(coeff[:,ss], [16, 84], axis=0)
                # Plot a vertical line
                ax.axvline(p_16, color='b', linestyle=':')
                ax.axvline(p_84, color='b', linestyle=':')
                ss += 1
        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)

        outfile2 = f'fig_bing_corner_{outroot}.png'
        plt.savefig(outfile2, dpi=300)
        print(f"Saved: {outfile2}")

    # a_nw
    if make_anw:
        outfile3 = f'fig_bing_anw_{outroot}.png'
        bing_plot.show_anw_fits(
            models, coeff,
            anw_true=dict(
                wave=l23_wave, a_dg=odict['adg'],
                a_ph=odict['aph']),
            perc=(16, 84), outfile=outfile3,)

def fig_pace_chi2(outfile:str='fig_pace_chi2.png',
                  model_names:list=['ExpBricaud', 'Pow'],
                  scl_noise='PACE', add_noise=True):

    # Load up
    ds = loisel23.load_ds(4,0)
    # Unpack
    wave = ds.Lambda.data
    Rrs = ds.Rrs.data
    a = ds.a.data
    bb = ds.bb.data
    aph = ds.aph.data

    i440 = np.argmin(np.abs(wave-440.))

    Chl = aph[:,i440] / 0.05582

    model_wave = anly_utils.PACE_wave
    models = model_utils.init(model_names, model_wave)
    nparam = models[0].nparam + models[1].nparam

    # Fits
    fit_file = anly_utils.chain_filename(
        model_names, scl_noise, add_noise, use_LM=True, 
        MODIS=False, PACE=True, SeaWiFS=False)
    d = np.load(fit_file)

    # Reconstruct
    #embed(header='fig_pace_chi2 1598')
    model_Rrs, a_mean, bb_mean = evaluate.reconstruct_chisq_fits(
        models, d['ans'], Chl=d['Chl'])#, bb_basis_params=None)

    # Calc chi2
    chi2s = []
    for ss in range(Rrs.shape[0]):
        chi2 = (model_Rrs[ss] - d['obs_Rrs'][ss])**2 / d['varRrs'][ss]
        chi2s.append(np.sum(chi2)/(Rrs.shape[1]-nparam))

    # Plot
    fig = plt.figure(figsize=(8,6))
    plt.clf()
    ax = plt.gca()

    ax.plot(Chl, chi2s, 'bo')

    ax.set_xlabel(r'$\rm Chl \; [mg \, m^{-3}]$')
    ax.set_ylabel(r'$\chi^2_\nu$')

    ax.set_xscale('log')
    ax.axhline(1., color='r', linestyle='--', label=r'$\chi^2_\nu=1$')

    plotting.set_fontsize(ax, 17)

    # Finish
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_four_panel_fit(p, idx, outfile:str,
             perc:tuple=(16,84), fontsize=12.):

    # Load up
    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=p.wv_min, wv_max=p.wv_max)
    l23_wave = odict['true_wave']

    model_wave = anly_utils_20.pace_wave(
        wv_min=p.wv_min, wv_max=p.wv_max)
    use_model_names = p.model_names.copy()
    models = model_utils.init(use_model_names, model_wave)

    # Load chains
    chain_file = anly_utils_20.chain_filename(p, idx=idx)
                                              #path='../../bing_2.0/Analysis/Fits')
    d = np.load(chain_file)

    # Init the other stuff..
    _ = model_utils.init_other_bits(models, Chl=d['Chl'], 
                                    Y=d['Y'])

    # Pack up
    Rrs_true=dict(wave=model_wave, spec=d['obs_Rrs'], var=d['varRrs'])
    anw_true=dict(wave=l23_wave, spec=odict['anw'])
    bbnw_true=dict(wave=l23_wave, spec=odict['bbnw'])

   # Unpack a little
    wave = models[0].wave
    chains = d['chains']

    a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains, perc=perc)
    # Generate params just in case
    params = np.median(chains, axis=[0,1])

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    # TODO -- FIX THIS!
    # THIS IS A HACK UNTIL I CAN RESOLVE bbw
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    idx = 170 # Random choie
    l23_bb = ds.bb.data[idx] 
    l23_bbnw = ds.bbnw.data[idx] 
    l23_bbw = l23_bb - l23_bbnw
    # Interpolate
    bb_w = np.interp(wave, l23_wave, l23_bbw)

    fig = plt.figure(figsize=(10,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)


    # #########################################################
    # a without water

    anw_clr = 'green'
    ax_anw = plt.subplot(gs[1])
    if anw_true is not None:
        ax_anw.plot(anw_true['wave'], anw_true['spec'], 'ko', label='True', zorder=1)
    ax_anw.plot(wave, a_mean-a_w, '-', color=anw_clr, label='Retreival')

    ax_anw.fill_between(wave, a_5-a_w, a_95-a_w, 
            color=anw_clr, alpha=0.5, label='Uncertainty') 

    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    # #########################################################
    # bb nw
    bb_clr = 'red'
    ax_bb = plt.subplot(gs[2])
    if bbnw_true is not None:
        ax_bb.plot(bbnw_true['wave'], bbnw_true['spec'], 'ko', label='True', zorder=1)
    ax_bb.plot(wave, bb_mean-bb_w, '-', color=bb_clr, label='Retrieval')
    ax_bb.fill_between(wave, bb_5-bb_w, bb_95-bb_w,
            color=bb_clr, alpha=0.5, label='Uncertainty') 

    #ax_bb.set_xlabel('Wavelength (nm)')
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')

    # #########################################################
    # Rs
    R_clr = 'orange'
    ax_R = plt.subplot(gs[0])
    if Rrs_true is not None:
        if 'var' in Rrs_true.keys():
            # Calcualte chi^2
            Rsig=np.sqrt(Rrs_true['var'])
            f = interp1d(wave, model_Rrs)
            mod_R = f(Rrs_true['wave'])
            chi2 = np.sum((Rrs_true['spec']-mod_R)**2 / Rsig**2)
            nparam = models[0].nparam + models[1].nparam
            red_chi2 = chi2 / (Rsig.size-nparam)
            #
            ax_R.errorbar(Rrs_true['wave'], Rrs_true['spec'], 
                yerr=Rsig, color='k', fmt='o', capsize=5,
                label=r'$\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$') 
        else:
            ax_R.plot(Rrs_true['wave'], Rrs_true['spec'], 'k+', label='True', zorder=1)
    #ax_R.plot(wave, gordon_Rrs, 'k+', label='L23 + Gordon')
    ax_R.plot(wave, model_Rrs, '-', color=R_clr, label='Fit', zorder=10)
    ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
            color=R_clr, alpha=0.5, zorder=10) 

    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')

    # Log scale y-axis
    ax_R.set_yscale('log')

    # #########################################################
    # aph, adg
    burn = 7000
    thin = 1
    prep_chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    ax_adgph = plt.subplot(gs[3])

    _ = bing_plot.show_anw_fits(models, prep_chains,
            anw_true=dict(
                wave=odict['wave'], a_dg=odict['adg'],
                a_ph=odict['aph']),
            perc=(16, 84), ax_anw=ax_adgph,
            no_show=True)
    
    # axes
    axes = [ax_anw, ax_bb, ax_R, ax_adgph]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
        


# ############################################################
def fig_multi_model(ps, lbls, idx:int, outfile:str,
             perc:tuple=(5,95), fontsize=12.):

    ls = ['-', '--', ':']

    # Load up
    odict = anly_utils_20.prep_l23_data(
        idx, wv_min=ps[0].wv_min, wv_max=ps[0].wv_max)
    l23_wave = odict['true_wave']

    model_wave = anly_utils_20.pace_wave(
        wv_min=ps[0].wv_min, wv_max=ps[0].wv_max)

    # Pack up
    anw_true=dict(wave=l23_wave, spec=odict['anw'])
    bbnw_true=dict(wave=l23_wave, spec=odict['bbnw'])
    wave = model_wave

    # Reconstruct the models

    bb_means = []
    all_chains = []
    model_Rrss = []
    adgs = []
    aphs = []
    for ss, p in enumerate(ps):
        models = model_utils.init(p.model_names, model_wave)
        # Load chains
        chain_file = anly_utils_20.chain_filename(p, idx=idx)
                                                #path='../../bing_2.0/Analysis/Fits')
        d = np.load(chain_file)

        # Init the other stuff..
        _ = model_utils.init_other_bits(models, Chl=d['Chl'], Y=d['Y'])

        # Reconstruct
        chains = d['chains']
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
                model_Rrs, sigRs = evaluate.reconstruct_from_chains(
                models, chains, perc=perc)

        # adg, aph
        burn = 7000
        thin = 1
        prep_chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
        a_dg, a_ph = models[0].eval_anw(prep_chains[..., :models[0].nparam],
                            retsub_comps=True)
        adg_mean = np.median(a_dg, axis=0)
        aph_mean = np.median(a_ph, axis=0)
        # Save
        bb_means.append(bb_mean)
        all_chains.append(chains)
        model_Rrss.append(model_Rrs)
        adgs.append(adg_mean)
        aphs.append(aph_mean)

        # Stats
        i440 = np.argmin(np.abs(model_wave-440.))
        aph440 = aph_mean[i440]
        print(f'{p.model_names[0]}: a_ph(440)={aph440:0.3f}')

        # One more
        if ss == 0:
            Rrs_true=dict(wave=model_wave, spec=d['obs_Rrs'], var=d['varRrs'])

    # THIS IS A HACK UNTIL I CAN RESOLVE bbw
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    idx = 170 # Random choice
    l23_bb = ds.bb.data[idx] 
    l23_bbnw = ds.bbnw.data[idx] 
    l23_bbw = l23_bb - l23_bbnw
    # Interpolate
    bb_w = np.interp(wave, l23_wave, l23_bbw)

    fig = plt.figure(figsize=(10,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)


    # #########################################################
    # bb nw
    bb_clr = 'red'
    ax_bb = plt.subplot(gs[2])
    ax_bb.plot(bbnw_true['wave'], bbnw_true['spec'], 'ko', label='True', zorder=1)

    # Loop on the models
    for p, lbl, bb_mean, l in zip(ps, lbls, bb_means, ls):
        ax_bb.plot(wave, bb_mean-bb_w, l, color=bb_clr, label=lbl)

    #ax_bb.set_xlabel('Wavelength (nm)')
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')


    # #########################################################
    # Rs
    R_clr = 'orange'
    ax_R = plt.subplot(gs[0])

    Rsig=np.sqrt(Rrs_true['var'])
    ax_R.errorbar(Rrs_true['wave'], Rrs_true['spec'], 
            yerr=Rsig, color='k', fmt='o', capsize=5,
            label='Obs')

    # Loop on the models
    for model_Rrs, lbl, l in zip(model_Rrss, lbls, ls):
        # Calcualte chi^2
        f = interp1d(wave, model_Rrs)
        mod_R = f(Rrs_true['wave'])
        chi2 = np.sum((Rrs_true['spec']-mod_R)**2 / Rsig**2)
        nparam = models[0].nparam + models[1].nparam
        red_chi2 = chi2 / (Rsig.size-nparam)
            #
        ax_R.plot(wave, model_Rrs, l, color=R_clr, zorder=10,
                label=lbl+r': $\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$') 

    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')

    # Log scale y-axis
    ax_R.set_yscale('log')

    # #########################################################
    # aph
    aph_clr = 'green'
    ax_aph = plt.subplot(gs[1])
    ax_aph.plot(odict['wave'], odict['aph'], 'ko', label='True', zorder=1)

    # Loop on the models
    for p, lbl, aph, l in zip(ps, lbls, aphs, ls):
        ax_aph.plot(wave, aph, l, color=aph_clr, label=lbl)
    ax_aph.set_ylabel(r'$a_{ph}(\lambda) \; [{\rm m}^{-1}]$')

    # #########################################################
    # adg
    adg_clr = 'blue'
    ax_adg = plt.subplot(gs[3])
    ax_adg.plot(odict['wave'], odict['adg'], 'ko', label='True', zorder=1)

    # Loop on the models
    for p, lbl, adg, l in zip(ps, lbls, adgs, ls):
        ax_adg.plot(wave, adg, l, color=adg_clr, label=lbl)
    ax_adg.set_ylabel(r'$a_{dg}(\lambda) \; [{\rm m}^{-1}]$')

    # axes
    axes = [ax_aph, ax_bb, ax_R, ax_adg]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)


    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
        


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Spectra -- blue/red with water
    if flg == 1:
        fig_spectra(170)#, bbscl=20)

    if flg == 2:
        fig_multi_fits()#[('Cst','Cst'), ('Exp','Cst'), ('Exp','Pow'), ('ExpBricaud','Pow')], 
                       #[170, 1032])

    # Figure 3
    if flg == 3:
        fig_bic_modis_pace()


    # BIC/AIC for PACE
    if flg == 5:
        fig_all_ic(PACE=True, outfile='fig_all_bic_PACE.png',
                   log_x=False)

    # BIC/AIC for PACE
    if flg == 6:
        fig_all_ic(SeaWiFS=True, outfile='fig_all_bic_SeaWiFS.png',
                   comp_ks=((2,3), (3,4)),
                   log_x=False)



    # ########################################
    # Supp
    if flg == 10:
        fig_u()

    if flg == 11:
        fig_Kd()

    # Satellite Noise
    if flg == 12:
        #fig_satellite_noise('SeaWiFS', 443)
        #fig_satellite_noise('SeaWiFS', 670)
        #fig_satellite_noise('MODIS_Aqua', 443)
        #fig_satellite_noise('MODIS_Aqua', 667)
        fig_pace_noise()


    if flg == 13:
        fig_Sexp()

    # aph and bbnw
    if flg == 14:
        # GIOP
        '''
        fig_aph_and_bbnw(['GIOP', 'Lee'], MODIS=True)
        fig_aph_and_bbnw(['GIOP', 'Lee'], MODIS=True, add_noise=True,
                         scl_noise='MODIS_Aqua',
                         outfile='fig_aph_and_bbnw_GIOP_noise.png')
        # GSM
        fig_aph_and_bbnw(['GSM', 'GSM'], SeaWiFS=True)
        fig_aph_and_bbnw(['GSM', 'GSM'], SeaWiFS=True, add_noise=True,
                         scl_noise='SeaWiFS', 
                         outfile='fig_aph_and_bbnw_GSM_noise.png')
        '''
        # PACE
        #fig_aph_and_bbnw(['GIOP', 'Lee'], PACE=True, add_noise=True,
        #                 scl_noise='PACE',
        #                 outfile='fig_aph_and_bbnw_GIOP_PACE_noise.png')
        fig_aph_and_bbnw(['ExpBricaud', 'Pow'], PACE=True, 
                         BING_file='../Analysis/BING_L23_results_ExpBricaudPow.csv',
                         add_noise=True, scl_noise='PACE',
                         outfile='fig_aph_and_bbnw_k5_PACE.png')
        fig_aph_and_bbnw(['GIOP', 'Lee'], PACE=True, 
                         BING_file='../Analysis/BING_L23_results_GIOPLee.csv',
                         add_noise=True, scl_noise='PACE',
                         outfile='fig_aph_and_bbnw_GIOP_PACE.png')


    # BIC/AIC for MODIS+L23
    if flg == 15:

        #fig_all_ic(MODIS=True, outfile='fig_all_bic_MODIS.png',
        #           log_x=False,
        #           comp_ks=((2,3), (3,4)))
        fig_all_ic(MODIS=True, outfile='fig_bic_MODIS_GIOP.png',
                   log_x=False,
                   comp_ks=((3,'GIOP'), (3,'GIOP+')), xmax=5)

    # BIC/AIC for SeaWiFS+GSM
    if flg == 16:

        fig_all_ic(SeaWiFS=True, outfile='fig_all_bic_SeaWiFS.png',
                   log_x=False,
                   comp_ks=((2,3), (3,4)))
        fig_all_ic(SeaWiFS=True, outfile='fig_bic_SeaWiFS_GSM.png',
                   log_x=False,
                   comp_ks=((3,'GSM'), (3,'GSM')), xmax=5)
        #fig_all_ic(MODIS=True, show_AIC=True, 
        #           outfile='fig_all_aic_MODIS.png')
        #fig_all_ic(MODIS=True, outfile='fig_all_bic_MODIS_GIOP.png',
        #           comp_ks=((2,3), (3,9)))

    # Degenerate solutions
    if flg == 17:
        # Actual fits
        #fig_mcmc_fit(['Every', 'Every'], idx=170, full_LM=False,
        #    use_LM=False)
        #fig_mcmc_fit(['Every', 'GSM'], idx=170, full_LM=False,
        #    use_LM=False)
        # Degenerates
        fig_degenerate_fits(['Every', 'Every'], idx=170, 
                           full_LM=False, use_LM=False)


    # Fits
    if flg == 30:
        #fig_mcmc_fit(['Exp', 'Pow'], idx=170, log_Rrs=True)
        #fig_mcmc_fit(['Exp', 'Pow'], idx=170, log_Rrs=True, use_LM=True)
        #fig_mcmc_fit(['Exp', 'Cst'], idx=170, log_Rrs=True, use_LM=True)
        #fig_mcmc_fit(['Exp', 'Cst'], idx=3315, log_Rrs=True, use_LM=True)
        #fig_mcmc_fit(['Exp', 'Pow'], idx=3315, log_Rrs=True, use_LM=True)
        #fig_mcmc_fit(['Cst', 'Cst'], idx=170, log_Rrs=True, use_LM=True)
        #fig_mcmc_fit(['Exp', 'Pow'], idx=170, 
        #             log_Rrs=True, use_LM=True, max_wave=700.)#, full_LM=False)
        #fig_mcmc_fit(['ExpBricaud', 'Pow'], idx=170, 
        #             log_Rrs=True, use_LM=True, max_wave=700.)#, full_LM=False)
        #fig_mcmc_fit(['ExpNMF', 'Pow'], idx=170, full_LM=False,
        #             log_Rrs=True, use_LM=True, max_wave=700.)#, full_LM=False)
        #fig_mcmc_fit(['ExpBricaud', 'Pow'], idx=170, full_LM=True,
        #fig_mcmc_fit(['GIOP', 'Lee'], idx=170, full_LM=True,
        #fig_mcmc_fit(['GIOP', 'Pow'], idx=170, full_LM=True,
        #    PACE=True, log_Rrs=True, use_LM=True)#, full_LM=False)
        #fig_mcmc_fit(['GSM', 'GSM'], idx=170, full_LM=False,
        #    PACE=True, log_Rrs=True, use_LM=False)#, full_LM=False)
        pass

    # Bayesian fits
    if flg == 31:
        fig_mcmc_fit(['GSM', 'GSM'], idx=170, full_LM=False, 
            SeaWiFS=True, use_LM=False, scl_noise='SeaWiFS')#, full_LM=False)
        fig_mcmc_fit(['GIOP', 'Lee'], idx=170, full_LM=False, 
            MODIS=True, use_LM=False, scl_noise='MODIS_Aqua')#, full_LM=False)

    # Corner
    if flg == 32:
        fig_corner(['GSM', 'GSM'], idx=170, full_LM=False,
            SeaWiFS=True, use_LM=False, scl_noise='SeaWiFS',
            show_log=True, add_noise=True)
        #fig_corner(['GSM', 'GSM'], idx=170, full_LM=False,
        #    SeaWiFS=True, use_LM=False, scl_noise='SeaWiFS',
        #    show_log=True)
        #fig_corner(['GIOP', 'Lee'], idx=170, full_LM=False,
        #    MODIS=True, use_LM=False, scl_noise='MODIS_Aqua',
        #    show_log=True)
        #fig_corner(['GIOP', 'Lee'], idx=1032, full_LM=False,
        #    MODIS=True, use_LM=False, scl_noise='MODIS_Aqua',
        #    show_log=True)

    # High aph
    if flg == 33:
        #fig_multi_fits(indices=[170,2590])
        fig_multi_fits(indices=[605,2951])

    # Individual 
    if flg == 34:
        # ExpBricaud, Pow
        if False:
            model_names=['ExpBricaud', 'Pow']
            p = param20.p_ntuple(model_names,
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=400., wv_max=700)

            fig_bing_figs(p, 'ExpBPow_170', make_fit=True, make_corner=True, idx=170)

        # GIOP, Lee
        if True:
            model_names=['GIOP', 'Lee']
            p = param20.p_ntuple(model_names,
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=400., wv_max=700.)

            fig_bing_figs(p, 'GIOP_2773', make_fit=True, make_corner=True)

        # GSM
        if False:
            model_names=['GSM', 'GSM']
            p = param20.p_ntuple(model_names,
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=400., wv_max=700.)

            fig_bing_figs(p, 'GSM', make_fit=True, make_corner=True)

    # PACE chi^2
    if flg == 36:
        fig_pace_chi2()

    # Low Chl, 4 panel
    if flg == 37:
        idx = 170
        model_names=['ExpBricaud', 'Pow']
        p = param20.p_ntuple(model_names,
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=400., wv_max=700)
        fig_four_panel_fit(p, idx, 'fig_low_chl_4panel.png')

    # High Chl, 4-panel
    if flg == 38:
        idx = 2773
        model_names=['ExpBricaud', 'Pow']
        p = param20.p_ntuple(model_names,
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=400., wv_max=700)
        fig_four_panel_fit(p, idx, 'fig_high_chl_4panel.png')

    # Multi-model, 4-panel
    if flg == 39:
        idx = 2773
        wv_min=400.
        wv_max=700.
        lbls = []
        # Model 1
        p1 = param20.p_ntuple(['ExpBricaud', 'Pow'],
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=wv_min, wv_max=wv_max)
        lbls.append('[k=5]')
        # Model 2
        p2 = param20.p_ntuple(['GIOP', 'Lee'],
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=wv_min, wv_max=wv_max)
        lbls.append('GIOP')
        # Model 3
        p3 = param20.p_ntuple(['GSM', 'GSM'],
                set_Sdg=False, sSdg=0.002, 
                scl_noise='PACE', 
                add_noise=True, wv_min=wv_min, wv_max=wv_max)
        lbls.append('GSM')
        #
        fig_multi_model([p1, p2, p3], lbls, idx, 'fig_multi_model.png')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # flg = 1 :: Figure 1; Spectra of water and non-water
        # flg = 2 :: Figure 2; Fits to example Rrs
        # flg = 3 :: Figure 3; BIC
        
        # flg = 10 :: Supp 1; fig_u

        # flg = 12 :: Satellite noise

        # flg = 14 :: a_ph(440), bbp(440) scatter fig_aph_and_bbnw
            # PACE and GIOP

        # flg = 17 :: Every, Every degenerate fit
        # flg = 17 :: arbitrary IOP model

        # New PACE figures
        # flg = 34 :: Rrs, anw, bbnw on high Chla
        # flg = 37 :: Low Chl, 4-panel :: fig_four_panel_fit
        # flg = 38 :: High Chl, 4-panel :: fig_four_panel_fit
        # flg = 39 :: Multi-model, 4-panel

    else:
        flg = sys.argv[1]

    main(flg)
