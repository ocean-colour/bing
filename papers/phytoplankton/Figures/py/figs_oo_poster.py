""" Figs for Gordon Analyses """
import os, sys
from importlib.resources import files

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
import pandas


from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import corner

from ocpy.utils import plotting 
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import modis as sat_modis

from bing import plotting as bing_plot
from bing.models import utils as model_utils
from bing.models import functions

#from bing.models import anw as bing_anw
#from bing.models import bbnw as bing_bbnw
#from bing import chisq_fit
#from bing import stats as bing_stats

# Local
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
def fig_multi_fits(models:list=None, 
                   indices:list=None, 
                   min_wave:float=400.,
                   max_wave:float=700.,
                 outfile='fig_oo_poster_fits.png'):

    if models is None:
        models = [('Cst','Cst'), ('Exp','Cst'), ('Exp','Pow'), ('ExpBricaud','Pow')]
    if indices is None:
        indices = [170, 1032]

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
        #embed(header='figs 407')
        ax_R.plot(wave, model_Rrs, '-', color=clr, label=f'[{models[0].name},{models[1].name}]', zorder=10)
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
    fig = plt.figure(figsize=(7,5))
    ax = plt.gca()

    # Colors
    ctotal ='gray'
    cwater ='black'
    cnw ='green'
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
    if show_total:
        ax.plot(wave, bb, ':', color=ctotal, label=r'$b_{b}$', zorder=1)

    if bbscl != 1.:
        ax.plot(wave, bbscl*bbw, ':', color=cwater, label=f'{bbscl}*'+r'$b_{b,w}$', zorder=1)
        ax.plot(wave, bbscl*bbnw, ':', color=cnw, label=f'{bbscl}*'+r'$b_{b,nw}$', zorder=1)
    else:
        ax.plot(wave, bbscl*bbw, ':', color=cwater, label=r'$b_{b,w}$', zorder=1)
        ax.plot(wave, bbscl*bbnw, ':', color=cnw, label=r'$b_{b,nw}$', zorder=1)

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

    plotting.set_fontsize(ax, 15)

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
    pace_file = files('oceancolor').joinpath(os.path.join(
        'data', 'satellites', 'PACE_error.csv'))
    actual_PACE_error = pandas.read_csv(pace_file)
    acut = (actual_PACE_error['wave'] < 700.) & (actual_PACE_error['wave'] > 400.)

    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    l23_PACE_error = sat_pace.gen_noise_vector(l23_wave)
    lcut = (l23_wave < 700.) & (l23_wave > 400.)

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
    
    # axes
    for ax in [ax_c]:
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
                lw = 1
            except ValueError:
                fs2n = s2n
                color = 'k'
                ls = '-'
                lw = 3
            ax.plot(srt, yvals, label=f'S/N={fs2n}', color=color, linewidth=lw, ls=ls)
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
                bb_wv:int=443, # Wave for bbnw
                aph_wv:int=443, # Wave for bbnw
                PACE:bool=False,
                no_errorbars:bool=True, outfile:str=None):


    # Outfile
    if outfile is None:
        outfile = outroot + f'_{model_names[0]}{model_names[1]}.png'

    # Load
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    aph = ds.aph.data
    iawv_l23 = np.argmin(np.abs(l23_wave-aph_wv))
    l23_aph = aph[:,iawv_l23]
    l23_bbnw = ds.bbnw.data

    if add_noise:
        error_text = 'Observational error'
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

    g_aph, sig_aph = anly_utils.calc_aph(
        models, d['Chl'], d['ans'], perrs, 1,
        wave=aph_wv)

    # bbnw
    bbnw_idx = d['ans'].shape[1]-1
    bbnw = anly_utils.calc_bbnw(
        models, d['ans'], perrs, bbnw_idx, bb_wv, Y=Y)

    def plot_lines(ax, xmin, xmax, scl):
        ax.plot([xmin, xmax], [xmin, xmax], 'k--', label='1 to 1')
        ax.plot([xmin, xmax], [scl*xmin, scl*xmax], 'k:', label=f'{scl} to {scl}')
        ax.plot([xmin, xmax], [xmin/scl, xmax/scl], 'k-.', label=f'{1./scl:0.1f} to {1./scl:0.1f}')

    # Figures
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # aph
    ax_ph = plt.subplot(gs[0])

    ax_ph.scatter(l23_aph, g_aph, s=1, color='b')#, label=model)
    xmin_aph, xmax_aph = 1e-4, 1
    plot_lines(ax_ph, xmin_aph, xmax_aph, scl)
    ax_ph.set_ylim(xmin_aph, xmax_aph)
    ax_ph.grid()

    ax_ph.set_xlabel(r'$a_{\rm ph}^{\rm L23}$'+f'({int(aph_wv)})')
    ax_ph.set_ylabel(r'$a_{\rm ph}^{\rm '+f'{model_names[0]}'+r'}'+f'({int(aph_wv)})'+r'$')

    def calc_stats(x, y, sigy):
        bias = np.median(y/x)
        diff = x - y
        std = np.std(diff/x)
        mae = np.mean(np.abs(diff)/x)
        #
        return std, bias, np.median(sigy/y), mae
    # Stats
    std, bias, err, mae = calc_stats(l23_aph, g_aph, sig_aph)
    print(f'aph stats: bias={bias:0.2f}, std={std:0.2f}')

    # Text
    ax_ph.text(0.95, 0.10, 
               f'{model_names[0]}/{sat}\n {error_text}\n  bias={int(100*bias)-100}%, MAE={int(100*mae)}%',
               fontsize=17,
               transform=ax_ph.transAxes, ha='right')

    
    # #####################################################################
    # bbnw
    ax_bb = plt.subplot(gs[1])

    ax_bb.scatter(l23_bbnw, bbnw, s=1, color='r')#, label=model)
    xmin_bb, xmax_bb = 1e-5, 3e-2
    plot_lines(ax_bb, xmin_bb, xmax_bb, scl)
    ax_bb.set_ylim(xmin_bb, xmax_bb)
    ax_bb.grid()

    ax_bb.set_xlabel(r'$b_{\rm b,nw}^{\rm L23} '+f'({int(bb_wv)})'+r'$')
    ax_bb.set_ylabel(r'$b_{\rm b,nw}^{\rm '+f'{model_names[0]}'+r'}'+f' ({int(bb_wv)})'+r'$')

    std, bias, err, mae = calc_stats(l23_bbnw, bbnw, sig_aph)
    print(f'bb stats: bias={bias:0.2f}, std={std:0.2f}')

    ax_bb.text(0.95, 0.10, 
               f'\n\nbias={int(100*bias)-100}%, MAE={int(100*mae)}%',
               fontsize=17,
               transform=ax_bb.transAxes, ha='right')

    
    for ss, ax in enumerate([ax_ph, ax_bb]):
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


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    if flg == 1:
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



    # Satellite Noise
    if flg == 12:
        #fig_satellite_noise('SeaWiFS', 443)
        #fig_satellite_noise('SeaWiFS', 670)
        #fig_satellite_noise('MODIS_Aqua', 443)
        fig_satellite_noise('MODIS_Aqua', 667)
        #fig_pace_noise()

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
        fig_aph_and_bbnw(['GIOP', 'Lee'], PACE=True, add_noise=True,
                         scl_noise='PACE',
                         outfile='fig_aph_and_bbnw_GIOP_PACE_noise.png')


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
        #fig_mcmc_fit(['Every', 'Every'], idx=170, full_LM=False,
        #    use_LM=False)
        fig_mcmc_fit(['Every', 'GSM'], idx=170, full_LM=False,
            use_LM=False)


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


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # flg = 1 :: Figure 1; Spectra of water and non-water
        # flg = 2 :: Figure 2; Fits to example Rrs
        # flg = 3 :: Figure 3; BIC
        
        # flg = 10 :: Supp 1; fig_u

    else:
        flg = sys.argv[1]

    main(flg)
