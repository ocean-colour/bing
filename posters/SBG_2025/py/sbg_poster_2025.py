""" Figs for SBG Poster (2025) """
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

from ocpy.utils import plotting 
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import sbg as sat_sbg

from bing.models import utils as model_utils
from bing import evaluate
from bing.parameters import standard
from bing.fitting import l23

#from bing.models import anw as bing_anw
#from bing.models import bbnw as bing_bbnw
#from bing import chisq_fit
#from bing import stats as bing_stats

# Local
sys.path.append(os.path.abspath("../../papers/bing_2.0/Analysis/py"))
import anly_utils_20

sys.path.append(os.path.abspath("../../papers/phytoplankton/Analysis/py"))
import anly_utils

from IPython import embed


# ################################
def fig_bic_sbg_pace(use_LM:bool=True, 
                outfile:str='fig_bic_sbg_pace.png', 
                log_x:bool=False):


    r_s2ns = [] #[0.05, 0.10, 0.2]

    fig = plt.figure(figsize=(14,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    MODIS = False
    add_noise = True
    xlbl = 'BIC'
    for ss in range(2):

        if ss==1:
            s2ns = r_s2ns + ['SBG']
            SBG = True
            PACE = False
            dataset = 'SBG (300m pixels)'
            ks = [4,5]
            scl_noise = 'SBG'
            chainroot='Fits/'
        else:
            s2ns = r_s2ns + ['OCI/PACE']
            PACE = True
            SBG = False
            dataset = 'PACE (1km pixels)'
            ks = [4,5]
            scl_noise = 'PACE'
            chainroot='../../papers/phytoplankton/Analysis/Fits/'

        #embed(header='fig_all_ic 571')
        Adict, Bdict = anly_utils.calc_ICs(
            ks, s2ns, add_noise=add_noise,
            use_LM=use_LM, MODIS=MODIS, 
            PACE=PACE,
            SBG=SBG, 
            chainroot=chainroot,
            scl_noise=scl_noise)
            
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
                color = 'k' if PACE else 'b'
                ls = '-'
                lw = 3
            ax.plot(srt, yvals, label=f'S/N={fs2n}', color=color, 
                    linewidth=lw, ls=ls)
            # Stats
            print(f'{subset}, {fs2n} -------------')
            print(f'% with BIC > 0: {100*np.sum(srt > 0)/srt.size}')

        ax.set_xlabel(r'$\Delta \, \rm '+xlbl+'$'+'  (without vs. with phytoplankton)')

        # Make it pretty
        # Title
        ax.text(0.9, 0.3, dataset, ha='right', va='top', 
                fontsize=23, transform=ax.transAxes)

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
        #ax.legend(fontsize=17, loc='lower right')#, frameon=False)

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
def fig_multi_model(ps, lbls, idx:int, outfile:str,
             perc:tuple=(5,95), fontsize=12.):

    ls = ['-', '--', ':']

    # Load up
    odict = l23.load_one_l23(
        idx, wv_min=ps[0].wv_min, wv_max=ps[0].wv_max)
    l23_wave = odict['true_wave']

    model_wave = sat_pace.wave(
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
        chain_file = anly_utils_20.chain_filename(
            p, idx=idx, path='Fits/')
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


# ############################################################
def fig_aph_and_bbnw(model_names:list, outroot='fig_aph_and_bbnw',
                scl_noise:float=0.02, add_noise:bool=False, 
                SeaWiFS:bool=False, MODIS:bool=False,
                bb_wv:int=440, # Wave for bbnw
                aph_wv:int=440, # Wave for bbnw
                BING_file:str=None,
                PACE:bool=False,
                SBG:bool=False,
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
        if PACE or SBG:
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
    elif SBG:
        sat = 'SBG'
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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # BIC/AIC for MODIS+L23
    if flg == 1:
        fig_bic_sbg_pace()

    # Multi model fit to one example
    if flg == 2:
        idx = 2773
        wv_min=400.
        wv_max=700.
        lbls = []

        # Model 1
        p_expb = standard.expb_pow(satellite='SBG', add_noise=True)
        p_giop = standard.giop(satellite='SBG', add_noise=True)
        p_gsm = standard.gsm(satellite='SBG', add_noise=True)

        lbls.append('[k=5]')
        lbls.append('GIOP')
        lbls.append('GSM')
        #
        fig_multi_model([p_expb, p_giop, p_gsm], 
                        lbls, idx, 'fig_SBG_multi_model.png')

    if flg == 3:
        fig_aph_and_bbnw(['GIOP', 'Lee'], SBG=True, 
                         BING_file='../Analysis/BING_L23_results_GIOPLee.csv',
                         add_noise=True, scl_noise='SBG',
                         outfile='fig_aph_and_bbnw_GIOP_SBG.png')


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

        # flg = 17 :: arbitrary IOP model

        # New PACE figures
        # flg = 34 :: Rrs, anw, bbnw on high Chla
        # flg = 37 :: Low Chl, 4-panel :: fig_four_panel_fit
        # flg = 38 :: High Chl, 4-panel :: fig_four_panel_fit
        # flg = 39 :: Multi-model, 4-panel

    else:
        flg = sys.argv[1]

    main(flg)