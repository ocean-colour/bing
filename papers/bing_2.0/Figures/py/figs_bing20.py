""" Figs for BING 2.09 """
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
import anly_utils_20

from IPython import embed

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)

def fig_uv_efficacy(
    model_names=['ExpBricaud', 'Pow'], 
    idx=170, outfile='fig_uv_efficacy.png',
            add_noise=True, PACE=True,
            scl_noise='PACE', 
            set_Sdg=0.002,
            set_beta=1.,
            max_wave:float=700.,
            nMC=100):


    # Load L23
    odict = anly_utils_20.prep_l23_data(
        idx, scl_noise=scl_noise)

    # Loop
    for min_wave in [350, 400]:
        model_wave = anly_utils_20.pace_wave(wv_min=min_wave,
                                             wv_max=max_wave)
        # Models
        #embed(header='figs 68')
        models = model_utils.init(model_names, model_wave)
        
        # Chain file
        chain_file = anly_utils_20.chain_filename(
            model_names, scl_noise, add_noise, idx=idx,
            PACE=PACE, beta=set_beta, Sdg=set_Sdg, 
            wv_min=min_wave, nMC=nMC)
        # Load
        d = np.load(chain_file)

        burn = 7000
        thin = 1
        chains = d['chains'][:,burn::thin, :, :].reshape(d['chains'].shape[0], -1, d['chains'].shape[-1])
        a_dg, a_ph = models[0].eval_anw(
            chains[..., :models[0].nparam], 
            retsub_comps=True)

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


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Spectra
    if flg == 1:
        fig_uv_efficacy()#, bbscl=20)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # flg = 1 :: Figure 1; Spectra of water and non-water

    else:
        flg = sys.argv[1]

    main(flg)
