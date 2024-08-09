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

from oceancolor.utils import plotting 
from oceancolor.hydrolight import loisel23
from oceancolor.satellites import pace as sat_pace
from oceancolor.satellites import seawifs as sat_seawifs
from oceancolor.satellites import modis as sat_modis

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


# ############################################################
def fig_build_up_the_fits(models:list=None, 
                   indices:list=None, 
                   min_wave:float=400.,
                   max_wave:float=700.,
                 outroot='fig_build_fits'): 

    all_models = [('Cst','Cst'), ('Exp','Cst'), ('Exp','Pow'), ('ExpBricaud','Pow')]
    if indices is None:
        indices = [170, 1032]

    nsteps = 7
    for ss in range(nsteps):
        outfile = outroot + f'_step{ss}.png'
        data_only=False
        Rrs_fit_only=False
        models = all_models.copy()

        if ss == 0:
            data_only=True
        elif ss == 1:  # Constant, but fit only
            Rrs_fit_only=True
            models = [all_models[0]]
        elif ss == 2:  # Constant, but fit only
            models = [all_models[0]]
        else:
            models = all_models[0:ss-1]
            
        fig = plt.figure(figsize=(12,6))
        plt.clf()
        gs = gridspec.GridSpec(2,3)

        talks_compare_models(models, indices[0], 
                    [plt.subplot(gs[0]), plt.subplot(gs[1]), 
                        plt.subplot(gs[2])],
                    lbl_wavelengths=False,
                    min_wave=min_wave, max_wave=max_wave,
                    data_only=data_only, 
                    Rrs_fit_only=Rrs_fit_only)
        talks_compare_models(models, indices[1], 
                    [plt.subplot(gs[3]), plt.subplot(gs[4]), 
                        plt.subplot(gs[5])],
                        min_wave=min_wave, max_wave=max_wave,
                        data_only=data_only, 
                    Rrs_fit_only=Rrs_fit_only)

        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")

def talks_compare_models(models:list, idx:int, axes:list, 
                   min_wave:float=None, max_wave:float=None,
                   add_noise:bool=False, scl_noise:float=None,
                   log_Rrs:bool=True, lbl_wavelengths:bool=True,
                   use_LM:bool=True, full_LM:bool=True, 
                   data_only:bool=False,
                   Rrs_fit_only:bool=False):

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
        aw = rdict['aw']
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

        if not data_only and not Rrs_fit_only:
            ax_anw.plot(wave, a_mean-aw, clr, label='Retreival')


        # #########################################################
        # b
        use_bbw = bbw
        ax_bb = axes[2]
        if ss == 0:
            ax_bb.plot(wave_true, bbnw, 'ko', label='True')
            ax_bb.set_ylabel(r'$b_{b,nw} (\lambda) \; [{\rm m}^{-1}]$')

        if not data_only and not Rrs_fit_only:
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
        if not data_only:
            ax_R.plot(wave, model_Rrs, '-', color=clr, label=f'[k={nparm}]', zorder=10)
        ax_R.legend(fontsize=lgsz, loc='lower left')

    # axes
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, 14)
        if lbl_wavelengths:
            ax.set_xlabel('Wavelength (nm)')
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels



def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Spectra
    if flg == 1:
        fig_build_up_the_fits()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)
