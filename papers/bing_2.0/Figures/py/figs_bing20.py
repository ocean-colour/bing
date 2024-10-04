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
from ocpy.utils import io as ocio
from ocpy.hydrolight import loisel23
from ocpy.satellites import pace as sat_pace

from bing import plotting as bing_plot
from bing.models import functions

#from bing.models import anw as bing_anw
#from bing.models import bbnw as bing_bbnw
#from bing import chisq_fit
#from bing import stats as bing_stats

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import anly_utils_20
import param

from IPython import embed

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)

def fig_uv_efficacy(model_names=['ExpBricaud', 'Pow'], 
    idx=170, outfile='fig_uv_efficacy.png',
    set_beta=1., nMC=100):
    wv_mins = [350, 400]

    # Parse outputs
    pdicts = []
    adg = []
    adg_16 = []
    adg_84 = []
    aph = []
    aph_16 = []
    aph_84 = []
    for ss, wv_min in enumerate(wv_mins): 
        p = param.p_ntuple(model_names,
                set_Sdg=True, sSdg=0.002, beta=set_beta, nMC=nMC,
                add_noise=True, wv_min=wv_min)
        # Load up the results
        chain_file = anly_utils_20.chain_filename(p, idx=idx)
        proc_file = chain_file.replace('.npz', '.json')

        #
        pdict = ocio.loadjson(proc_file)
        pdicts.append(pdict)
        #
        adg.append(pdict['adg_400'])
        adg_16.append(adg[-1]-pdict['adg_5']) # Fix to 16 eventually
        adg_84.append(pdict['adg_95']-adg[-1])
        # aph
        aph.append(pdict['aph_440'])
        aph_16.append(aph[-1]-pdict['aph_5'])
        aph_84.append(pdict['aph_95']-aph[-1])

        # True answer
        if ss == 0:
            adg_true = pdict['l23_adg_400']
            aph_true = pdict['l23_aph_440']

    #
    fig = plt.figure(figsize=(8,4))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    plt.clf()

    # ############################
    # a_dg
    ax_dg = plt.subplot(gs[0])
    # Plot with error
    ax_dg.errorbar(wv_mins, adg, yerr=[adg_16, adg_84], fmt='o', 
                   color='k', label='Retrieved')
    # Horizonal line at true
    ax_dg.axhline(adg_true, color='b', linestyle='--', label='Truth')

    ax_dg.set_xlabel('Wavelength (nm)')
    ax_dg.set_ylabel(r'$a_{\rm dg} (400)$')

    # ############################
    # a_ph
    ax_ph = plt.subplot(gs[1])
    # Plot with error
    ax_ph.errorbar(wv_mins, aph, yerr=[aph_16, aph_84], fmt='o', 
                   color='k', label='Retrieved')
    # Horizonal line at true
    ax_ph.axhline(aph_true, color='b', linestyle='--', label='Truth')

    ax_ph.set_xlabel('Wavelength (nm)')
    ax_ph.set_ylabel(r'$a_{\rm ph} (440)$')

    for ax in [ax_dg, ax_ph]:
        ax.set_ylim(0., None)
        plotting.set_fontsize(ax, 15.)
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
