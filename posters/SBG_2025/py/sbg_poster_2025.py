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
sys.path.append(os.path.abspath("../../papers/bing_2.0/Analysis/py"))
import anly_utils_20
import param as param20

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


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # BIC/AIC for MODIS+L23
    if flg == 1:

        fig_bic_sbg_pace()


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