""" Figs for Kd Analyses """
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
    #ax.set_xlabel(r'Lee+2002 $K_d(a,b_b)$ ordinate')
    ax.set_xlabel(r'$a(\lambda)  + 4.18 \, [1-0.52 \, \exp(-10.8 a)] \, b_b(\lambda)$')
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



def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    if flg == 1:
        fig_Kd()


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
