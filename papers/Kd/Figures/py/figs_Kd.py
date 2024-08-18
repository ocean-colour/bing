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
from ocpy.water import absorption as water_abs


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import anly_utils

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

def fig_Hooker2013(outfile='fig_Hooker2013.png'):

    # Load
    ds = loisel23.load_ds(4,0)

    # Unpack
    l23_wave = ds.Lambda.data
    l23_Rrs = ds.Rrs.data
    all_a = ds.a.data
    all_b = ds.b.data
    all_bnw = ds.bnw.data
    all_bbnw = ds.bbnw.data
    all_bb = ds.bb.data
    all_adg = ds.ag.data + ds.ad.data
    all_ad = ds.ad.data
    all_ag = ds.ag.data
    all_aph = ds.aph.data
    all_anw = ds.anw.data
    #
    all_bw = all_b - all_bnw
    #
    all_bbd = ds.bbd.data
    all_bbph = ds.bbph.data
    l23_aw = all_a[0] - all_anw[0]
    l23_bbw = all_bb[0] - all_bbnw[0]

    # Profile too
    ds_profile = loisel23.load_ds(4,0, profile=True)
    all_Kd = ds_profile.KEd_z[1,:,:]

    # Setup
    i350 = np.argmin(np.abs(l23_wave-350.))
    i440 = np.argmin(np.abs(l23_wave-440.))
    i750 = np.argmin(np.abs(l23_wave-750.))  # Should have been 780

    Kd_350 = all_Kd[:,i350]
    Kd_750 = all_Kd[:,i750]
    acdom_440 = all_ag[:,i440]

    more_wave = np.arange(320., 781., 1.)
    aw = water_abs.a_water(more_wave)
    #
    i780 = np.argmin(np.abs(more_wave-780.))  

    # H0
    bb780 = 0.
    H0 = aw[i780] + 4.18 * (1 - 0.52 * np.exp(-10.8*aw[i780])) * bb780

    # H1
    H1 = 4.18 * l23_bbw[i350]

    # acdom
    Sexp = 0.015
    rexp = np.exp(-Sexp*120)

    # Hooker+2013
    def acdom_hooker2013(kd_ratio):
        return 0.293 * kd_ratio - 0.015

    Kd_ratio = Kd_350/Kd_750

    # Figure
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    #
    ax.plot(Kd_350/Kd_750, acdom_440, 'o', ms=1, label='L23')
    # Label
    ax.set_xlabel(r'$K_d(350)/K_d(750)$')
    ax.set_ylabel(r'$a_{\rm cdom}(440) \, [\rm m^{-1}]$')
    # Hooker+2013
    x = np.linspace(Kd_ratio.min(), Kd_ratio.max(), 1000) 
    ax.plot(x, acdom_hooker2013(x), 'r-', label='Hooker+2013')
    # Derivation
    ax.plot(x, H0*rexp*x - H1*rexp, 'k-', label='Theory')
    #
    ax.legend(fontsize=15.)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plotting.set_fontsize(ax, 17.)
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

    if flg == 2:
        fig_Hooker2013()

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
