""" Figures for PACE-Argo BGC biomass validation """
import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

from scipy.stats import pearsonr

from ocpy.utils import plotting


def gen_cb(img, lbl, csz=17.):
    """Generate a colorbar for the given image."""
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_giop_vs_bnw_colored(outfile='fig_giop_vs_bnw_colored.png',
                            csv_file=None,
                            color_by='beta'):
    """
    Generate a figure comparing GIOP bbp(700) vs BING Bnw, colored by another variable.

    Parameters:
        outfile (str): The filename of the output figure
        csv_file (str): Path to the matched Argo BGC profiles CSV file.
        color_by (str): Column name to use for coloring points.
            Options: 'beta', 'closest_dist_km', 'lat', etc.

    """
    # Load data
    if csv_file is None:
        csv_file = os.path.join(
            os.path.dirname(__file__),
            '../../Analysis/matched_argo_bgc_profiles_bbp.csv')

    df = pd.read_csv(csv_file)

    # Extract columns
    giop_bbp_700 = df['GIOP_bbp_700'].values
    bnw = df['Bnw'].values
    color_var = df[color_by].values

    # Filter out invalid values
    valid = (giop_bbp_700 > 0) & (bnw > 0) & (giop_bbp_700 != -32767)
    giop_bbp_700 = giop_bbp_700[valid]
    bnw = bnw[valid]
    color_var = color_var[valid]

    # Figure
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Scatter plot with color
    sc = ax.scatter(giop_bbp_700, bnw, s=25, c=color_var,
                    cmap='viridis', alpha=0.7, edgecolors='none')

    # Colorbar
    cbar_label = {
        'beta': r'Spectral slope $\beta$',
        'closest_dist_km': 'Match distance [km]',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'GIOP_Y': 'GIOP Y',
    }.get(color_by, color_by)
    gen_cb(sc, cbar_label)

    # 1:1 line
    lims = [min(giop_bbp_700.min(), bnw.min()) * 0.5,
            max(giop_bbp_700.max(), bnw.max()) * 2]
    ax.plot(lims, lims, 'k--', lw=1.5, label='1:1', zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Offset line
    med_off = np.median(bnw/giop_bbp_700)
    ax.plot(lims, med_off*np.array(lims), 'k:', lw=1.5, 
            label=f'Offset by {med_off:0.2f}', zorder=0)

    # Labels
    ax.set_xlabel(r'GIOP $b_{bp}(700)$ [m$^{-1}$]')
    ax.set_ylabel(r'BING $B_{nw}$ [m$^{-1}$]')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)

    plotting.set_fontsize(ax, 15.)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    return fig, ax


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Spectra -- blue/red with water
    if flg == 1:
        fig_giop_vs_bnw_colored()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # flg = 1 :: Figure 1; Spectra of water and non-water
        # flg = 2 :: Figure 2; k=2,5 fits
        # flg = 3 :: Figure 6; BIC
        # flg = 37 :: Figure 7 Low Chl, 4-panel :: fig_four_panel_fit
        # flg = 38 :: Fig 9 High Chl, 4-panel :: fig_four_panel_fit
        
        # flg = 10 :: Supp 1; fig_u

        # flg = 12 :: Satellite noise

        # flg = 14 :: a_ph(440), bbp(440) scatter fig_aph_and_bbnw
            # PACE and GIOP

        # flg = 17 :: Every, Every degenerate fit; arbitrary IOP model

        # New PACE figures
        # flg = 34 :: Rrs, anw, bbnw on high Chla
        # flg = 39 :: Multi-model, 4-panel

        # flg = 40 :: k=5, ExpBricaud, Pow; fig_bing_figs

    else:
        flg = sys.argv[1]

    main(flg)