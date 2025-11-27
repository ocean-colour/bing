"""
Module for fitting matchup data with GIOP model.

This module processes the entire set of Argo BGC matched profiles
using the GIOP (Generalized IOP) algorithm with Lee backscattering model.
It performs both least-squares and MCMC fitting to extract IOPs with uncertainties.
"""

import os
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
mpl.rcParams['font.family'] = 'stixgeneral'

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import corner
import pandas

from ocpy.water import absorption
from ocpy.water import scattering as w_scattering
from ocpy.utils import plotting
from ocpy.pace import io as pace_io

from bing import evaluate
from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf
from bing.fitting import chisq_fit

# Locals
from grab_pace_granules import closest_Rrs

from IPython import embed


def fit_giop_single(items):
    """
    Fit a single spectrum with GIOP model.

    Parameters:
    -----------
    items : tuple
        (iwave, ispec, isig, Chl_initial) containing wavelengths, Rrs,
        uncertainties, and initial chlorophyll estimate

    Returns:
    --------
    tuple : (models_giop, chains_giop, ans_giop, stats_giop)
        Fitted models, MCMC chains, least-squares solution, and statistics
        Returns (None, None, None, None) if fit fails
    """
    iwave, ispec, isig, Chl_initial = items

    # Init GIOP models
    p_giop = standard.giop()
    models_giop = model_utils.init(p_giop.model_names, iwave)
    bing_priors.set_standard_priors(models_giop, p_giop)

    # Set Chl for phytoplankton absorption
    models_giop[0].set_aph(Chl_initial)

    # Compute Y for Lee backscattering model
    i440 = np.argmin(np.abs(iwave - 440.))
    i555 = np.argmin(np.abs(iwave - 555.))
    models_giop[1].compute_Y(ispec[i440], ispec[i555])

    # Setup bounds for least-squares
    low_bounds, high_bounds = [], []
    low_bounds += [item['pmin'] for item in p_giop.apriors]
    low_bounds += [item['pmin'] for item in p_giop.bpriors]
    high_bounds += [item['pmax'] for item in p_giop.apriors]
    high_bounds += [item['pmax'] for item in p_giop.bpriors]
    bounds = (np.array(low_bounds), np.array(high_bounds))

    # Initial guess for GIOP (adg, aph, bbp)
    p0 = [-1, -1, -1]
    fit_items = [(ispec, isig**2, p0, 0)]

    try:
        ans_giop, cov_giop, idx = chisq_fit.fit(fit_items[0], models_giop, bounds=bounds)
    except RuntimeError:
        print("GIOP least-squares fit failed")
        return None, None, None, None

    # Update Chl based on fit results (aph parameter)
    Chl_fitted = 10**ans_giop[1] / 0.05582
    models_giop[0].set_aph(Chl_fitted)

    # Initialize MCMC
    pdict_giop = bing_inf.init_mcmc(models_giop, nsteps=p_giop.nsteps, nburn=p_giop.nburn)
    pdict_giop['Chl'] = np.array([Chl_fitted])
    pdict_giop['Y'] = np.array([models_giop[1].Y])

    # MCMC fitting
    print("----- Fitting GIOP with MCMC -----")
    p0_giop = ans_giop.tolist()
    fit_items = [(ispec, isig**2, p0_giop, 0)]

    try:
        chains_giop, idx = bing_inf.fit_one(
            fit_items[0], models=models_giop, pdict=pdict_giop, chains_only=True)
        stats_giop = evaluate.calc_stats(chains_giop)
    except Exception as e:
        print(f"GIOP MCMC fit failed: {e}")
        return models_giop, None, ans_giop, None

    return models_giop, chains_giop, ans_giop, stats_giop


def fit_giop_matchup(imatched: pandas.Series, outfile: str,
                     Chl_source: str = 'expbricaud',
                     debug: bool = False,
                     nclosest: int = 1,
                     n_cores: int = 10):
    """
    Fit GIOP model to matched Argo BGC profile data.

    This function loads PACE Rrs data at Argo float locations and fits
    the GIOP model (GIOP absorption + Lee backscattering) using both
    least-squares and MCMC methods.

    Parameters:
    -----------
    imatched : pandas.Series
        Series containing matched profile metadata including:
        - cruise, profile: Argo float identifiers
        - lat, lon: Float location
        - time: Observation time
        - closest_file: PACE granule filename
        - Chl (optional): Pre-fitted chlorophyll from ExpBricaud
    outfile : str
        Path to save output .npz file
    Chl_source : str
        Source for initial chlorophyll estimate:
        - 'expbricaud': Use pre-fitted Chl from ExpBricaud (requires 'aph' column)
        - 'oc4': Use OC4 band ratio algorithm
        - 'fixed': Use fixed value of 0.1 mg/m^3
    debug : bool
        Enable debugging mode
    nclosest : int
        Number of closest PACE pixels to fit
    n_cores : int
        Number of parallel processes for fitting

    Outputs:
    --------
    Saves .npz file with:
    - chains_giop: MCMC chains (nsamples, nwalkers, nparams)
    - LM_giop: Least-squares solutions (nclosest, nparams)
    - wave: Wavelength arrays
    - Rrs: Rrs spectra
    - Rrs_sig: Uncertainties
    - med, p14, p86: MCMC statistics
    - Y: Lee backscatter parameter
    - bbp_700, bbp_442, bbp_s: Derived backscatter products
    - model_names: ['GIOP', 'Lee']

    Also generates a .png plot of the fit for the closest pixel.
    """
    out_dict = {}

    # Load PACE file
    gfile = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP',
                         imatched.closest_file)
    print(f"----- Loading {gfile} -----")
    xds, flags = pace_io.load_oci_l2(gfile)

    # Find closest Rrs
    d_min, dmin_ij = closest_Rrs(xds, (imatched.lat, imatched.lon),
                                  nclosest=nclosest)
    if d_min is None:
        print("No good data found")
        out_dict['LM_giop'] = np.array([-999] * 3)
        np.savez(outfile, **out_dict)
        return

    if debug:
        embed(header='fit_giop.py:142')

    # Parse wavelength range
    gd_wave = (xds.wavelength.data >= 400.) & (xds.wavelength.data <= 700.)

    # Get initial Chl estimate for each pixel
    items = []
    for ss in range(nclosest):
        ix, iy = dmin_ij[0][ss], dmin_ij[1][ss]
        iwave = xds.wavelength.data[gd_wave]
        ispec = xds.Rrs.data[ix, iy, gd_wave]
        isig = xds.Rrs_unc.data[ix, iy, gd_wave]

        # Determine initial Chl
        if Chl_source == 'expbricaud':
            if 'aph' in imatched.index and not np.isnan(imatched.aph):
                Chl_init = imatched.aph
            else:
                print(f"Warning: aph not available, using fixed Chl=0.1")
                Chl_init = 0.1
        elif Chl_source == 'oc4':
            # Use OC4 band ratio
            from ocpy.chl import oc4
            i443 = np.argmin(np.abs(iwave - 443.))
            i490 = np.argmin(np.abs(iwave - 490.))
            i510 = np.argmin(np.abs(iwave - 510.))
            i555 = np.argmin(np.abs(iwave - 555.))
            ratio = np.max([ispec[i443], ispec[i490], ispec[i510]]) / ispec[i555]
            Chl_init = oc4(ratio)
        else:  # fixed
            Chl_init = 0.1

        items.append((iwave, ispec, isig, Chl_init))

    # Parallel fitting
    map_fn = partial(fit_giop_single)

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = nclosest // n_cores if nclosest // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items, chunksize=chunksize),
                           total=nclosest, desc="GIOP fitting"))

    # Collect results
    all_ans_giop, all_stats_giop = [], []
    all_lon, all_lat, all_idx, all_dist = [], [], [], []
    all_spec, all_Y = [], []
    ok_ss = []

    for ss, aa in enumerate(answers):
        if aa[2] is not None:  # Check if fit succeeded
            ok_ss.append(ss)
            all_ans_giop.append(aa[2])
            all_stats_giop.append(aa[3])
            all_Y.append(aa[0][1].Y if aa[0] is not None else np.nan)

            # Coordinates
            ix, iy = dmin_ij[0][ss], dmin_ij[1][ss]
            all_idx.append(np.array([ix, iy]))
            all_lon.append(xds.longitude.data[ix, iy])
            all_lat.append(xds.latitude.data[ix, iy])
            all_dist.append(d_min[ss])
            all_spec.append(items[ss])

    # Check if all fits failed
    if len(all_ans_giop) == 0:
        print("All GIOP fits failed: saving -999")
        out_dict['LM_giop'] = np.array([-999] * 3)
        out_dict['wave'] = items[0][0]
        out_dict['Rrs'] = items[0][1]
        out_dict['Rrs_sig'] = items[0][2]
        out_dict['Rrs_idx'] = np.array([dmin_ij[0][0], dmin_ij[1][0]])
        np.savez(outfile, **out_dict)
        return

    # Get closest successful fit for detailed output
    models_giop, chains_giop, ans_giop, stats_giop = answers[ok_ss[0]]
    ispec, isig = all_spec[0][1], all_spec[0][2]

    # Save results
    if chains_giop is not None:
        out_dict['chains_giop'] = chains_giop
    out_dict['LM_giop'] = np.stack(all_ans_giop)
    out_dict['wave'] = np.stack([it[0] for it in all_spec])
    out_dict['Rrs'] = np.stack([it[1] for it in all_spec])
    out_dict['Rrs_sig'] = np.stack([it[2] for it in all_spec])
    out_dict['Rrs_idx'] = np.stack(all_idx)
    out_dict['lon'] = np.array(all_lon)
    out_dict['lat'] = np.array(all_lat)
    out_dict['dist'] = np.array(all_dist)
    out_dict['Y'] = np.array(all_Y)

    if all_stats_giop[0] is not None:
        out_dict['med'] = np.stack([stats['med'] for stats in all_stats_giop])
        out_dict['p14'] = np.stack([stats['p14'] for stats in all_stats_giop])
        out_dict['p86'] = np.stack([stats['p86'] for stats in all_stats_giop])

    out_dict['model_names'] = [model.name for model in models_giop]

    # Calculate derived products (bbp at 700nm, 442nm, and spectral slope)
    # bbp values are at 600nm pivot, need to convert
    bbp_700_vals = []
    bbp_442_vals = []
    bbp_s_vals = []

    for ans, Y in zip(all_ans_giop, all_Y):
        bbp_600 = 10**ans[2]  # Last parameter is log10(bbp) at 600nm
        # Lee model: bbp(λ) = bbp_600 * (600/λ)^Y
        bbp_700 = bbp_600 * (600.0 / 700.0)**Y
        bbp_442 = bbp_600 * (600.0 / 442.0)**Y
        bbp_700_vals.append(bbp_700)
        bbp_442_vals.append(bbp_442)
        bbp_s_vals.append(Y)

    out_dict['bbp_700'] = np.array(bbp_700_vals)
    out_dict['bbp_442'] = np.array(bbp_442_vals)
    out_dict['bbp_s'] = np.array(bbp_s_vals)

    np.savez(outfile, **out_dict)
    print(f"Saved: {outfile}")

    # Plot the closest fit
    if chains_giop is not None:
        print("----- Plotting GIOP fit -----")
        title = f'GIOP: Float={imatched.cruise}-{imatched.profile}, ' + \
                f'lat={imatched.lat:.1f}, lon={imatched.lon:.1f}, ' + \
                f'time={imatched.time[:19]}, {imatched.closest_id[12:-9]}, ' + \
                f'dist={all_dist[0]:.1f} km, Y={all_Y[0]:.3f}'
        Rrs_obs = dict(wave=models_giop[0].wave, spec=ispec, var=isig**2)
        plotfile = outfile.replace('.npz', '_GIOP.png')
        plot_giop_fit(models_giop, chains_giop, Rrs_obs, title,
                     stats=stats_giop, Y=all_Y[0], outfile=plotfile)


def plot_giop_fit(models, chains, Rrs_obs, title: str,
                  stats: dict = None, Y: float = None,
                  outfile: str = None,
                  perc: tuple = (14, 86),
                  show_Rsig: bool = True):
    """
    Plot GIOP fitting results.

    Similar to fitting.plot_fit but adapted for GIOP model specifics.
    Shows adg, aph, bbp components and their uncertainties.
    """
    # Calculate statistics if not provided
    if stats is None:
        stats = evaluate.calc_stats(chains, perc=perc)

    # Reconstruct spectra
    a_mean, bb_mean, a_5, a_95, bb_5, bb_95, model_Rrs, sigRs = \
        evaluate.reconstruct_from_chains(models, chains, perc=perc)

    wave = models[0].wave

    # Water components
    a_w = absorption.a_water(wave, data='IOCCG')
    bb_w = w_scattering.bbw_from_l23(wave)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.3, 1, 1], hspace=0.02)

    ax_res = plt.subplot(gs[0])
    ax_R = plt.subplot(gs[2], sharex=ax_res)
    ax_anw = plt.subplot(gs[3])
    ax_bb = plt.subplot(gs[4])
    ax_c = plt.subplot(gs[5])

    ax_res.set_xticklabels([])

    # Plot non-water absorption
    ax_anw.plot(wave, a_mean - a_w, 'b-', label='Retrieval')
    ax_anw.fill_between(wave, a_5 - a_w, a_95 - a_w,
                        color='b', alpha=0.5, label='Uncertainty')
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    ax_anw.set_yscale('log')

    # GIOP parameters (adg, aph)
    model = models[0]
    ypos = 0.1
    param_labels = ['adg', 'aph']
    for ss in range(model.nparam):
        lsig = stats['med'][ss] - stats[f'p{perc[0]:02d}'][ss]
        hsig = stats[f'p{perc[1]:02d}'][ss] - stats['med'][ss]
        ax_anw.text(0.05, ypos,
                   f'{param_labels[ss]} = {stats["med"][ss]:.3f}' +
                   r'$^{+' + f'{hsig:.3f}' + r'}_{-' + f'{lsig:.3f}' + r'}$',
                   transform=ax_anw.transAxes, fontsize=13.)
        ypos += 0.11

    # Plot non-water backscattering
    ax_bb.plot(wave, bb_mean - bb_w, 'g-', label='Retrieval')
    ax_bb.fill_between(wave, bb_5 - bb_w, bb_95 - bb_w,
                       color='g', alpha=0.5, label='Uncertainty')
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')

    # Lee model parameters (bbp at 600nm, Y is derived)
    model = models[1]
    ss = models[0].nparam  # bbp parameter index
    lsig = stats['med'][ss] - stats[f'p{perc[0]:02d}'][ss]
    hsig = stats[f'p{perc[1]:02d}'][ss] - stats['med'][ss]
    ax_bb.text(0.05, 0.1,
              f'bbp@600nm = {stats["med"][ss]:.3f}' +
              r'$^{+' + f'{hsig:.3f}' + r'}_{-' + f'{lsig:.3f}' + r'}$',
              transform=ax_bb.transAxes, fontsize=13.)
    if Y is not None:
        ax_bb.text(0.05, 0.21, f'Y = {Y:.3f}',
                  transform=ax_bb.transAxes, fontsize=13.)

    # Calculate chi-squared
    Rsig = np.sqrt(Rrs_obs['var'])
    f = interp1d(wave, model_Rrs)
    mod_R = f(Rrs_obs['wave'])
    chi2 = np.sum((Rrs_obs['spec'] - mod_R)**2 / Rsig**2)
    nparam = models[0].nparam + models[1].nparam
    red_chi2 = chi2 / (Rsig.size - nparam)

    # Plot Rrs
    if show_Rsig:
        ax_R.errorbar(Rrs_obs['wave'], Rrs_obs['spec'],
                     yerr=Rsig, color='gray', fmt='o', capsize=3,
                     label='Obs', zorder=1)
    ax_R.plot(Rrs_obs['wave'], Rrs_obs['spec'], 'k+', zorder=5)
    ax_R.plot(wave, model_Rrs, 'r-', label='Fit', zorder=10)
    ax_R.fill_between(wave, model_Rrs - sigRs, model_Rrs + sigRs,
                     color='r', alpha=0.5, zorder=10)
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [{\rm sr}^{-1}]$')
    ax_R.text(0.05, 0.1, r'$\chi^2_\nu = ' + f'{red_chi2:0.2f}' + r'$',
             fontsize=15., transform=ax_R.transAxes)

    # Plot residuals
    residuals = (Rrs_obs['spec'] - mod_R) / Rsig
    for y in [-2, 0, 2]:
        ls = ':' if y == 0 else '--'
        ax_res.axhline(y=y, color='gray', linestyle=ls, alpha=0.5)
    ax_res.scatter(Rrs_obs['wave'], residuals, color='k', s=1.)
    ax_res.set_ylabel('Residuals\n(σ)', fontsize=12)
    ymx = 1.2 * np.max(np.abs(residuals))
    ax_res.set_ylim([-ymx, ymx])
    ax_res.grid(True, alpha=0.3)
    ax_res.minorticks_on()

    fontsize = 15.
    plotting.set_fontsize(ax_res, fontsize)

    # Format axes
    axes = [ax_anw, ax_bb, ax_R]
    for ax in axes:
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    # Mini corner plot
    mini_corner_giop(models, chains, ['adg', 'bbp'], outfile='tmpc_giop.png')
    img = mpimg.imread('tmpc_giop.png')
    ax_c.imshow(img)
    ax_c.axis('off')

    # Title
    fig.suptitle(title, fontsize=14, y=0.99)

    # Save
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")
    else:
        plt.show()


def mini_corner_giop(models, chains, show_params: list, outfile: str = None):
    """Create mini corner plot for GIOP parameters."""
    # Burn/thin chains
    coeff = evaluate.thin_burn_chains(chains)

    # Select parameters to show
    keep = np.array([False] * coeff.shape[1])
    cnt = 0
    clbls = []
    for model in models:
        for param in model.pnames:
            if param in show_params:
                keep[cnt] = True
                clbls.append(param)
            cnt += 1

    coeff = coeff[:, keep]

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize': 17},
        color='k',
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


def slurp_giop_fits(matched: pandas.DataFrame,
                    outfile_pattern: str = None,
                    debug: bool = False):
    """
    Extract GIOP fit results from saved files and add to matchup table.

    Parameters:
    -----------
    matched : pandas.DataFrame
        DataFrame with matched profiles
    outfile_pattern : str, optional
        Pattern for output files. If None, uses set_giop_outfile()
    debug : bool
        Debug mode

    Returns:
    --------
    pandas.DataFrame
        Updated DataFrame with GIOP results added as columns:
        - GIOP_adg, GIOP_aph, GIOP_bbp_600
        - GIOP_bbp_700, GIOP_bbp_442, GIOP_bbp_s
        - GIOP_Y
    """
    giop_adg_vals = []
    giop_aph_vals = []
    giop_bbp_600_vals = []
    giop_bbp_700_vals = []
    giop_bbp_442_vals = []
    giop_bbp_s_vals = []
    giop_Y_vals = []

    for ss in range(len(matched)):
        imatched = matched.iloc[ss]

        if outfile_pattern is None:
            outfile = set_giop_outfile(imatched)
        else:
            outfile = outfile_pattern.format(
                cruise=imatched.cruise, profile=imatched.profile)

        print(f'Processing {ss+1}/{len(matched)}: {os.path.basename(outfile)}...')

        # Load results
        if not os.path.exists(outfile):
            print(f"Warning: Missing {outfile}")
            giop_adg_vals.append(np.nan)
            giop_aph_vals.append(np.nan)
            giop_bbp_600_vals.append(np.nan)
            giop_bbp_700_vals.append(np.nan)
            giop_bbp_442_vals.append(np.nan)
            giop_bbp_s_vals.append(np.nan)
            giop_Y_vals.append(np.nan)
            continue

        d = np.load(outfile)

        if 'LM_giop' not in d or d['LM_giop'][0, 0] == -999:
            print(f"Skipping {outfile} (no valid fit)")
            giop_adg_vals.append(np.nan)
            giop_aph_vals.append(np.nan)
            giop_bbp_600_vals.append(np.nan)
            giop_bbp_700_vals.append(np.nan)
            giop_bbp_442_vals.append(np.nan)
            giop_bbp_s_vals.append(np.nan)
            giop_Y_vals.append(np.nan)
            continue

        # Extract GIOP parameters (use closest pixel, index 0)
        # LM_giop has shape (nclosest, nparams) where nparams=3 for GIOP
        giop_adg_vals.append(10**d['LM_giop'][0, 0])  # log10(adg)
        giop_aph_vals.append(10**d['LM_giop'][0, 1])  # log10(aph)
        giop_bbp_600_vals.append(10**d['LM_giop'][0, 2])  # log10(bbp@600)

        # Derived products
        if 'bbp_700' in d:
            giop_bbp_700_vals.append(d['bbp_700'][0])
            giop_bbp_442_vals.append(d['bbp_442'][0])
            giop_bbp_s_vals.append(d['bbp_s'][0])
            giop_Y_vals.append(d['Y'][0])
        else:
            giop_bbp_700_vals.append(np.nan)
            giop_bbp_442_vals.append(np.nan)
            giop_bbp_s_vals.append(np.nan)
            giop_Y_vals.append(np.nan)

        if debug:
            break

    # Add to DataFrame
    matched['GIOP_adg'] = np.array(giop_adg_vals)
    matched['GIOP_aph'] = np.array(giop_aph_vals)
    matched['GIOP_bbp_600'] = np.array(giop_bbp_600_vals)
    matched['GIOP_bbp_700'] = np.array(giop_bbp_700_vals)
    matched['GIOP_bbp_442'] = np.array(giop_bbp_442_vals)
    matched['GIOP_bbp_s'] = np.array(giop_bbp_s_vals)
    matched['GIOP_Y'] = np.array(giop_Y_vals)

    return matched


def set_giop_outfile(imatched: pandas.Series):
    """Generate output filename for GIOP fits."""
    outfile = os.path.join(os.getenv('OS_COLOR'), 'Biomass', 'Fits_GIOP',
                          f'Argo_{imatched.cruise}_{imatched.profile:03d}_GIOP_fits.npz')
    return outfile


# Command line interface
if __name__ == '__main__':
    import sys

    # Configuration
    test = False
    fit_all = False
    slurp = False

    # Parse simple command line args
    if len(sys.argv) > 1:
        if 'test' in sys.argv:
            test = True
        if 'fit' in sys.argv:
            fit_all = True
        if 'slurp' in sys.argv:
            slurp = True
    else:
        # Default: run test
        test = True

    # Load matched profiles
    match_file = 'matched_argo_bgc_profiles_bbp.csv'
    matched = pandas.read_csv(match_file)

    # Ensure output directory exists
    fits_dir = os.path.join(os.getenv('OS_COLOR'), 'Biomass', 'Fits_GIOP')
    os.makedirs(fits_dir, exist_ok=True)

    if test:
        print("=" * 60)
        print("RUNNING TEST MODE: Fitting single profile with GIOP")
        print("=" * 60)

        # Test on a single profile
        idx = 0
        imatched = matched.iloc[idx]
        print(f"\nFitting profile: {imatched.cruise}-{imatched.profile}")
        print(f"Location: lat={imatched.lat:.2f}, lon={imatched.lon:.2f}")

        outfile = set_giop_outfile(imatched)
        fit_giop_matchup(imatched, outfile, Chl_source='expbricaud',
                        nclosest=5, n_cores=5)

        print("\nTest complete! Check output file:")
        print(f"  {outfile}")

    if fit_all:
        print("=" * 60)
        print("FITTING ALL PROFILES WITH GIOP")
        print(f"Total profiles: {len(matched)}")
        print("=" * 60)

        clobber = False
        failed = []

        for ss in range(len(matched)):
            imatched = matched.iloc[ss]
            print("\n" + "*" * 60)
            print(f"Processing {ss+1}/{len(matched)}: {imatched.cruise}-{imatched.profile}")
            print("*" * 60)

            # Check if already processed
            outfile = set_giop_outfile(imatched)
            if os.path.exists(outfile) and not clobber:
                print(f"Already processed, skipping...")
                continue

            # Fit
            try:
                fit_giop_matchup(imatched, outfile, Chl_source='expbricaud',
                               nclosest=10, n_cores=10)
            except Exception as e:
                print(f"ERROR fitting {imatched.cruise}-{imatched.profile}: {e}")
                failed.append((ss, imatched.cruise, imatched.profile, str(e)))

        print("\n" + "=" * 60)
        print("FITTING COMPLETE")
        if failed:
            print(f"Failed profiles: {len(failed)}")
            for ss, cruise, profile, err in failed:
                print(f"  {ss}: {cruise}-{profile}: {err}")
        print("=" * 60)

    if slurp:
        print("=" * 60)
        print("EXTRACTING GIOP RESULTS FROM FITS")
        print("=" * 60)

        matched = slurp_giop_fits(matched)

        # Save updated matched file
        output_file = match_file.replace('.csv', '_with_GIOP.csv')
        matched.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        print(f"Total profiles: {len(matched)}")
        print(f"Successful GIOP fits: {(~matched.GIOP_bbp_700.isna()).sum()}")
