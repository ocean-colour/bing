# Fit the full Loisel et al. (2023) synthetic dataset with BING.
#
# Goal: examine biases in the bbp (and anw) retrievals against the
# known L23 truth.  We begin with the *elastic* L23 spectra generated
# with the *standard* (constant) Gordon coefficients and add no noise.
#
# This module is modeled after fit_with_argo.py but targets the L23
# synthetic dataset, reusing the bing.fitting.l23 machinery for loading,
# prepping and (batch) MCMC fitting.

import os
import glob
import json

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np

import pandas

from bing.parameters import standard
from bing.parameters import p_ntuple
from bing import io as bing_io
from bing.preproc import convert_to_satwave
from bing.rt import defs as rt_defs
from bing.models import utils as model_utils

from bing.fitting import l23
from bing.fitting import inference as bing_inf
from bing.priors import priors as bing_priors

from ocpy.hydrolight import loisel23

# Locals
import fitting
import lowest_bbp

from IPython import embed

# Output directory for the L23 fits (NPZ + JSON + PNG per spectrum)
OUTDIR = os.path.join(os.getenv('OS_COLOR'), 'Biomass', 'L23_Fits')

# The three indices used in debug mode (a low/medium/high water-type spread)
DEBUG_IDX = [3003, 170, 180]

# Total number of spectra in the L23 dataset
N_L23 = 3320

# Fit/save/plot the dataset in chunks of this many spectra so the MCMC
# chains for the whole dataset never have to live in memory at once.
BATCH_SIZE = 500


def get_l23_params(satellite: str = 'PACE', nsteps: int = 40000,
                   nburn: int = 1000):
    """Build the BING parameter tuple for the baseline L23 fits.

    The baseline uses the ExpBricaud + power-law model pair with the
    *elastic* Gordon forward model and the *standard* (constant) Gordon
    coefficients.  No inelastic terms (Raman / fluorescence) are included.
    Standard PACE noise is added to the synthetic spectra.

    Parameters
    ----------
    satellite : str, optional
        Wavelength grid / noise model to use ('PACE', 'MODIS', 'SeaWiFS',
        'SBG', or 'L23' for the native Hydrolight grid).  Default 'PACE'.
    nsteps : int, optional
        Number of MCMC steps.  Default 40000.
    nburn : int, optional
        Number of burn-in steps.  Default 1000.

    Returns
    -------
    namedtuple
        The BING parameter tuple consumed by the bing.fitting.l23 code.
    """
    # add_noise=True with the default scl_noise=satellite gives standard
    # PACE noise; variable_Gordon=False selects the standard constant
    # Gordon coefficients; the include_* flags stay off so the forward
    # model is purely elastic.
    p = standard.expb_pow(
        satellite=satellite,
        nsteps=nsteps,
        nburn=nburn,
        add_noise=True,
        scl_noise='PACE',
        variable_Gordon=False,
        include_Raman=False,
        include_Chl_fl=False)
    return p


def plot_l23_fit(p, models, chains, odict, obs_Rrs, varRrs, idx,
                 outfile: str, show_corner: bool = False):
    """Make the standard BING diagnostic figure for one L23 fit.

    Overlays the *true* non-water absorption and backscattering (from the
    L23 truth) as black lines so retrieval biases are visible by eye, and
    annotates the true bbp amplitude (at the 600 nm power-law pivot, the
    quantity the fitted Bnw parameter targets) plus the true Aph and Adg
    amplitudes (at 440 / 400 nm, matching the ExpBricaud parameters).

    Parameters
    ----------
    p : namedtuple
        BING parameter tuple (used to rebuild the radiative-transfer dict).
    models : list
        Two-element [absorption_model, backscattering_model] list.
    chains : np.ndarray
        MCMC chains, shape (nsteps, nwalkers, nparam).
    odict : dict
        L23 truth dictionary from load_one_l23 (has 'true_wave', 'anw',
        'bbnw', 'Chl').
    obs_Rrs : np.ndarray
        Observed (synthetic) Rrs on the model wavelength grid.
    varRrs : np.ndarray
        Variance of the observed Rrs on the model wavelength grid.
    idx : int
        L23 spectrum index (for the figure title).
    outfile : str
        Path to the output PNG.
    show_corner : bool, optional
        If True, include the mini corner plot panel.  Default False.

    Returns
    -------
    None
    """
    model_wave = models[0].wave

    # Interpolate the L23 truth onto the model wavelength grid so the
    # black "True" lines line up with the retrieval panels.
    true_anw = convert_to_satwave(odict['true_wave'], odict['anw'],
                                  model_wave)
    true_bbp = convert_to_satwave(odict['true_wave'], odict['bbnw'],
                                  model_wave)

    # True bbp amplitude at the 600 nm power-law pivot (matches Bnw).
    # bb_anno reports it in both linear and log10 (Bnw is fit in log10).
    true_Bnw = float(np.interp(600., odict['true_wave'], odict['bbnw']))
    bb_anno = (f'True bbp(600) = {true_Bnw:.4f}\n'
               f'True log Bnw = {np.log10(true_Bnw):.3f}')

    # True Aph / Adg amplitudes in the ExpBricaud parameterization:
    #   Adg = a_dg(400 nm), Aph = a_ph(440 nm), both fit in log10.
    true_Adg = float(np.interp(400., odict['true_wave'], odict['adg']))
    true_Aph = float(np.interp(440., odict['true_wave'], odict['aph']))
    anw_anno = (f'True log Adg = {np.log10(true_Adg):.3f}\n'
                f'True log Aph = {np.log10(true_Aph):.3f}')

    # Radiative-transfer config must match the one used to fit.
    rt_dict = rt_defs.rt_dict_from_p(p)

    # Observation dict expected by fitting.plot_fit.
    Rrs_obs = dict(wave=model_wave, spec=obs_Rrs, var=varRrs)

    title = (f'L23 idx={idx}, Chl={odict["Chl"]:.3f}, '
             f'elastic + standard Gordon, PACE noise')

    fitting.plot_fit(models, chains, Rrs_obs, title, rt_dict,
                     true_anw=true_anw, true_bbp=true_bbp,
                     bb_anno=bb_anno, anw_anno=anw_anno, show_Rsig=True,
                     show_corner=show_corner, outfile=outfile)


def fit_one_l23(idx: int, p=None, outdir: str = None,
                clobber: bool = False, plot: bool = True):
    """Fit a single L23 spectrum and save the results plus a figure.

    Parameters
    ----------
    idx : int
        L23 spectrum index (0-3319).
    p : namedtuple, optional
        BING parameter tuple.  If None, the baseline elastic/standard-Gordon
        configuration from get_l23_params() is used.
    outdir : str, optional
        Output directory.  Defaults to the module-level OUTDIR.
    clobber : bool, optional
        If False and the output NPZ already exists, the fit is skipped.
    plot : bool, optional
        If True, write the diagnostic PNG figure.  Default True.

    Returns
    -------
    str or None
        The output root (path without extension), or None if skipped.
    """
    if p is None:
        p = get_l23_params()
    if outdir is None:
        outdir = OUTDIR
    os.makedirs(outdir, exist_ok=True)

    outroot = os.path.join(outdir, f'L23_{idx:04d}')

    # Skip if already done
    if os.path.exists(outroot + '.npz') and not clobber:
        print(f"Already fit {outroot}.npz, skipping...")
        return outroot

    print("=" * 80)
    print(f"Fitting L23 idx={idx}...")
    print("=" * 80)

    # Full MCMC fit via the L23 wrapper
    chains, models, prep_dict, _, extras = l23.fit_one(p, idx)
    odict = prep_dict['odict']

    # Save the fit (NPZ + JSON) in the standard BING layout
    bing_io.save_fit(outroot, p, models, chains, prep_dict['p0'],
                     extras['obs_Rrs'], extras['varRrs'])

    # Figure with the true anw / bbp drawn as black lines
    if plot:
        plot_l23_fit(p, models, chains, odict, extras['obs_Rrs'],
                     extras['varRrs'], idx, outroot + '.png')

    return outroot


def _prep_spectrum(idx: int, models: list):
    """Generate one synthetic spectrum and its initial guess for fitting.

    Uses ``lowest_bbp.generate_pace_spectrum`` (elastic L23 Rrs on the PACE
    grid, with PACE noise) and builds the log10 initial guess from the true
    IOPs, mirroring the conversion in ``l23.prep_one_l23``.

    Parameters
    ----------
    idx : int
        L23 spectrum index.
    models : list
        Two-element [absorption_model, backscattering_model] list, with
        priors already attached (needed to know which params are log10).

    Returns
    -------
    dict
        Keys: 'Rrs', 'varRrs', 'p0' (log10 where appropriate), 'odict'
        (L23 truth), and 'Chl' (derived from the Aph initial guess).
    """
    spec = lowest_bbp.generate_pace_spectrum(idx=idx, seed=None,
                                             use_elastic=True)
    odict = spec['odict']

    # Initial guess from the true IOPs interpolated to the model grid
    l23_wave = odict['true_wave']
    model_anw = convert_to_satwave(l23_wave, odict['anw'], models[0].wave)
    model_bbnw = convert_to_satwave(l23_wave, odict['bbnw'], models[1].wave)
    p0_a = models[0].init_guess(model_anw)
    p0_b = models[1].init_guess(model_bbnw)
    p0 = np.concatenate((np.atleast_1d(p0_a), np.atleast_1d(p0_b)))

    # init_guess returns linear amplitudes; log-prior params must be log10
    # (Adg, Aph, Bnw).  Walk the priors in order and convert in place.
    cnt = 0
    for ss in [0, 1]:
        for prior in models[ss].priors.priors:
            if prior.flavor[0:3] == 'log':
                p0[cnt] = np.log10(p0[cnt])
            cnt += 1

    # Chl from the (now log10) Aph guess, matching the Bricaud convention
    Chl = 10**p0[2] / 0.05582

    return dict(Rrs=spec['Rrs'], varRrs=spec['sigRrs']**2,
                p0=p0, odict=odict, Chl=Chl)


def _save_plot_one(args):
    """Save one fit (NPZ/JSON) and write its diagnostic PNG.

    Worker function for parallel save/plot via ProcessPoolExecutor; takes a
    single packed-tuple argument so it can be mapped directly.

    Parameters
    ----------
    args : tuple
        ``(p_dict, models, chains, p0, Rrs, varRrs, odict, idx)`` for one
        spectrum (see fit_all_l23 for the meaning of each element).  ``p``
        is passed as a plain dict (``p._asdict()``) because the dynamic
        BING namedtuple is not picklable across processes; it is rebuilt
        here with ``p_ntuple.gen``.

    Returns
    -------
    int
        The L23 index that was written (for progress tracking).
    """
    p_dict, models, chains, p0, Rrs, varRrs, odict, idx = args
    p = p_ntuple.gen(**p_dict)
    outroot = os.path.join(OUTDIR, f'L23_{idx:04d}')
    bing_io.save_fit(outroot, p, models, chains, p0, Rrs, varRrs)
    plot_l23_fit(p, models, chains, odict, Rrs, varRrs, idx,
                 outroot + '.png')
    return idx


def fit_all_l23(debug: bool = True, satellite: str = 'PACE',
                n_cores: int = 10, clobber: bool = False,
                batch_size: int = BATCH_SIZE):
    """Fit the L23 dataset in batches and save fits + figures per batch.

    The spectra are generated with ``lowest_bbp.generate_pace_spectrum``
    (elastic L23 Rrs on the PACE grid + PACE noise).  To keep memory
    bounded on the full dataset, the indices are processed in chunks of
    ``batch_size``: each chunk is prepped, fit in parallel via
    ``bing_inf.fit_batch``, saved/plotted, and its chains are released
    before the next chunk starts.

    Parameters
    ----------
    debug : bool, optional
        If True, fit only the three DEBUG_IDX spectra.  If False, fit all
        N_L23 spectra.  Default True.
    satellite : str, optional
        Wavelength grid / noise model.  Default 'PACE'.
    n_cores : int, optional
        Number of CPU cores for the parallel MCMC.  Default 10.
    clobber : bool, optional
        If False, indices whose output NPZ already exists are skipped.
    batch_size : int, optional
        Number of spectra fit per chunk.  Default BATCH_SIZE (500).

    Returns
    -------
    None
    """
    p = get_l23_params(satellite=satellite)
    os.makedirs(OUTDIR, exist_ok=True)

    # Which indices to fit
    if debug:
        all_idx = list(DEBUG_IDX)
    else:
        all_idx = list(range(N_L23))

    # Drop indices already on disk (unless clobbering)
    if not clobber:
        todo = []
        for idx in all_idx:
            outroot = os.path.join(OUTDIR, f'L23_{idx:04d}')
            if os.path.exists(outroot + '.npz'):
                print(f"Already fit idx={idx}, skipping...")
            else:
                todo.append(idx)
        all_idx = todo

    if len(all_idx) == 0:
        print("Nothing to fit.")
        return

    # Init models once (the PACE grid is fixed by wv_min/wv_max), attach
    # priors, and build the shared MCMC config.  These are reused for every
    # batch; only the per-spectrum data and Chl change.
    spec0 = lowest_bbp.generate_pace_spectrum(idx=all_idx[0], seed=None,
                                              use_elastic=True)
    models = model_utils.init(p.model_names, spec0['pace_wave'])
    bing_priors.set_standard_priors(models, p)

    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)
    pdict['Y'] = None  # Pow backscattering does not use Y
    # Chl is looked up by L23 idx inside fit_one, so index by idx (not by
    # batch position) — this was the source of the IndexError.
    pdict['Chl'] = np.zeros(N_L23)

    rt_dict = rt_defs.rt_dict_from_p(p)

    # Process in batches to bound memory
    n_done = 0
    n_batches = (len(all_idx) + batch_size - 1) // batch_size
    for b0 in range(0, len(all_idx), batch_size):
        batch_idx = all_idx[b0:b0 + batch_size]
        print("=" * 80)
        print(f"Batch {b0 // batch_size + 1}/{n_batches}: "
              f"{len(batch_idx)} spectra")
        print("=" * 80)

        # Prep this batch and stage the per-idx Chl values
        Rrs, varRrs, params, odicts = [], [], [], []
        for idx in batch_idx:
            d = _prep_spectrum(idx, models)
            Rrs.append(d['Rrs'])
            varRrs.append(d['varRrs'])
            params.append(d['p0'])
            odicts.append(d['odict'])
            pdict['Chl'][idx] = d['Chl']

        # Build the fit items: (Rrs, varRrs, p0, idx)
        items = [(Rrs[ss], varRrs[ss], params[ss], idx)
                 for ss, idx in enumerate(batch_idx)]

        # Parallel MCMC for this batch
        print(f"Fitting {len(items)} spectra on {n_cores} cores...")
        all_chains, sub_idx = bing_inf.fit_batch(
            models, pdict, items, rt_dict, n_cores=n_cores)
        assert np.all(np.array(batch_idx) == sub_idx)

        # Save + plot each spectrum in this batch, in parallel.  save_fit
        # reconstructs IOPs from the chains (CPU-heavy) and plotting is
        # slow, so fan the per-spectrum work out across cores.
        print(f"Saving + plotting {len(batch_idx)} fits on "
              f"{n_cores} cores...")
        # p is sent as a plain dict (the BING namedtuple is not picklable)
        p_dict = dict(p._asdict())
        save_items = [
            (p_dict, models, all_chains[ss], params[ss], Rrs[ss],
             varRrs[ss], odicts[ss], idx)
            for ss, idx in enumerate(batch_idx)]
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = max(1, len(save_items) // n_cores)
            list(tqdm(executor.map(_save_plot_one, save_items,
                                   chunksize=chunksize),
                      total=len(save_items)))

        # Release this batch's chains before the next one
        n_done += len(batch_idx)
        del all_chains

    print(f"Done! Fit {n_done} L23 spectra into {OUTDIR}")


def _med_lo_hi(stats, pnames, perc, name):
    """Pull (median, lower, upper) for one parameter from a stats dict.

    Parameters
    ----------
    stats : dict
        Stats block from a saved fit (keys 'med', 'pXX', 'pYY'), as written
        by bing.io.save_fit / bing.evaluate.calc_stats.
    pnames : list of str
        Ordered parameter names matching the stats arrays.
    perc : tuple of int
        The (lower, upper) percentiles used when the stats were computed.
    name : str
        Parameter name to extract (e.g. 'Bnw').

    Returns
    -------
    tuple of float
        (median, lower-percentile, upper-percentile) in the parameter's
        native space (log10 for amplitudes, linear for slopes).
    """
    ii = pnames.index(name)
    med = stats['med'][ii]
    lo = stats[f'p{perc[0]:02d}'][ii]
    hi = stats[f'p{perc[1]:02d}'][ii]
    return float(med), float(lo), float(hi)


def parse_fits(indir: str = None, outfile: str = None,
               wv_min: float = 400., wv_max: float = 700.):
    """Parse the saved L23 fits into a pandas DataFrame and write a CSV.

    Reads each fit's lightweight JSON sidecar (the fitted values live in the
    ``stats`` block, so the heavy NPZ chains never need loading) and pairs
    the retrievals with the L23 truth.  Amplitude parameters (Bnw, Aph,
    Adg) are fit in log10; they are converted to linear here so the fitted
    and true columns are directly comparable.  Their 1-sigma uncertainty is
    reported as half the linear credible-interval width.  Slope parameters
    (beta, Sdg) are linear and reported as-is.

    Reference wavelengths match the model parameterizations:
      - bbp  : 600 nm (power-law pivot, the Bnw amplitude)
      - aph  : 440 nm (Bricaud normalization, the Aph amplitude)
      - adg  : 400 nm (exponential pivot, the Adg amplitude)

    Parameters
    ----------
    indir : str, optional
        Directory holding the L23 fits.  Defaults to OUTDIR.
    outfile : str, optional
        Path to the output CSV.  Defaults to
        ``<indir>/L23_fit_summary.csv``.
    wv_min, wv_max : float, optional
        Wavelength range passed to load_one_l23 for the truth (should match
        the fitting configuration).  Defaults 400/700 nm.

    Returns
    -------
    pandas.DataFrame
        One row per fit, with columns: idx, bbp, bbp_sig, beta, beta_sig,
        aph, aph_sig, adg, adg_sig, Sdg, Sdg_sig, true_bbp, true_aph,
        true_adg.
    """
    if indir is None:
        indir = OUTDIR
    if outfile is None:
        outfile = os.path.join(indir, 'L23_fit_summary.csv')

    # The JSON sidecars hold everything we need (stats + pnames + perc).
    json_files = sorted(glob.glob(os.path.join(indir, 'L23_*.json')))
    if len(json_files) == 0:
        print(f"No L23 fits found in {indir}")
        return None

    # Load L23 once for the truth comparison.  Use the *elastic* dataset
    # (1, 0) so the truth matches how the fitted spectra were generated by
    # lowest_bbp.generate_pace_spectrum(use_elastic=True).
    ds = loisel23.load_ds(1, 0)

    rows = []
    for jfile in json_files:
        # idx from the filename stem: L23_<idx>.json
        stem = os.path.splitext(os.path.basename(jfile))[0]
        idx = int(stem.split('_')[1])

        # Read the fit metadata / stats
        with open(jfile, 'r') as f:
            meta = json.load(f)
        stats = meta['stats']
        pnames = list(meta['pnames'])
        perc = tuple(meta['stats_perc'])

        # Fitted amplitudes (log10 -> linear); sigma = half linear interval
        bbp_med, bbp_lo, bbp_hi = _med_lo_hi(stats, pnames, perc, 'Bnw')
        bbp = 10**bbp_med
        bbp_sig = 0.5 * (10**bbp_hi - 10**bbp_lo)

        aph_med, aph_lo, aph_hi = _med_lo_hi(stats, pnames, perc, 'Aph')
        aph = 10**aph_med
        aph_sig = 0.5 * (10**aph_hi - 10**aph_lo)

        adg_med, adg_lo, adg_hi = _med_lo_hi(stats, pnames, perc, 'Adg')
        adg = 10**adg_med
        adg_sig = 0.5 * (10**adg_hi - 10**adg_lo)

        # Fitted slopes (linear); sigma = half the credible interval
        beta_med, beta_lo, beta_hi = _med_lo_hi(stats, pnames, perc, 'beta')
        beta_sig = 0.5 * (beta_hi - beta_lo)

        Sdg_med, Sdg_lo, Sdg_hi = _med_lo_hi(stats, pnames, perc, 'Sdg')
        Sdg_sig = 0.5 * (Sdg_hi - Sdg_lo)

        # L23 truth at the matching reference wavelengths
        odict = load_one_l23_truth(idx, ds, wv_min, wv_max)
        true_bbp = float(np.interp(600., odict['true_wave'],
                                   odict['bbnw']))
        true_aph = float(np.interp(440., odict['true_wave'],
                                   odict['aph']))
        true_adg = float(np.interp(400., odict['true_wave'],
                                   odict['adg']))

        rows.append(dict(
            idx=idx,
            bbp=bbp, bbp_sig=bbp_sig,
            beta=beta_med, beta_sig=beta_sig,
            aph=aph, aph_sig=aph_sig,
            adg=adg, adg_sig=adg_sig,
            Sdg=Sdg_med, Sdg_sig=Sdg_sig,
            true_bbp=true_bbp, true_aph=true_aph, true_adg=true_adg))

    # Build and write the DataFrame
    df = pandas.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"Parsed {len(df)} fits -> {outfile}")

    return df


def load_one_l23_truth(idx: int, ds, wv_min: float, wv_max: float):
    """Thin wrapper around l23.load_one_l23 for the truth dictionary.

    Parameters
    ----------
    idx : int
        L23 spectrum index.
    ds : xarray.Dataset
        Pre-loaded L23 dataset (avoids reloading per call).
    wv_min, wv_max : float
        Wavelength range for the truth.

    Returns
    -------
    dict
        The L23 truth dictionary (see l23.load_one_l23).
    """
    return l23.load_one_l23(idx, ds=ds, wv_min=wv_min, wv_max=wv_max)


def main(flg):
    flg = int(flg)

    # Debug: fit only 3 spectra
    if flg == 1:
        fit_all_l23(debug=True, n_cores=3)

    # Full dataset
    if flg == 2:
        fit_all_l23(debug=False, n_cores=20)

    # Parse the saved fits into a CSV
    if flg == 3:
        parse_fits()


# Command line
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 1  # default: debug run of 3 spectra
    else:
        flg = sys.argv[1]

    main(flg)
