# Compare JR (Frouin) Rrs extractions to PACE Rrs from BING fits

import os, sys

import numpy as np
import pandas

from matplotlib import pyplot as plt

from ocpy.utils import plotting

from bing import io as bing_io

# Locals
import biomass_io
import fitting
import jr_utils

from IPython import embed


# Default folder where JR fit outputs are stored (relative to py/)
FROUIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Frouin')


def compare_jr_to_pace(inputs, outdir: str = 'Frouin',
                      match_file: str = None, show: bool = True,
                      use_jr_idx:bool=True):
    """Compare the JR-extracted Rrs to the PACE Rrs used by the BING fit.

    Loads the JR matchup spectrum for the requested Argo float and the
    saved PACE fit file for the same profile, then overlays the two
    spectra on a single figure.

    Parameters
    ----------
    inputs : int or tuple
        if int, it is the JR index otherwise
        Argo float (cruise, profile) identifying the matchup to plot.
    outdir : str
        Directory to write the comparison figure into. Created if absent.
    match_file : str, optional
        Override path to the matched-Argo CSV (defaults to v3 table).
    show : bool
        If True, call ``plt.show()`` after saving.

    Returns
    -------
    dict
        The JR extraction dictionary from :func:`jr_utils.extract_rrs`
        with the loaded PACE fit data added under key ``'pace_fit'``.
    """
    if match_file is None:
        match_file = 'matched_argo_bgc_profiles_bbp_v3.csv'

    # Resolve the matched Argo row to get the PACE fit file path
    matched = pandas.read_csv(match_file)

    if use_jr_idx:
        jr_idx = inputs
        #
        mdict = jr_utils.match_jr_to_argo(jr_idx)
        cruise, profile = mdict['cruise'], mdict['profile']
    else:
        raise NotImplementedError()

    imatched = biomass_io.get_matched_profile(matched, cruise, profile)

    print(f"Working on {cruise}-{profile:03d}...")

    # Load JR Rrs for this (cruise, profile) via jr_utils
    jr = jr_utils.extract_rrs(jr_idx)
    #print(f"  JR match: dist={jr['match_dist_km']:.2f} km, "
    #      f"dt={jr['match_dt_hours']:.2f} h")

    # Load PACE Rrs from $OS_COLOR/Biomass/Fits via load_fit_data
    fit_file = biomass_io.get_fit_file_path(imatched)
    if not os.path.isfile(fit_file):
        raise FileNotFoundError(f"PACE fit file not found: {fit_file}")
    pace = biomass_io.load_fit_data(fit_file)

    # Take the first spectrum
    pace_wave = pace['wave'][0]
    pace_Rrs = pace['Rrs'][0]
    pace_Rrs_sig = pace['Rrs_sig'][0]

    # Build output figure path
    os.makedirs(outdir, exist_ok=True)
    base = f"JR_vs_PACE_{imatched.cruise}_{imatched.profile:03d}"
    outroot = os.path.join(outdir, base)

    # Plot in the style of fit_with_argo.py
    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # JR spectrum with its reported std as error bars
    ax.errorbar(jr['wavelengths'], jr['Rrs_mean'], yerr=jr['Rrs_std'],
                color='tab:blue', fmt='o', capsize=4, ms=4,
                label='JR (Frouin)', zorder=10)

    # PACE Rrs: shaded band shows pixel-to-pixel spread + central curve
    #ax.fill_between(pace_wave,
    #                pace_Rrs_mean - pace_Rrs_std,
    #                pace_Rrs_mean + pace_Rrs_std,
    #                color='orange', alpha=0.25,
    #                label='PACE pixel spread')
    ax.errorbar(pace_wave, pace_Rrs, yerr=pace_Rrs_sig,
                color='k', fmt='-', lw=1.5,
                label='PACE (BING input)', zorder=20)

    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_{\rm rs}$')
    ax.set_title(f"Argo {imatched.cruise}-{imatched.profile:03d}: "
                 f"JR vs. PACE Rrs")
    ax.legend(fontsize=15.)
    plotting.set_fontsize(ax, 15.)

    outfig = outroot + '.png'
    plt.savefig(outfig, dpi=300)
    print(f"Saved: {outfig}")
    if show:
        plt.show()
    plt.close()

    # Return the data dict for downstream inspection
    jr['pace_fit'] = pace
    jr['pace_wave'] = pace_wave
    jr['pace_Rrs_mean'] = pace_Rrs
    jr['pace_Rrs_std'] = pace_Rrs_sig
    return jr


def fit_jr_rrs(cruise_profile, outdir: str = None,
               wv_min: float = 400., wv_max: float = 700.,
               show: bool = False):
    """Fit a JR Rrs extraction with BING and write outputs to Frouin/.

    Pulls the JR-processed spectrum for the requested Argo (cruise,
    profile), restricts to the BING fitting window (default 400-700 nm),
    runs the standard LM + MCMC pipeline via :func:`fitting.fit_me`,
    saves the fit through :func:`bing.io.save_fit`, and renders a
    diagnostic figure with :func:`fitting.plot_fit`.

    Parameters
    ----------
    cruise_profile : tuple of (int, int)
        Argo float (cruise, profile) selecting the JR spectrum to fit.
    outdir : str, optional
        Folder for the saved fit and figure. Defaults to
        ``papers/biomass/Analysis/Frouin``.
    wv_min, wv_max : float
        Wavelength window (nm) retained from the JR spectrum before
        fitting. BING's PACE-style configuration covers 400-700 nm.
    show : bool
        Forwarded to ``plt.show`` after the figure is written.

    Returns
    -------
    dict
        Keys: ``outroot``, ``fit_file``, ``plot_file``, ``wave``,
        ``Rrs``, ``Rrs_sig``, ``stats``, ``ans`` (LM best-fit),
        ``chains``.
    """
    if outdir is None:
        outdir = FROUIN_DIR
    os.makedirs(outdir, exist_ok=True)

    # Load the JR Rrs spectrum for this matchup via jr_utils
    mdict = jr_utils.match_argo_to_jr(cruise_profile[0], cruise_profile[1])
    jr = jr_utils.extract_rrs(mdict['jr_idx'])
    dt = (pandas.Timestamp(mdict['argo_row']['closest_time']) - mdict['argo_row']['time'])
    dt_hours = dt.seconds / 3600.0
    #print(f"  JR match: dist={jr['match_dist_km']:.2f} km, "
    #      f"dt={jr['match_dt_hours']:.2f} h")
    #embed(header='175 of jr_analysis.py')

    # Subset to the BING fitting window (PACE-like, 400-700 nm)
    wave_all = jr['wavelengths']
    gd = (wave_all >= wv_min) & (wave_all <= wv_max)
    iwave = wave_all[gd].astype(float)
    ispec = jr['Rrs_mean'][gd].astype(float)
    isig = jr['Rrs_std'][gd].astype(float)

    # Floor zero/negative reported std so chi^2 stays finite in the LM step
    #floor_sig = 1e-5
    #isig = np.where(isig > 0, isig, floor_sig)

    # File naming mirrors the biomass_io convention but tagged "JR_"
    base = f"JR_{cruise_profile[0]}_{cruise_profile[1]:03d}_fits"
    outroot = os.path.join(outdir, base)
    fit_file = outroot + '.npz'

    # Run BING fit (LM + MCMC) via the standard pipeline used by fit_with_argo
    print("=" * 80)
    print(f"Fitting JR Rrs for "
          f"{cruise_profile[0]}-{cruise_profile[1]:03d} ...")
    print("=" * 80)
    models, chains, ans, stats, rt_dict, pdict, p = fitting.fit_me(
        [iwave, ispec, isig])

    # Persist fit via bing.io.save_fit (writes .npz + .json)
    bing_io.save_fit(outroot, p, models, chains, ans, ispec, isig**2)
    print(f"Saved fit: {fit_file}")

    # Plot via fitting.plot_fit, write figure into the Frouin/ folder
    title = (f'JR Argo {cruise_profile[0]}-{cruise_profile[1]:03d} '
             f"(match dist={mdict['argo_row']['closest_dist_km']:.1f} km, "
             f"dt={dt_hours:.1f} h)")
    Rrs_obs = dict(wave=iwave, spec=ispec, var=isig**2)
    plot_file = outroot + '.png'
    fitting.plot_fit(models, chains, Rrs_obs, title, rt_dict,
                     stats=stats, show_Rsig=True, outfile=plot_file)
    if show:
        plt.show()
    plt.close('all')

    return {
        'outroot': outroot,
        'fit_file': fit_file,
        'plot_file': plot_file,
        'wave': iwave,
        'Rrs': ispec,
        'Rrs_sig': isig,
        'stats': stats,
        'ans': ans,
        'chains': chains,
    }


def list_jr_matchups(jr_file: str = None, argo_file: str = None,
                     verbose: bool = True):
    """Read the JR matchup CSV and print the Argo (cruise, profile) for each row.

    Each row in ``jr_test_matchup_L1B.csv`` corresponds to one Rrs
    spectrum but only carries the PACE geolocation/time (not the Argo
    float identifiers). This method reverse-matches each JR row to the
    nearest entry in ``matched_argo_bgc_profiles_bbp_v3.csv`` using the
    same distance/time metric as :func:`jr_utils.extract_rrs`.

    Parameters
    ----------
    jr_file, argo_file : str, optional
        Override paths for the JR and Argo CSVs (see ``jr_utils``).
    verbose : bool
        If True (default), print one line per spectrum.

    Returns
    -------
    pandas.DataFrame
        The JR DataFrame with ``cruise``, ``profile``,
        ``match_dist_km``, ``match_dt_hours``, ``argo_lat``,
        ``argo_lon``, and ``argo_time`` columns appended.
    """
    # Load both tables using the helpers from jr_utils
    jr_df = jr_utils.load_jr_data(jr_file)
    argo_df = jr_utils.load_argo_data(argo_file)

    # For each JR row, find the closest Argo (cruise, profile).
    # Rows with missing lat/lon/time are reported as unmatched.
    cruises, profiles, dists, dts = [], [], [], []
    argo_lats, argo_lons, argo_times = [], [], []
    for jr_idx, jr_row in jr_df.iterrows():
        if (pandas.isna(jr_row['PACE_lat']) or
                pandas.isna(jr_row['PACE_lon']) or
                pandas.isna(jr_row['time'])):
            cruises.append(-1)
            profiles.append(-1)
            dists.append(np.nan)
            dts.append(np.nan)
            argo_lats.append(np.nan)
            argo_lons.append(np.nan)
            argo_times.append(pandas.NaT)
            continue

        mdict = jr_utils.match_jr_to_argo(jr_idx)
        # Pull the matched Argo profile's lat/lon/time for reporting
        argo_row = argo_df.loc[mdict['argo_row']]

        cruises.append(int(mdict['cruise']))
        profiles.append(int(mdict['profile']))
        dists.append(float(mdict['dist_km']))
        dts.append(float(mdict['dt_hours']))
        argo_lats.append(float(argo_row['lat']))
        argo_lons.append(float(argo_row['lon']))
        argo_times.append(pandas.Timestamp(argo_row['time']))

    # Annotate the DataFrame with the matched Argo IDs
    jr_df = jr_df.copy()
    jr_df['cruise'] = cruises
    jr_df['profile'] = profiles
    jr_df['match_dist_km'] = dists
    jr_df['match_dt_hours'] = dts
    jr_df['argo_lat'] = argo_lats
    jr_df['argo_lon'] = argo_lons
    jr_df['argo_time'] = argo_times

    if verbose:
        print(f"JR spectra in {jr_file or 'jr_test_matchup_L1B.csv'} "
              f"({len(jr_df)} rows):")
        for ii, row in jr_df.iterrows():
            if row['cruise'] < 0:
                print(f"  [{ii:2d}] (no PACE lat/lon/time -- unmatched)")
                continue
            # UT time of the matched Argo profile (ISO, seconds resolution)
            ut = pandas.Timestamp(row['argo_time']).strftime(
                '%Y-%m-%dT%H:%M:%S')
            print(f"  [{ii:2d}] cruise={row['cruise']:>7d}  "
                  f"profile={row['profile']:>4d}  "
                  f"dist={row['match_dist_km']:6.2f} km  "
                  f"dt={row['match_dt_hours']:7.2f} h  "
                  f"Argo lat={row['argo_lat']:8.3f}  "
                  f"lon={row['argo_lon']:8.3f}  "
                  f"UT={ut}")

    return jr_df


def main(flg):
    flg = int(flg)

    # Compare a single matchup
    if flg == 1:
        # Same example used by fit_with_argo.main(flg=2)
        jr_idx = 0
        compare_jr_to_pace(jr_idx)

    # Compare the standard set of test cases
    if flg == 2:
        for jr_idx in range(9):
            try:
                compare_jr_to_pace(jr_idx, show=True)
            except Exception as e:
                print(f"  Skipping {jr_idx}: {e}")

    # Fit a single JR Rrs spectrum with BING
    if flg == 3:
        cruise_profile = (7902226,4) # Clearest sky
        fit_jr_rrs(cruise_profile)

    # Fit the same trio of test cases
    if flg == 4:
        jr_df = list_jr_matchups()
        for ii, row in jr_df.iterrows():
            cp = (row['cruise'], row['profile'])
            try:
                fit_jr_rrs(cp)
            except Exception as e:
                print(f"  Skipping {cp}: {e}")

    # List the (cruise, profile) for every spectrum in the JR CSV
    if flg == 5:
        list_jr_matchups()


# Command line
if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 1
    else:
        flg = sys.argv[1]

    main(flg)
