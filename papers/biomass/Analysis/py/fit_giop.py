"""
Module for fitting matchup data with GIOP model.

This module processes the entire set of Argo BGC matched profiles
using the GIOP (Generalized IOP) algorithm with Lee backscattering model.
It performs both least-squares and MCMC fitting to extract IOPs with uncertainties.
"""

import os
import numpy as np

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

import pandas


from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import chisq_fit
from bing import plotting as bing_plotting


# Locals
import fitting as m_fitting

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
    tuple : (models_giop, ans_giop, cov_giop)
        Models (list): BING GIOP and Lee backscattering models
        Fitted GIOP parameters and covariance matrix.
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
        return None, None, None

    return models_giop, ans_giop, cov_giop


def fit_giop_matchup(imatched: pandas.Series, outfile: str,
                     Chl_source: str = 'expbricaud',
                     plot:bool=False):
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
    plot : bool
        Whether to generate a plot of the fit for the closest pixel.

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
    fits_file = m_fitting.set_outfile(imatched)
    fits = np.load(fits_file)

    # Unpack
    iwave = fits['wave'][0] 
    ispec = fits['Rrs'][0]
    isig =fits['Rrs_sig'][0]

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

    items = [(iwave, ispec, isig, Chl_init)]

    # Parallel fitting
    models_giop, ans_giop, cov_giop = fit_giop_single(items[0])

    # Check if all fits failed
    if ans_giop is None:
        print("All GIOP fits failed")
        return None

    # Calculate
    bbp_pivot = 10**ans_giop[-1]
    bbp_700 = bbp_pivot * (models_giop[1].pivot/700)**(models_giop[1].Y)

    # Plot
    Chl = 10**ans_giop[-1] / 0.05582
    if plot:
        bing_plotting.show_fits(models_giop, ans_giop,  
                            Chl, 
                            models_giop[1].Y,
                figsize=(12,4),
                fontsize=13.,
                Rrs_true=dict(wave=models_giop[0].wave, 
                              spec=ispec, var=isig**2),
                  log_abb=True,
                  show=True,
                )

    # Save
    out_dict = {}
    out_dict['Rrs'] = ispec
    out_dict['Rrs_sig'] = isig
    out_dict['wave'] = iwave
    out_dict['ans_giop'] = ans_giop
    out_dict['cov_giop'] = cov_giop
    out_dict['CDOM'] = 10**ans_giop[0]
    out_dict['bbp_700'] = bbp_700
    out_dict['Y'] = models_giop[1].Y
    out_dict['Chl'] = Chl
    out_dict['model_names'] = [model.name for model in models_giop]
    np.savez(outfile, **out_dict)
    print(f"Saved GIOP fit results to {outfile}")

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
    giop_Chl_vals = []
    giop_CDOM_vals = []

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
            giop_Chl_vals.append(np.nan)
            giop_CDOM_vals.append(np.nan)
            continue

        d = np.load(outfile)

        if 'Chl' not in d or not np.isfinite(d['Chl']):
            print(f"Skipping {outfile} (no valid fit)")
            giop_adg_vals.append(np.nan)
            giop_aph_vals.append(np.nan)
            giop_bbp_600_vals.append(np.nan)
            giop_bbp_700_vals.append(np.nan)
            giop_bbp_442_vals.append(np.nan)
            giop_bbp_s_vals.append(np.nan)
            giop_Y_vals.append(np.nan)
            giop_CDOM_vals.append(np.nan)
            giop_Chl_vals.append(np.nan)
            continue

        # Extract GIOP parameters (use closest pixel, index 0)
        # LM_giop has shape (nclosest, nparams) where nparams=3 for GIOP
        #giop_adg_vals.append(10**d['LM_giop'][0, 0])  # log10(adg)
        #giop_aph_vals.append(10**d['LM_giop'][0, 1])  # log10(aph)

        # Derived products
        if 'bbp_700' in d:
            giop_bbp_700_vals.append(float(d['bbp_700']))
            #giop_bbp_442_vals.append(d['bbp_442'][0])
            #giop_bbp_s_vals.append(d['bbp_s'][0])
            giop_Y_vals.append(float(d['Y']))
            giop_Chl_vals.append(float(d['Chl']))
            giop_CDOM_vals.append(float(d['CDOM']))
        else:
            giop_bbp_700_vals.append(np.nan)
            giop_bbp_442_vals.append(np.nan)
            giop_bbp_s_vals.append(np.nan)
            giop_Y_vals.append(np.nan)
            giop_Chl_vals.append(np.nan)
            giop_CDOM_vals.append(np.nan)

        if debug:
            break

    # Add to DataFrame
    #matched['GIOP_adg'] = np.array(giop_adg_vals)
    #matched['GIOP_aph'] = np.array(giop_aph_vals)
    #matched['GIOP_bbp_600'] = np.array(giop_bbp_600_vals)
    matched['GIOP_bbp_700'] = np.array(giop_bbp_700_vals)
    #matched['GIOP_bbp_442'] = np.array(giop_bbp_442_vals)
    #matched['GIOP_bbp_s'] = np.array(giop_bbp_s_vals)
    matched['GIOP_Y'] = np.array(giop_Y_vals)
    matched['GIOP_Chl'] = np.array(giop_Chl_vals)
    matched['GIOP_CDOM'] = np.array(giop_CDOM_vals)

    return matched


def set_giop_outfile(imatched: pandas.Series):
    """Generate output filename for GIOP fits."""
    outfile = os.path.join(os.getenv('OS_COLOR'), 'Biomass', 'Fits',
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
                         plot=True)

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
                fit_giop_matchup(imatched, outfile, Chl_source='expbricaud')
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
        #output_file = match_file.replace('.csv', '_with_GIOP.csv')
        matched.to_csv(match_file, index=False)
        print(f"\nSaved results to: {match_file}")
        print(f"Total profiles: {len(matched)}")
        print(f"Successful GIOP fits: {(~matched.GIOP_bbp_700.isna()).sum()}")
