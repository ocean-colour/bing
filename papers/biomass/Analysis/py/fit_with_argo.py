# Fit a PACE spectrum constraining the backscattering to Argo data

import os, sys
import glob
import numpy as np

from matplotlib import pyplot as plt

import pandas

from ocpy.utils import plotting

from bing.parameters import standard
from bing import io as bing_io

#
# Locals
import biomass_io
import fitting

from IPython import embed

def find_an_example(idx, match_file:str=None):
    if match_file is None:
        match_file = 'matched_argo_bgc_profiles_bbp_v3.csv'
    matched = pandas.read_csv(match_file)

    # Find largest departures
    reld = ((matched.BING_Bnw - matched.argo_bbp700) / matched.BING_Bnw).values
    rsrt = np.argsort(reld)
    idx = rsrt[idx]
    imatched = matched.iloc[idx]
    print(imatched)
    #
    return imatched

def fit_with_and_without_argo(cruise_profile, outdir='Argo_Constrained', 
                              load_fits:bool=True, match_file:str=None):
    if match_file is None:
        match_file = 'matched_argo_bgc_profiles_bbp_v3.csv'
    # Grab it
    matched = pandas.read_csv(match_file)
    mt = (matched.cruise == cruise_profile[0]) & (matched.profile == cruise_profile[1])
    idx = np.where(mt)[0]
    if len(idx) != 1:
        raise ValueError(f"Found {len(idx)} matches for {cruise_profile[0]}-{cruise_profile[1]}")
    imatched = matched.iloc[idx[0]]
    #embed(header='49 of fit_with_argo.py')

    # Prep
    fit_file = biomass_io.get_fit_file_path(imatched)
    base, _ = os.path.splitext(os.path.basename(fit_file))
    outroot_S = os.path.join(outdir, base)
    outroot_C = os.path.join(outdir, base.replace('Argo_', 'Argo_Constrained_'))
    print(f"Working on {imatched.cruise}-{imatched.profile:03d}...")

    #embed(header='57 of fit_with_argo.py')

    # Load
    d = biomass_io.load_fit_data(fit_file)
    items = [d['wave'][0], d['Rrs'][0], d['Rrs_sig'][0]]

    # Standard
    if not load_fits:
        print("="*80)
        print("Fitting unconstrained...")
        print("="*80)
        p_S = standard.expb_pow(satellite='PACE', add_noise=False,
                variable_Gordon=True, include_Raman=True, 
                include_Chl_fl=True, phi_C=0.02, double_gaussian=True)
        # Fit
        models_S, chains_S, ans_S, stats_S, rt_dict_S, pdict_S, p_S = fitting.fit_me(
            items, in_p=p_S)
        # Reconstruct
        #a_mean_S, bb_mean_S, a_5_S, a_95_S, bb_5_S, bb_95_S,\
        #    model_Rrs_S, sigRs_S = evaluate.reconstruct_from_chains(
        #    models_S, chains_S, rt_dict_S)#, perc=perc)
        # Save
        bing_io.save_fit(outroot_S, p_S, models_S, chains_S, ans_S, items[1], items[2]**2)

    # Load for Rrs
    fits_S = bing_io.load_fit(outroot_S)
    model_Rrs_S = fits_S['Rrs_recon']

    # With Argo
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Constrain by Argo
    bpriors[0]=dict(flavor='gaussian', mean=np.log10(imatched.argo_bbp700), sigma=0.02, pmin=-6., pmax=5)

    # Uniform for beta from 0. - 2. (positive here means negative slope)
    bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

    p_C = standard.expb_pow(satellite='PACE', add_noise=False,
                variable_Gordon=True, include_Raman=True, bpriors=bpriors,
                include_Chl_fl=True, phi_C=0.02, double_gaussian=True)

    print("="*80)
    print("Fitting constrained...")
    print("="*80)
    models_C, chains_C, ans_C, stats_C, rt_dict_C, pdict_C, p_C = fitting.fit_me(items, in_p=p_C)
    bing_io.save_fit(outroot_C, p_C, models_C, chains_C, ans_C, items[1], items[2]**2)
    # Load for Rrs
    fits_C = bing_io.load_fit(outroot_C)
    model_Rrs_C = fits_C['Rrs_recon']

    # Plot

    fig = plt.figure(figsize=(12,7))
    ax = plt.gca()
    #
    ax.errorbar(items[0], items[1], yerr=items[2],
                    color='k',  fmt='o', capsize=5) 
    #ax.plot(models_V[0].wave, model_Rrs_V, label='Vanilla',zorder=10)
    ax.plot(items[0], model_Rrs_C, label='Argo',zorder=20)
    ax.plot(items[0], model_Rrs_S, color='orange', label='Unconstrained')

    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_{\rm rs}$')
    #
    ax.legend(fontsize=17.)
    plotting.set_fontsize(ax, 17.)
    #
    outfig = outroot_C + '.png'
    plt.savefig(outfig, dpi=300)
    print(f"Saved: {outfig}")
    plt.show()
    plt.close()



def examine_parameter_changes(indir:str='Argo_Constrained', verbose:bool=True):
    """Build a DataFrame comparing free vs. Argo-constrained fit parameters.

    Loads every matched pair of Argo_<cruise>_<profile>_fits and
    Argo_Constrained_<cruise>_<profile>_fits in ``indir`` via
    :func:`bing.io.load_fit`, pulls the median parameter values from each fit's
    ``stats`` block, and returns one row per (cruise, profile) with both sets
    of parameters plus their differences.

    Parameters
    ----------
    indir : str
        Folder containing the saved fit files.
    verbose : bool
        Print a short summary of the parameter changes.

    Returns
    -------
    pandas.DataFrame
        One row per cruise/profile pair with columns:
        ``cruise``, ``profile``, ``<pname>_free``, ``<pname>_cons``,
        ``<pname>_diff`` (constrained - free, in log10 space for amplitudes).
    """
    # Find every constrained fit and pair it to its free counterpart.
    cons_files = sorted(glob.glob(os.path.join(
        indir, 'Argo_Constrained_*_fits.npz')))

    rows = []
    for cons_path in cons_files:
        # Strip extension and derive the matching free-fit root.
        cons_root = os.path.splitext(cons_path)[0]
        free_root = cons_root.replace('Argo_Constrained_', 'Argo_')

        # Skip if the free counterpart is missing on disk.
        if not os.path.exists(free_root + '.npz'):
            if verbose:
                print(f"Skipping {cons_root}: no free counterpart")
            continue

        # Parse cruise/profile from the filename stem
        # (format: Argo_Constrained_<cruise>_<profile>_fits)
        stem = os.path.basename(cons_root)
        parts = stem.split('_')
        cruise = int(parts[2])
        profile = int(parts[3])

        # Load both fits
        free = bing_io.load_fit(free_root)
        cons = bing_io.load_fit(cons_root)

        # Pull median parameter values (in the model's native log10/linear mix)
        pnames = list(free['pnames'])
        med_free = np.asarray(free['stats']['med'])
        med_cons = np.asarray(cons['stats']['med'])

        row = {'cruise': cruise, 'profile': profile}
        for jj, pname in enumerate(pnames):
            row[f'{pname}_free'] = med_free[jj]
            row[f'{pname}_cons'] = med_cons[jj]
            row[f'{pname}_diff'] = med_cons[jj] - med_free[jj]
        rows.append(row)

    df = pandas.DataFrame(rows)

    if verbose and len(df) > 0:
        # Print the diff columns so the caller sees the parameter shifts.
        diff_cols = [c for c in df.columns if c.endswith('_diff')]
        # Add Bnw too
        diff_cols.append('Bnw_free')
        diff_cols.append('Bnw_cons')
        diff_cols.append('beta_free')
        diff_cols.append('beta_cons')

        print("Parameter changes (constrained - free):")
        print(df[['cruise', 'profile'] + diff_cols].to_string(index=False))

    return df


def main(flg):
    flg= int(flg)

    # Find one
    if flg == 1:
        idx = -25  # 10x higher PACE;  probably clouds/glint
        idx = -35  # 10x higher PACE;  probably clouds/glint
        idx = -45  # Same as above
        idx = -55  # Not quite as extreme
        idx = -100  # Less extreme, but similar
        idx = -130  #
        find_an_example(idx)

    # Fit
    if flg == 2:
        cruise_profile_n25 = (5906537,85) # idx = -25
        cruise_profile_n55 = (6903823,387) # idx = -55
        cruise_profile_n130 = (6903823,427) # idx = -130
        # JR
        cruise_profile_jr6 = (7902226,4) # Clearest sky
        fit_with_and_without_argo(cruise_profile_jr6, load_fits=False)

        #for cruise_profile in [cruise_profile_n25, cruise_profile_n55, cruise_profile_n130]:
        #    fit_with_and_without_argo(cruise_profile, load_fits=False)

    # Compare parameters between free and constrained fits
    if flg == 3:
        examine_parameter_changes()

# Command line
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)
    