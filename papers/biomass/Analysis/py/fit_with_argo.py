# Fit a PACE spectrum constraining the backscattering to Argo data

import os, sys
import numpy as np

from matplotlib import pyplot as plt

import xarray
import pandas

from ocpy.pace import io as pace_io
from ocpy.utils import plotting

from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf
from bing import evaluate
from bing.fitting import chisq_fit
from bing import plotting as bing_plotting
from bing import evaluate

#
# Locals
from grab_pace_granules import closest_Rrs
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
    outfile = os.path.join(outdir, os.path.basename(fit_file).replace('Argo_', 'Argo_Constrained_'))
    print(f"Working on {imatched.cruise}-{imatched.profile:03d}...")

    # Load
    d = biomass_io.load_fit_data(fit_file)
    items = [d['wave'][0], d['Rrs'][0], d['Rrs_sig'][0]]

    # Standard
    if not load_fits:
        print("="*80)
        print("Fitting unconstrained...")
        print("="*80)
        models_S, chains_S, ans_S, stats_S, rt_dict_S, pdict_S = fitting.fit_me(items)
        embed(header='73 of fit_with_argo.py')
        a_mean_S, bb_mean_S, a_5_S, a_95_S, bb_5_S, bb_95_S,\
            model_Rrs_S, sigRs_S = evaluate.reconstruct_from_chains(
            models_S, chains_S, rt_dict_S)#, perc=perc)
    else:
        embed(header='77 of fit_with_argo.py')
        raise NotImplementedError("Not implemented yet")

    # With Argo
    bpriors=[dict(flavor='log_uniform', pmin=-6, pmax=5)]*2

    # Constrain by Argo
    bpriors[0]=dict(flavor='gaussian', mean=np.log10(imatched.argo_bbp700), sigma=0.02, pmin=-6., pmax=5)

    # Uniform for beta from 0. - 2. (positive here means negative slope)
    bpriors[1]=dict(flavor='uniform', pmin=0., pmax=2.)

    p = standard.expb_pow(satellite='PACE', add_noise=False,
                variable_Gordon=True, include_Raman=True, bpriors=bpriors,
                include_Chl_fl=True, phi_C=0.02, double_gaussian=True)
    print("="*80)
    print("Fitting constrained...")
    print("="*80)
    models_C, chains_C, ans_C, stats_C, rt_dict_C, pdict_C = fitting.fit_me(items, in_p=p)
    a_mean_C, bb_mean_C, a_5_C, a_95_C, bb_5_C, bb_95_C,\
            model_Rrs_C, sigRs_C = evaluate.reconstruct_from_chains(
            models_C, chains_C, rt_dict_C)#, perc=perc)

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
    ax.legend()
    plotting.set_fontsize(ax, 15.)
    #
    outfig = outfile.replace('npz', 'png')
    plt.savefig(outfig, dpi=300)
    print(f"Saved: {outfig}")
    plt.show()
    plt.close()



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
        
        fit_with_and_without_argo(cruise_profile_n25, load_fits=False)

# Command line
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)
    