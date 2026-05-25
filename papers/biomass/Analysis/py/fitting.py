
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

from correct_atmosphere import downwelling

from bing import evaluate
from bing import plotting as bing_plot
from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs
from bing.rt import chl_fl

# Locals
from grab_pace_granules import closest_Rrs
import biomass_io

from IPython import embed

#def chains_to_param(fit:dict):
#    iwave = fit['wave']
#
#    # Init models
#    p = standard.expb_pow()
#    models = model_utils.init(p.model_names, iwave)
#    bing_priors.set_standard_priors(models, p)
#    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)


def fit_me(items:list[tuple], debug:bool=False, in_p=None, return_early:bool=False,
    guess_vals:np.ndarray=None):
    """
    Fit a single spectrum.

    Parameters:
    -----------
    items : list[tuple]
        A list containing the following items:
        - iwave : numpy.ndarray
        - ispec : numpy.ndarray
        - isig : numpy.ndarray
    in_p : dict, optional
        A dictionary containing the parameters for the model.
    return_early : bool, optional
        If True, return the models, ans, and rt_dict.
    guess_vals : np.ndarray, optional
        An array containing the initial guess values for the parameters.

    Returns:
    --------
    tuple: (models, chains, ans, stats)
        - models : list
        - chains : numpy.ndarray
        - ans : numpy.ndarray
        - stats : dict
        - rt_dict : dict
    """

    iwave, ispec, isig = items

    # Init models
    if in_p is None:
        p = standard.expb_pow(satellite='PACE', add_noise=False,
            variable_Gordon=True, include_Raman=True,
            include_Chl_fl=True, phi_C=0.02, double_gaussian=True)
    else:
        p = in_p
    models = model_utils.init(p.model_names, iwave)

    # RT
    if p.variable_Gordon:
        models[0].init_var_gordon()
    if p.include_Chl_fl:
        Ed = downwelling.downwelling_irradiance(models[0].wave, 0.)
        Ed_em = downwelling.downwelling_irradiance(chl_fl.LAMBDA_FL_PRIMARY, 0.)
        models[0].init_Chl_fluorescence(Ed=Ed, Ed_em=Ed_em)
    
    # Priors
    bing_priors.set_standard_priors(models, p)

    # Initialize the MCMC
    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)

    # Fit with LM for first guess
    low_bounds, high_bounds = [], []
    low_bounds += [item['pmin'] for item in p.apriors]
    low_bounds += [item['pmin'] for item in p.bpriors]
    high_bounds += [item['pmax'] for item in p.apriors]
    high_bounds += [item['pmax'] for item in p.bpriors]
    #
    bounds = (np.array(low_bounds), np.array(high_bounds))

    if guess_vals is None:
        p0 = [-2, 0.015, -2, -2, 1.5]
    else:
        p0 = guess_vals
    items = [(ispec, isig**2, p0, 0)]

    rt_dict = rt_defs.rt_dict_from_p(p)
    #embed(header='101 of fitting.py')

    # LM
    try:
        ans, cov, idx = chisq_fit.fit(items[0], models, rt_dict, bounds=bounds)
    #except RuntimeError:
    except:
        embed(header='127 of fitting.py')
        print("Fit failed: saving -999")
        return None, None, None, None
    
    # Return here?
    if return_early:
        return models, ans, rt_dict 
    
    # Plot?
    if debug:
        embed(header='121 of fitting.py')
        Chl = 10**ans[2] / 0.05582
        _ = bing_plot.show_fits(models, ans, rt_dict, Chl, None,
                figsize=(12,4), fontsize=13., show=True,
                Rrs_true=dict(wave=models[0].wave, spec=ispec, var=isig**2),
                log_abb=True )

    # Now the MCMC
    p0 = ans.tolist()
    items = [(ispec, isig**2, p0, 0)]
    pdict['Chl'] = np.array([10**p0[2] / 0.05582])
    pdict['Y'] = None

    print("----- Fitting with MCMC -----")
    chains, idx = bing_inf.fit_one(
        items[0], models=models, pdict=pdict, chains_only=True, rt_dict=rt_dict)
    stats = evaluate.calc_stats(chains)

     # Return
    return models, chains, ans, stats, rt_dict, pdict, p

def fit_one(imatched:pandas.Series, outfile:str, debug:bool=False,
            nclosest:int=1, n_cores:int=10):
    """
    Perform spectral fitting on matched data using a combination of 
    least-squares fitting and Markov Chain Monte Carlo (MCMC) methods.

    Parameters:
    -----------
    imatched : pandas.Series
        A pandas Series containing metadata and information about the 
        matched data point, including latitude, longitude, cruise, profile, 
        and closest file path.
    outfile : str
        Path to the output file where the results will be saved.
    debug : bool, optional
        If True, enables debugging mode with an interactive session. 
        Default is False.

    Workflow:
    ---------
    1. Load the PACE file corresponding to the matched data.
    2. Find the closest remote sensing reflectance (Rrs) data point(s)
    3. Parse the spectral data and uncertainties for the selected wavelengths.
    4. Initialize models and priors for the fitting process.
    5. Perform a least-squares fit to obtain an initial guess for the parameters.
    6. If the least-squares fit fails, save a placeholder result and exit.
    7. Use the initial guess to perform MCMC fitting and calculate statistics.
    8. Save the fitting results, including chains, statistics, and metadata.
    9. Generate and save a plot of the fitting results for the closest good fit

    Outputs:
    --------
    - A `.npz` file containing the fitting results, including:
        - MCMC chains
        - Least-squares fit parameters
        - Wavelengths, Rrs, and uncertainties
        - Median, 5th percentile, and 95th percentile statistics
        - Model names
    - A `.png` file with a plot of the fitting results.

    Notes:
    ------
    - The function assumes the existence of specific modules and functions 
        such as `pace_io.load_oci_l2`, `closest_Rrs`, `model_utils.init`, 
        `bing_priors.set_standard_priors`, `bing_inf.init_mcmc`, 
        `chisq_fit.fit`, `bing_inf.fit_one`, and `evaluate.calc_stats`.
    - The function also assumes that the environment variable `OS_COLOR` 
        is set and points to the base directory for the PACE data files.

    Exceptions:
    -----------
    - If the least-squares fitting fails, a RuntimeError is caught, and 
        placeholder results are saved with parameter values set to -999.
    """
    # Save
    out_dict = {}

    # Load PACE file
    gfile = os.path.join(biomass_io.PACE_L2_AOP_PATH,
                            imatched.closest_file)
    print(f"----- Loading {gfile} -----")
    xds, flags = pace_io.load_oci_l2(gfile)

    # Find closest Rrs
    d_min, dmin_ij = closest_Rrs(xds, (imatched.lat, imatched.lon),
                                 nclosest=nclosest)
    if d_min is None:
        print("No good data after all")
        out_dict['LM'] = np.array([-999]*5)
        np.savez(outfile, **out_dict)
        return

    #if debug:
    #    embed(header='196 of fitting.py')

    # Parse out the data
    gd_wave = (xds.wavelength.data >= 400.) &  (xds.wavelength.data <= 700.) 

    map_fn = partial(fit_me)

    # Setup
    items = []
    for ss in range(nclosest):
        ix, iy = dmin_ij[0][ss], dmin_ij[1][ss]
        iwave = xds.wavelength.data[gd_wave]
        ispec = xds.Rrs.data[ix,iy,gd_wave]
        isig = xds.Rrs_unc.data[ix,iy,gd_wave]
        items.append((iwave, ispec, isig))

    # Fit
    if debug:
        models, chains, ans, stats, rt_dict, pdict, p = fit_me(items[0], debug=True) 

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = nclosest // n_cores if nclosest // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items, chunksize=chunksize), total=nclosest))

    # Grab em all
    all_ans, all_stats, all_lon, all_lat = [], [], [], []
    all_idx, all_dist = [], []
    all_spec = []
    ok_ss = []
    for ss, aa in enumerate(answers):
        if aa[2] is not None:
            ok_ss.append(ss)
            # Grab
            all_ans.append(aa[2])
            all_stats.append(aa[3])
            # Coords
            ix, iy = dmin_ij[0][ss], dmin_ij[1][ss]
            all_idx.append(np.array([ix, iy]))
            all_lon.append(xds.longitude.data[ix,iy])
            all_lat.append(xds.latitude.data[ix,iy])
            all_dist.append(d_min[ss])
            # Spectra
            all_spec.append(items[ss])


    # Bust?
    if len(all_ans) == 0:
        print("All fits failed: saving -999")
        out_dict['LM'] = np.array([-999]*5)
        out_dict['wave'] = items[0][0]
        out_dict['Rrs'] = items[0][1]
        out_dict['Rrs_sig'] = items[0][2]
        out_dict['Rrs_idx'] = np.array([dmin_ij[0][0], dmin_ij[1][0]]) # ij in xds
        np.savez(outfile, **out_dict)
        return

    # Grab the closest
    models, chains, ans, stats, rt_dict, pdict = answers[ok_ss[0]]
    ispec, isig = all_spec[0][1], all_spec[0][2]

    out_dict['chains'] = chains
    out_dict['LM'] = np.stack(all_ans)
    out_dict['wave'] = np.stack([it[0] for it in all_spec])
    out_dict['Rrs'] = np.stack([it[1] for it in all_spec])
    out_dict['Rrs_sig'] = np.stack([it[2] for it in all_spec])
    out_dict['Rrs_idx'] = np.stack(all_idx) # ij in xds
    out_dict['lon'] = np.array(all_lon)
    out_dict['lat'] = np.array(all_lat)
    out_dict['dist'] = np.array(all_dist)
    #
    out_dict['med'] = np.stack([stats['med'] for stats in all_stats])
    out_dict['p14'] = np.stack([stats['p14'] for stats in all_stats])
    out_dict['p86'] = np.stack([stats['p86'] for stats in all_stats])
    out_dict['model_names'] = [model.name for model in models]

    np.savez(outfile, **out_dict)
    print(f"Saved: {outfile}")

    # Plot me
    print("----- Plotting -----")
    title = f'Float={imatched.cruise}-{imatched.profile}, lat={imatched.lat:.1f},'+\
    f'lon={imatched.lon:.1f}, time={imatched.time[:19]}, {imatched.closest_id[12:-7]}, dist={all_dist[0]:.1f} km'
    Rrs_obs=dict(wave=models[0].wave, spec=ispec, var=isig**2)
    plotfile=outfile.replace('.npz', '.png')
    plot_fit(models, chains, Rrs_obs, title, rt_dict, show_Rsig=True,
                   outfile=plotfile)

    

def plot_fit(models, chains, Rrs_obs, title:str, rt_dict:dict, stats:dict=None,
             outfile:str=None, 
             ulist:list=None, 
             perc:tuple=(14,86),
             show_Rsig:bool=False):

    # Do this first
    mini_corner(models, chains, ['Sdg', 'beta', 'Bnw'],
                outfile='tmpc.png')

    if stats is None:
        stats = evaluate.calc_stats(chains, perc=perc)

    # Wavelengths
    wave = models[0].wave

    # Unpack for development
    if ulist is not None:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = ulist
    else:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains, rt_dict, perc=perc)

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    bb_w = w_scattering.bbw_from_l23(wave)

    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.3, 1, 1], hspace=0.02)

    # Then modify the subplot assignments:
    # Change ax_R = plt.subplot(gs[0]) to:
    ax_res = plt.subplot(gs[0])  # New residuals axis
    ax_R = plt.subplot(gs[2], sharex=ax_res)  # Rrs axis, now at position 2

    # Keep the other axes but update their positions:
    ax_anw = plt.subplot(gs[3])  # was gs[1], now gs[3]
    ax_bb = plt.subplot(gs[4])   # was gs[2], now gs[4]
    ax_c = plt.subplot(gs[5])    # was gs[3], now gs[5]

    # After calculating chi2 and before the Rrs plotting section, add residual plotting:
    # (This would go right after the red_chi2 calculation, around line 102)



    # Hide x-axis labels for residual plot (they'll show on main Rrs plot)
    ax_res.set_xticklabels([])

    # Update axes list to include residual axis:
    axes = [ax_anw, ax_bb, ax_R, ax_res]

    # #########################################################
    # a without water

    ax_anw.plot(wave, a_mean-a_w, 'b-', label='Retrieval')
    ax_anw.fill_between(wave, a_5-a_w, a_95-a_w, 
        color='b', alpha=0.5, label='Uncertainty') 

    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    ax_anw.set_yscale('log')

    # Parameters
    model = models[0]
    ypos = 0.1
    for ss in range(model.nparam):
        lsig = stats['med'][ss] - stats[f'p{perc[0]:02d}'][ss]
        hsig = stats[f'p{perc[1]:02d}'][ss] - stats['med'][ss]
        ax_anw.text(0.05, ypos, 
                  r''+f'{model.pnames[ss]} = {stats['med'][ss]:.3f}'+
                  r'$^{+'+f'{hsig:.3f}'+
                  r'}_{-'+f'{lsig:.3f}'+r'}$',
            transform=ax_anw.transAxes, fontsize=13.)
        ypos += 0.11

    # #########################################################
    # bb nw
    ax_bb.plot(wave, bb_mean-bb_w, 'g-', label='Retrieval')
    ax_bb.fill_between(wave, bb_5-bb_w, bb_95-bb_w,
            color='g', alpha=0.5, label='Uncertainty') 
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')
    #ax_bb.set_yscale('log')

    # Parameters
    model = models[1]
    ypos = 0.1
    for tt in range(model.nparam):
        ss = tt + models[0].nparam
        lsig = stats['med'][ss] - stats[f'p{perc[0]:02d}'][ss]
        hsig = stats[f'p{perc[1]:02d}'][ss] - stats['med'][ss]
        ax_bb.text(0.05, ypos, 
                  r''+f'{model.pnames[tt]} = {stats['med'][ss]:.3f}'+
                  r'$^{+'+f'{hsig:.3f}'+
                  r'}_{-'+f'{lsig:.3f}'+r'}$',
            transform=ax_bb.transAxes, fontsize=13.)
        ypos += 0.11

    # #########################################################
    # Rs
    
    # Calcualte chi^2
    Rsig=np.sqrt(Rrs_obs['var'])
    f = interp1d(wave, model_Rrs)
    mod_R = f(Rrs_obs['wave'])
    chi2 = np.sum((Rrs_obs['spec']-mod_R)**2 / Rsig**2)
    nparam = models[0].nparam + models[1].nparam
    red_chi2 = chi2 / (Rsig.size-nparam)

    if show_Rsig:
        ax_R.errorbar(Rrs_obs['wave'], Rrs_obs['spec'], 
            yerr=Rsig, color='gray', fmt='o', capsize=3,
            #label=r'Obs; $\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$',
            label='Obs',# $\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$',
            zorder=1) 
    ax_R.plot(Rrs_obs['wave'], Rrs_obs['spec'], 'k+', #label='Obs', 
              zorder=5)
    ax_R.plot(wave, model_Rrs, 'r-', label='Fit', zorder=10)
    ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
            color='r', alpha=0.5, zorder=10) 
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')
    #ax_R.set_yscale('log')
    ax_R.text(0.05, 0.1,
              r'$\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$',
              fontsize=15., transform=ax_R.transAxes)

    # Plot residuals
    residuals = (Rrs_obs['spec'] - mod_R) / Rsig  # Normalized residuals
    for y in [-2,0,2]:
        ls = ':' if y == 0 else '--'
        ax_res.axhline(y=y, color='gray', linestyle=ls, alpha=0.5)
    ax_res.scatter(Rrs_obs['wave'], residuals, color='k',
                   s=1.)
    ax_res.set_ylabel('Residuals\n(σ)', fontsize=12)
    ymx = 1.2*np.max(np.abs(residuals)) 
    ax_res.set_ylim([-ymx, ymx])
    ax_res.grid(True, alpha=0.3)
    ax_res.minorticks_on()

    fontsize = 15.
    plotting.set_fontsize(ax_res, fontsize)

    # axes
    axes = [ax_anw, ax_bb, ax_R]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    # Mini corner plot
    img = mpimg.imread('tmpc.png')
    ax_c.imshow(img)
    ax_c.axis('off') 

    # Title
    fig.suptitle(title, fontsize=14, y=0.99)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")
    else:
        plt.show()

def mini_corner(models, chains, show_params:list,
                outfile:str=None):
    """
    Mini corner plot of the model parameters.
    
    Parameters:
        models (list): List of model objects.
        chains (ndarray): Chains of model parameters.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
    """
    #if ax is None:
    #    fig, ax = plt.subplots(figsize=(8, 8))

    # Burn/thin the chains
    coeff = evaluate.thin_burn_chains(chains)

    # Grab the parameters to show
    keep = np.array([False]*coeff.shape[1])
    cnt = 0
    clbls = []
    for model in models:
        for param in model.pnames:
            if param in show_params:
                keep[cnt] = True
                clbls.append(param)
            cnt += 1

    # Cut
    coeff = coeff[:,keep]
    
    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        #truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")


def fit_em_all(match_file:str, clobber = False, nclosest:int=10, debug:bool=False):
    """
    Fit all the matches in the matched file.
    """
    matched = pandas.read_csv(match_file)

    for ss in range(len(matched)):
        #if ss < 798:
        #    continue
        imatched = matched.iloc[ss]
        print("*"*50)
        print("*"*50)
        print(f"Fitting {ss+1}/{len(matched)}...")
        print("*"*50)
        print("*"*50)

        if debug and ss > 0:
            break

        if debug:
            imatched.closest_file = 'PACE_OCI.20250601T070425.L2.OC_AOP.V3_1.nc'
            imatched.closest_id = 'PACE_OCI_L2_AOP_PACE_OCI.20250601T070425.L2.OC_AOP.V3_1.nc_3.1'

        # Check
        outfile = biomass_io.get_fit_file_path(imatched)
        if os.path.exists(outfile) and not clobber:
            print(f"Already fitted {outfile}, skipping...")
            continue

        # Fit one
        print(f"Fitting {imatched.cruise}-{imatched.profile:03d}...")
        fit_one(imatched, outfile, nclosest=nclosest)#, debug=True)

    print(f"Fitted {len(matched)} profiles")

# Command line
if __name__ == '__main__':

    test = True
    fit_em = False
    slurp_em = False

    #match_file = 'matched_argo_bgc_profiles_bbp.csv'
    match_file = 'matched_argo_bgc_profiles_bbp_v2.csv'
    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    #embed(header='556 of fitting.py')

    if test:
        # Load the matched file
        imatched = matched.iloc[0]
        imatched.closest_file = 'PACE_OCI.20250601T070425.L2.OC_AOP.V3_1.nc'
        imatched.closest_id = 'PACE_OCI_L2_AOP_PACE_OCI.20250601T070425.L2.OC_AOP.V3_1.nc_3.1'

        # Fit one
        outfile = biomass_io.get_fit_file_path(imatched)
        #fit_one(imatched, outfile, nclosest=10)
        fit_one(imatched, outfile, nclosest=2)#, debug=True)

    if fit_em:
        match_file='matched_argo_bgc_profiles_bbp.csv'
        fitting.fit_em_all(match_file, clobber=False, nclosest=10)


    if slurp_em:
        slurp_fits(matched)#debug=True)
