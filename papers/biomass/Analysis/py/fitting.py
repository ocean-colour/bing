
import os
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
mpl.rcParams['font.family'] = 'stixgeneral'

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

def fit_one(imatched:pandas.Series, outfile:str, debug:bool=False):

    # Load PACE file
    gfile = os.path.join(os.getenv('OS_COLOR'), 'PACE', 'L2_AOP', 
                     imatched.closest_file)
    print(f"----- Loading {gfile} -----")
    xds, flags = pace_io.load_oci_l2(gfile)

    # Find closest Rrs
    d_min, dmin_ij = closest_Rrs(xds, (imatched.lat, imatched.lon))

    if debug:
        embed(header='44 of fitting.py')

    # Parse out the data
    ix, iy = dmin_ij
    gd_wave = (xds.wavelength.data >= 400.) &  (xds.wavelength.data <= 700.) 
    iwave = xds.wavelength.data[gd_wave]
    ispec = xds.Rrs.data[ix,iy,gd_wave]
    isig = xds.Rrs_unc.data[ix,iy,gd_wave]

    # Init models
    p = standard.expb_pow()
    models = model_utils.init(p.model_names, iwave)
    bing_priors.set_standard_priors(models, p)
    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)

    # Fit with LM for first guess
    low_bounds, high_bounds = [], []
    low_bounds += [item['pmin'] for item in p.apriors]
    low_bounds += [item['pmin'] for item in p.bpriors]
    high_bounds += [item['pmax'] for item in p.apriors]
    high_bounds += [item['pmax'] for item in p.bpriors]
    #
    bounds = (np.array(low_bounds), np.array(high_bounds))

    p0 = [-1, 0.015, -1, -1, 1.5]
    items = [(ispec, isig**2, p0, 0)]
    ans, cov, idx = chisq_fit.fit(items[0], models, bounds=bounds)

    # Now the MCMC
    p0 = ans.tolist()
    items = [(ispec, isig**2, p0, 0)]
    pdict['Chl'] = np.array([10**p0[2] / 0.05582])
    pdict['Y'] = None

    print("----- Fitting with MCMC -----")
    chains, idx = bing_inf.fit_one(
        items[0], models=models, pdict=pdict, chains_only=True)
    stats = evaluate.calc_stats(chains)

    # Save
    out_dict = {}
    out_dict['chains'] = chains
    out_dict['LM'] = ans
    out_dict['wave'] = iwave
    out_dict['Rrs'] = ispec
    out_dict['Rrs_sig'] = isig
    out_dict['Rrs_idx'] = np.array(dmin_ij) # ij in xds
    #
    out_dict['med'] = stats['med']
    out_dict['p05'] = stats['p05']
    out_dict['p95'] = stats['p95']
    out_dict['model_names'] = [model.name for model in models]

    np.savez(outfile, **out_dict)
    print(f"Saved: {outfile}")

    # Plot me
    print("----- Plotting -----")
    title = f'Float={imatched.cruise}-{imatched.profile}, lat={imatched.lat},'+\
    f'lon={imatched.lon}, time={imatched.time[:19]}, {imatched.closest_id[12:-9]}'
    Rrs_obs=dict(wave=models[0].wave, spec=ispec, var=isig**2)
    plotfile=outfile.replace('.npz', '.png')
    plot_fit(models, chains, Rrs_obs, title, show_Rsig=True,
                   outfile=plotfile)

    

def plot_fit(models, chains, Rrs_obs, title:str, stats:dict=None,
             outfile:str=None, 
             ulist:list=None, perc:tuple=(5,95),
             show_Rsig:bool=False):

    # Do this first
    mini_corner(models, chains, ['Sdg', 'beta', 'Bnw'],
                outfile='tmpc.png')

    if stats is None:
        stats = evaluate.calc_stats(chains)

    # Wavelengths
    wave = models[0].wave

    # Unpack for development
    if ulist is not None:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = ulist
    else:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains, perc=perc)

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    bb_w = w_scattering.bbw_from_l23(wave)

    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[1])
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
    ax_bb = plt.subplot(gs[2])
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
    ax_R = plt.subplot(gs[0])
    
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

    # axes
    axes = [ax_anw, ax_bb, ax_R]
    fontsize = 15.
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    # Mini corner plot
    ax_c = plt.subplot(gs[3])
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

def slurp_fits():

    # Load up Argo profiles, already matched to PACE
    match_file = 'matched_argo_bgc_profiles_bbp.csv'
    matched = pandas.read_csv(match_file)

    beta_vals = []
    Bnw_vals = []
    aph_vals = []

    for ss in range(len(matched)):
        imatched = matched.iloc[ss]
        outfile = set_outfile(imatched)

        # Load
        if not os.path.exists(outfile):
            embed(header=f"303: Missing {outfile}...")
        d = np.load(outfile)
        Bnw_vals.append(d['med'][3])
        beta_vals.append(d['med'][4])
        aph_vals.append(d['med'][2])

    # Add to matched
    matched['Bnw'] = 10**np.array(Bnw_vals)
    matched['beta'] = beta_vals
    matched['aph'] = aph_vals

    # Write
    matched.to_csv(match_file, index=False)
    print(f'Wrote {len(matched)} profiles to {match_file}')

def set_outfile(imatched:pandas.Series):
    outfile = os.path.join(os.getenv('OS_COLOR'), 'Biomass', 'Fits',
            f'Argo_{imatched.cruise}_{imatched.profile:03d}_fits.npz')
    return outfile

# Command line
if __name__ == '__main__':

    test = False
    fit_em = True
    slurp_em = False

    match_file = 'matched_argo_bgc_profiles_bbp.csv'
    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

    if test:
        # Load the matched file
        imatched = matched.iloc[30]

        # Fit one
        outfile = set_outfile(imatched)
        fit_one(imatched, outfile)#, debug=True)

    if fit_em:
        clobber = False
        for ss in range(len(matched)):
            imatched = matched.iloc[ss]
            print(f"Fitting {ss+1}/{len(matched)}...")

            # Check
            outfile = set_outfile(imatched)
            if os.path.exists(outfile) and not clobber:
                print(f"Already fitted {outfile}, skipping...")
            #

            # Fit one
            print(f"Fitting {imatched.cruise}-{imatched.profile:03d}...")
            fit_one(imatched, outfile)

    if slurp_em:
        slurp_fits()