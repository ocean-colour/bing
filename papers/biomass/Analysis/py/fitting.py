
import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from ocpy.water import absorption
from ocpy.water import scattering as w_scattering
from ocpy.utils import plotting

from bing import evaluate


def plot_fit(models, chains, Rrs_obs, stats:dict=None,
             outfile:str=None, 
             ulist:list=None, perc:tuple=(5,95),
             show_Rsig:bool=False):

    # Do this first
    mini_corner(models, chains, ['Sdg', 'beta', 'Bnw'],
                outfile='tmpc.png')

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
    ax_bb.set_yscale('log')

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
