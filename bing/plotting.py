""" Routines for plotting """
import numpy as np

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

from ocpy.water import absorption
from ocpy.hydrolight import loisel23
from ocpy.utils import plotting

from bing import chisq_fit
from bing import evaluate

from IPython import embed

# ############################################################
def show_fits(models:list, inputs:np.ndarray,
             ex_a_params:np.ndarray, ex_bb_params:np.ndarray,
             outfile:str=None,
             figsize:tuple=(14,6),
             fontsize:float=12.,
             anw_true:dict=None, 
             bbnw_true:dict=None,
             xqaa:dict=None,
             Rrs_true:dict=None,
             show_params:bool=False,
             log_Rrs:bool=True):
    """
    Plots the fit results for the given models and inputs.

    Parameters:
        models (list): A list of models.
        inputs (np.ndarray): The input data for the models.
            ans: The optimized parameters for the curve fitting.
            or 
            chains: The MCMC chains.
        outfile (str, optional): The path to save the plot as an image file. Default is None.
        figsize (tuple, optional): The size of the figure. Default is (14, 6).
        fontsize (float, optional): The font size of the plot labels. Default is 12.0.
        anw_true (dict, optional): The true values for `a_nw`. Default is None.
        bbnw_true (dict, optional): The true values for `b_bnw`. Default is None.
        Rrs_true (dict, optional): 
            The true values for `R_rs`. Default is None.
            wave: Wavelength values
            spec: 
        show_params (bool, optional): Whether to show the parameters. Default is False.
        log_Rrs (bool, optional): Whether to use a logarithmic scale for the y-axis of `R_rs`. Default is True.

    Returns:
        axes (list): A list of the axes objects used in the plot.
    """
    # Unpack a little
    wave = models[0].wave

    if inputs.ndim == 1:
        use_LM = True
        params = inputs
    else:
        use_LM = False
        chains = inputs

    # Reconstruc
    if use_LM:
        model_Rrs, a_mean, bb_mean = evaluate.reconstruct_chisq_fits(
            models, params, Chl=ex_a_params, bb_basis_params=ex_bb_params)
            #d_chains['Chl'], bb_basis_params=d_chains['Y']) # Lee
    else:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains)
        # Generate params just in case
        params = np.median(chains, axis=[0,1])
        #embed(header='show_fit 70')

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    # TODO -- FIX THIS!
    # THIS IS A HACK UNTIL I CAN RESOLVE bbw
    ds = loisel23.load_ds(4,0)
    l23_wave = ds.Lambda.data
    idx = 170 # Random choie
    l23_bb = ds.bb.data[idx] 
    l23_bbnw = ds.bbnw.data[idx] 
    l23_bbw = l23_bb - l23_bbnw
    # Interpolate
    bb_w = np.interp(wave, l23_wave, l23_bbw)

    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,3)
    


    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[1])
    if anw_true is not None:
        ax_anw.plot(anw_true['wave'], anw_true['spec'], 'ko', label='True', zorder=1)
    ax_anw.plot(wave, a_mean-a_w, 'r-', label='Retreival')

    if not use_LM:
        ax_anw.fill_between(wave, a_5-a_w, a_95-a_w, 
            color='r', alpha=0.5, label='Uncertainty') 

    if xqaa is not None:
        ax_anw.plot(xqaa['wave'], xqaa['anw'], ':', color='orange', label='XQAA')
    
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    #ax_anw.plot(wave_true, adg, '-', color='brown', label=r'$a_{\rm dg}$')
    #ax_anw.plot(wave_true, aph, 'b-', label=r'$a_{\rm ph}$')

    #if set_abblim:
    #    ax_anw.set_ylim(bottom=0., top=2*(a_true-aw).max())
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels


    # #########################################################
    # bb nw
    ax_bb = plt.subplot(gs[2])
    if bbnw_true is not None:
        ax_bb.plot(bbnw_true['wave'], bbnw_true['spec'], 'ko', label='True', zorder=1)
    ax_bb.plot(wave, bb_mean-bb_w, 'g-', label='Retrieval')
    if not use_LM:
        ax_bb.fill_between(wave, bb_5-bb_w, bb_95-bb_w,
            color='g', alpha=0.5, label='Uncertainty') 

    if xqaa is not None:
        ax_bb.plot(xqaa['wave'], xqaa['bbnw'], ':', color='orange', label='XQAA')

    #ax_bb.set_xlabel('Wavelength (nm)')
    ax_bb.set_ylabel(r'$b_{b,nw}(\lambda) \; [{\rm m}^{-1}]$')

    #if set_abblim:
    #    ax_bb.set_ylim(bottom=0., top=2*show_bb.max())


    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0])
    if Rrs_true is not None:
        if 'var' in Rrs_true.keys():
            # Calcualte chi^2
            Rsig=np.sqrt(Rrs_true['var'])
            f = interp1d(wave, model_Rrs)
            mod_R = f(Rrs_true['wave'])
            chi2 = np.sum((Rrs_true['spec']-mod_R)**2 / Rsig**2)
            nparam = models[0].nparam + models[1].nparam
            red_chi2 = chi2 / (nparam-1)
            #
            ax_R.errorbar(Rrs_true['wave'], Rrs_true['spec'], 
                yerr=Rsig, color='k', fmt='o', capsize=5,
                label=r'$\chi^2_\nu = '+f'{red_chi2:0.2f}'+r'$') 
        else:
            ax_R.plot(Rrs_true['wave'], Rrs_true['spec'], 'k+', label='True', zorder=1)
    #ax_R.plot(wave, gordon_Rrs, 'k+', label='L23 + Gordon')
    ax_R.plot(wave, model_Rrs, 'b-', label='Fit', zorder=10)
    if not use_LM:
        ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
            color='b', alpha=0.5, zorder=10) 

    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')

    # Show params?
    if show_params:
        ypos = 0.05
        ip = 0
        for model in models:
            for ss in range(model.nparam):
                ax_R.text(0.05, ypos, f'{model.pnames[ss]} = {10**params[ip]:.2f}',
                    transform=ax_R.transAxes, fontsize=13.)
                ypos += 0.07
                ip += 1
    
    # Log scale y-axis
    if log_Rrs:
        ax_R.set_yscale('log')
    else:
        raise ValueError("Not ready for linear scale yet")
        #ax_R.set_ylim(bottom=0., top=1.1*Rrs_true.max())
    
    # axes
    axes = [ax_anw, ax_bb, ax_R]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")

    return axes

def show_abs_fit(models:list, chains:np.ndarray,
             ex_a_params:np.ndarray, ex_bb_params:np.ndarray,
             outfile:str=None,
             figsize:tuple=(14,6),
             fontsize:float=12.,
             burn:int=7000,
             thin:int=1,
             anw_true:dict=None, 
             bbnw_true:dict=None,
             xqaa:dict=None,
             Rrs_true:dict=None,
             show_params:bool=False,
             log_Rrs:bool=True):

    # Unpack a little
    wave = models[0].wave

    # Burn/thin the chains
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    # Calc
    a = models[0].eval_a(chains[..., :models[0].nparam])

    # Reconstruct
    a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains)
    # Generate params just in case
    params = np.median(chains, axis=[0,1])

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')

    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    

    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[0])

    if anw_true is not None:
        for clr, key in zip(['b','g'], ['a_dg', 'a_ph']):
            ax_anw.plot(anw_true['wave'], 
                    anw_true[key], 'o', color=clr, 
                    label='True adg', zorder=1)
    # 
    ax_anw.plot(wave, a_mean-a_w, 'r-', label='Retreival')
    ax_anw.fill_between(wave, a_5-a_w, a_95-a_w, 
        color='r', alpha=0.5, label='Uncertainty') 

    if xqaa is not None:
        ax_anw.plot(xqaa['wave'], xqaa['anw'], ':', color='orange', label='XQAA')
    
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
