"""
Visualization Module for BING
==============================

This module provides plotting functions for visualizing bio-optical
parameter retrievals, MCMC diagnostics, and fit quality assessment.

Key plotting functions:
- show_fits: Multi-panel plot showing Rrs, absorption, and backscattering fits
- show_anw_fits: Detailed absorption component plots (a_dg, a_ph)
- corner_plot: MCMC posterior corner plots using the corner package
- hist2d: 2D histogram/contour plots for posterior visualization

All plots use STIX fonts for publication-quality figures and support
both MCMC chain inputs and least-squares fit results.

Examples
--------
>>> from bing import plotting
>>> from bing.models import utils as model_utils
>>>
>>> # Show fit results
>>> axes, model_Rrs = plotting.show_fits(
...     models, chains, rt_dict,
...     ex_a_params=Chl, ex_bb_params=Y,
...     Rrs_true={'wave': wave, 'spec': Rrs, 'var': varRrs},
...     perc=(5, 95))
>>>
>>> # Corner plot for MCMC diagnostics
>>> fig = plotting.corner_plot(chains, models=models)
"""
import numpy as np

from scipy.interpolate import interp1d
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from ocpy.water import absorption
from ocpy.water import scattering as w_scattering
from ocpy.utils import plotting

from bing import evaluate

from IPython import embed

# ############################################################
def log_param_mask(models:list, rt_dict:dict=None):
    """
    Which fitted parameters are log10 amplitudes, for display purposes.

    Models declare this as a ``log_params`` list of booleans; anything
    that does not (or that sets it to None) is treated as all-log10,
    which is the historical behaviour of this module.  Use it to decide
    which values to exponentiate before printing and which axis labels
    to wrap in log10(...).

    When ``rt_dict['fit_Bp']`` is True (M3 task 3, design §3.3) the
    fitted vector carries a trailing B_p element, whose entry here is
    **False**: B_p is a *linear-space* ratio (like the slopes), a
    deliberate plan choice among the log10 amplitudes -- it must be
    neither exponentiated nor log10-labelled.

    Parameters
    ----------
    models : list
        One or more model objects, in parameter order (typically
        [a_model, bb_model]).  A bare model is also accepted.
    rt_dict : dict, optional
        Radiative transfer configuration. Only 'fit_Bp' (default False)
        is consulted; None is the fixed-B_p default (mask over the model
        parameters only, exactly as before).

    Returns
    -------
    list of bool
        One entry per parameter, concatenated across the models, plus a
        trailing False for B_p when rt_dict['fit_Bp'] is True.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    mask = []
    for model in models:
        declared = getattr(model, 'log_params', None)
        if declared is None:
            mask += [True]*model.nparam
        else:
            mask += list(declared)
    # Free B_p rides as the trailing element of the fitted vector and is
    # linear-space by construction (its prior flavor is 'uniform').
    if rt_dict is not None and rt_dict.get('fit_Bp', False):
        mask.append(False)
    return mask


def show_fits(models:list, inputs:np.ndarray, rt_dict:dict,
             ex_a_params:np.ndarray, ex_bb_params:np.ndarray,
             outfile:str=None,
             figsize:tuple=(14,6),
             fontsize:float=12.,
             anw_true:dict=None,
             bbnw_true:dict=None,
             xqaa:dict=None,
             Rrs_true:dict=None,
             show_params:bool=False,
             perc:tuple=(5,95),
             log_Rrs:bool=True,
             show:bool=False,
             log_abb:bool=False,
             geom=None):
    """
    Create multi-panel figure showing Rrs, absorption, and backscattering fits.

    Generates a publication-quality figure with three panels:
    1. Rrs: Observed vs modeled remote sensing reflectance
    2. a_nw: Non-water absorption coefficient
    3. bb_nw: Non-water backscattering coefficient

    Supports both MCMC chains (with uncertainty bands) and least-squares
    results (point estimates only).

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    inputs : np.ndarray
        Either:
        - MCMC chains with shape (nsteps, nwalkers, nparam)
        - Least-squares parameters with shape (nparam,)
        The function auto-detects based on ndim.
    rt_dict : dict
        Radiative transfer configuration dictionary.
    ex_a_params : float or np.ndarray
        Extra parameters for absorption model (e.g., Chl for Bricaud).
    ex_bb_params : float or np.ndarray
        Extra parameters for backscattering model (e.g., Y for Lee).
    outfile : str, optional
        Path to save figure (PNG, PDF, etc.). If None, figure is not saved.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (14, 6).
    fontsize : float, optional
        Font size for axis labels and legends. Default is 12.
    anw_true : dict, optional
        True non-water absorption for comparison:
        {'wave': wavelengths, 'spec': a_nw values}
    bbnw_true : dict, optional
        True non-water backscattering for comparison:
        {'wave': wavelengths, 'spec': bb_nw values}
    xqaa : dict, optional
        XQAA algorithm results for comparison:
        {'wave': wavelengths, 'anw': a_nw, 'bbnw': bb_nw}
    Rrs_true : dict, optional
        Observed Rrs for comparison:
        {'wave': wavelengths, 'spec': Rrs, 'var': variance (optional)}
        If 'var' is provided, reduced chi-squared is computed and displayed.
    show_params : bool, optional
        Display fitted parameter values on the Rrs panel. Default is False.
    perc : tuple, optional
        Percentiles for uncertainty bands. Default is (5, 95).
    log_Rrs : bool, optional
        Use logarithmic y-axis for Rrs panel. Default is True.
    log_abb : bool, optional
        Use logarithmic y-axis for absorption/backscattering panels.
        Default is False.
    show : bool, optional
        Display the figure interactively. Default is False.
    geom : bing.rt.geometry.ObsGeometry, optional
        Fixed per-pixel viewing/illumination geometry, forwarded to
        evaluate.reconstruct_from_chains (MCMC inputs) or
        evaluate.reconstruct_chisq_fits (least-squares inputs) -- both
        paths thread it identically (PR #27, Bugbot finding 2). Required
        whenever rt_dict['rt_backend'] selects a robust backend; ignored
        by the default Gordon backend.

    Returns
    -------
    axes : list
        List of matplotlib axes objects [ax_anw, ax_bb, ax_Rrs].
    model_Rrs : np.ndarray
        Modeled Rrs values at model wavelengths.

    Notes
    -----
    For MCMC inputs, uncertainty bands show the credible interval defined
    by the perc parameter. For least-squares, only point estimates are shown.

    Water contributions (a_w, bb_w) are automatically subtracted to show
    non-water components.
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
        # geom rides along here too (PR #27, Bugbot finding 2): a robust-
        # backend LM fit is reconstructed through the same forward model
        # (and the same geometry) the fit itself used, exactly like the
        # MCMC branch below.
        model_Rrs, a_mean, bb_mean = evaluate.reconstruct_chisq_fits(
            models, params, rt_dict, Chl=ex_a_params,
            bb_basis_params=ex_bb_params, geom=geom)
            #d_chains['Chl'], bb_basis_params=d_chains['Y']) # Lee
    else:
        a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
            model_Rrs, sigRs = evaluate.reconstruct_from_chains(
            models, chains, rt_dict, perc=perc, geom=geom)
        # Generate params just in case
        params = np.median(chains, axis=[0,1])
        #embed(header='show_fit 70')

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    bb_w = w_scattering.bbw_from_l23(wave)
    
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
    ax_anw.plot(wave, a_mean-a_w, 'r-', label='Retrieval')

    if not use_LM:
        ax_anw.fill_between(wave, a_5-a_w, a_95-a_w, 
            color='r', alpha=0.5, label='Uncertainty') 

    if xqaa is not None:
        ax_anw.plot(xqaa['wave'], xqaa['anw'], ':', color='orange', label='XQAA')
    
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    if log_abb:
        ax_anw.set_yscale('log')

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

    if log_abb:
        ax_bb.set_yscale('log')

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
            red_chi2 = chi2 / (Rsig.size-nparam)
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
        # Only exponentiate the log10 amplitudes; slopes/exponents are
        # already linear and printing 10**beta is simply wrong.
        is_log = log_param_mask(models)
        for model in models:
            for ss in range(model.nparam):
                if is_log[ip]:
                    val = f'{10**params[ip]:.2f}'
                else:
                    val = f'{params[ip]:.3f}'
                ax_R.text(0.05, ypos, f'{model.pnames[ss]} = {val}',
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

    if show:
        plt.show()

    return axes, model_Rrs

def show_anw_fits(models:list, prep_chains:np.ndarray,
             outfile:str=None,
             figsize:tuple=(9,6),
             fontsize:float=12.,
             perc:tuple=(5,95),
             ax_anw=None,
             no_show:bool=False,
             adg_clr = 'blue', aph_clr = 'green',
             anw_true:dict=None):
    """
    Plot absorption component retrievals (a_dg and a_ph separately).

    Creates a detailed view of the absorption decomposition, showing
    dissolved/detrital (a_dg) and phytoplankton (a_ph) components with
    uncertainty bands from MCMC chains.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
        The absorption model must support retsub_comps=True in eval_anw().
    prep_chains : np.ndarray
        Pre-processed MCMC chains (after burn-in removal) with shape
        (nsamples, nparam). Use evaluate.thin_burn_chains() to prepare.
    outfile : str, optional
        Path to save figure. If None, figure is not saved.
    figsize : tuple, optional
        Figure size in inches. Default is (9, 6).
    fontsize : float, optional
        Font size for labels. Default is 12.
    perc : tuple, optional
        Percentiles for uncertainty bands. Default is (5, 95).
    ax_anw : matplotlib.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    no_show : bool, optional
        If True, suppress plt.show(). Default is False.
    adg_clr : str, optional
        Color for a_dg lines and bands. Default is 'blue'.
    aph_clr : str, optional
        Color for a_ph lines and bands. Default is 'green'.
    anw_true : dict, optional
        True absorption components for comparison:
        {'wave': wavelengths, 'a_dg': a_dg values, 'a_ph': a_ph values}

    Returns
    -------
    ax_anw : matplotlib.Axes
        The axes object with the plot.

    Notes
    -----
    Prints a_ph(440) retrieval statistics to stdout for quick validation.
    """
    # Unpack a little
    wave = models[0].wave

    # Calc
    a_dg, a_ph = models[0].eval_anw(prep_chains[..., :models[0].nparam], retsub_comps=True)
    adg_mean = np.median(a_dg, axis=0)
    adg_low, adg_high = np.percentile(a_dg, perc, axis=0)
    aph_mean = np.median(a_ph, axis=0)
    aph_low, aph_high = np.percentile(a_ph, perc, axis=0)

    # Stats
    i440 = np.argmin(np.abs(wave-440.))
    #print(f'Fit: a_dg(440) = {adg_mean[i440]:0.3f} +/- {0.5*(adg_high[i440]-adg_low[i440]):0.3f}')
    print(f'Fit: a_ph(440) = {aph_mean[i440]:0.4f} +/- {0.5*(aph_high[i440]-aph_low[i440]):0.4f}')
    if anw_true is not None:
        i440 = np.argmin(np.abs(anw_true['wave']-440.))
        print(f'True: a_ph(440) = {anw_true["a_ph"][i440]:0.4f}')

    # #########################################################
    # Plot the solution
    lgsz = 14.

    if ax_anw is None:
        fig = plt.figure(figsize=figsize)
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax_anw = plt.subplot(gs[0])
    

    # #########################################################
    # a without water
    if anw_true is not None:
        for clr, key, marker in zip(['b','g'], ['a_dg', 'a_ph'], ['o','s']):
            ax_anw.plot(anw_true['wave'], 
                    anw_true[key], marker, color=clr, 
                    label=f'True {key}', zorder=1)
    # 
    ax_anw.plot(wave, adg_mean, '-', color=adg_clr, label='a_dg Retrieval')
    ax_anw.fill_between(wave, adg_low, adg_high, color=adg_clr, alpha=0.5) 
    ax_anw.plot(wave, aph_mean, '-', color=aph_clr, label='a_ph Retrieval')
    ax_anw.fill_between(wave, aph_low, aph_high, color=aph_clr, alpha=0.5) 

    ax_anw.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')

    # axes
    axes = [ax_anw]
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, fontsize)
        ax.set_xlabel('Wavelength (nm)')
        ax.legend(fontsize=15.)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")
    elif not no_show:
        plt.show()

    return ax_anw

def corner_plot(chains, models:list=None,
           outfile:str=None,
           show:bool=True, show_log:bool=True,
           rt_dict:dict=None):
    """
    Create corner plot showing MCMC posterior distributions.

    Generates a corner plot (triangle plot) showing 1D and 2D marginal
    posterior distributions for all model parameters. Uses the corner
    package with customizations for BING.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains with shape (nsteps, nwalkers, nparam). Under
        ``rt_dict['fit_Bp']`` (M3 task 3, design §3.3) nparam includes
        the trailing B_p column.
    models : list, optional
        List of two model objects for parameter naming. If None, generic
        labels are used.
    outfile : str, optional
        Path to save figure. If None, figure is not saved.
    show : bool, optional
        Display the figure interactively. Default is True.
    show_log : bool, optional
        Show parameters in log10 space (as fitted). If False, converts
        to linear space. Default is True.
    rt_dict : dict, optional
        Radiative transfer configuration. Only 'fit_Bp' (default False)
        is consulted: when True, the trailing chain column is labeled
        ``'B_p'`` and treated as linear-space (never log10-wrapped or
        exponentiated -- its ``log_params`` semantics are False). None
        is the fixed-B_p default, exactly the pre-M3 behavior.

    Returns
    -------
    fig : matplotlib.Figure
        The corner plot figure object.

    Notes
    -----
    - Burn-in of 7000 steps is applied internally
    - 90% credible intervals (5th and 95th percentiles) are shown as
      vertical dashed lines on the 1D histograms
    - Parameter labels include log10 notation when show_log=True
    """
    # Init the models
    #models = model_utils.init(p.model_names, d_chains['wave'])

    burn = 7000
    thin = 1
    coeff = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])

    # Which columns are log10 amplitudes?  Slopes/exponents are linear,
    # so they must be neither exponentiated nor log10-labelled -- and so
    # is a free B_p tail (rt_dict['fit_Bp']; log_param_mask appends its
    # False entry).
    if models is not None:
        is_log = log_param_mask(models, rt_dict=rt_dict)
    else:
        is_log = [True]*coeff.shape[-1]
        if rt_dict is not None and rt_dict.get('fit_Bp', False):
            is_log[-1] = False

    if not show_log:
        coeff = coeff.copy()
        for kk, kk_log in enumerate(is_log[:coeff.shape[-1]]):
            if kk_log:
                coeff[:, kk] = 10**coeff[:, kk]

    truths = None

    # Labels
    if models is not None:
        clbls = evaluate.chain_param_names(models, rt_dict=rt_dict)
        # Add log 10, but only for the parameters that are in log10
        clbls = [r'$\log_{10}('+f'{clbl}'+r')$' if (kk_log and show_log)
                 else clbl
                 for clbl, kk_log in zip(clbls, is_log)]
    else:
        clbls = None

    # Replace Aph with Cph
    #for ss, clbl in enumerate(clbls):
    #    if 'Aph' in clbl:
    #        clbls[ss] = clbl.replace('Aph', 'Cph')
    #embed(header='figs 407')

    if show_log and truths is not None:
        truths = np.log10(truths)

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    # Add 95%
    ss = 0
    for ax in fig.get_axes():
        if len(ax.get_title()) > 0:
            # Calculate the percntile
            p_5, p_95 = np.percentile(coeff[:,ss], [5, 95], axis=0)
            # Plot a vertical line
            ax.axvline(p_5, color='b', linestyle=':')
            ax.axvline(p_95, color='b', linestyle=':')
            ss += 1



    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")

    if show:
        plt.show()

    return fig

def _set_xlim(force, new_fig, ax, new_xlim):
    if force or new_fig:
        return ax.set_xlim(new_xlim)
    xlim = ax.get_xlim()
    return ax.set_xlim([min(xlim[0], new_xlim[0]), max(xlim[1], new_xlim[1])])


def _set_ylim(force, new_fig, ax, new_ylim):
    if force or new_fig:
        return ax.set_ylim(new_ylim)
    ylim = ax.get_ylim()
    return ax.set_ylim([min(ylim[0], new_ylim[0]), max(ylim[1], new_ylim[1])])


def hist2d(
    x,
    y,
    bins=20,
    range=None,
    axes_scale=["linear", "linear"],
    weights=None,
    levels=None,
    smooth=None,
    ax=None,
    color=None,
    quiet=False,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    no_fill_contours=False,
    fill_contours=False,
    contour_kwargs=None,
    contourf_kwargs=None,
    data_kwargs=None,
    pcolor_kwargs=None,
    new_fig=True,
    force_range=False,
    **kwargs,
):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    y : array_like[nsamples,]
       The samples.

    axes_scale : iterable (2,)
        Scale (``"linear"``, ``"log"``) to use for each dimension.

    quiet : bool
        If true, suppress warnings for small datasets.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool
        Draw the individual data points.

    plot_density : bool
        Draw the density colormap.

    plot_contours : bool
        Draw the contours.

    no_fill_contours : bool
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool
        Fill the contours.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    pcolor_kwargs : dict
        Any additional keyword arguments to pass to the `pcolor` method when
        adding the density colormap.

    bins : int or [int, int], optional
        The number of bins for the histogram in each dimension (default: 40).
        Can be an integer or a tuple/list of two integers.
    range : [[float, float], [float, float]], optional
        The range of values for the x and y axes. If None, uses the data range.
    weights : array-like, optional
        An array of weights for each sample. Default is None (no weighting).
    smooth : float, optional
        Standard deviation for Gaussian smoothing of the density. Default is 0 (no smoothing).
    color : str or tuple, optional
        Color for the data points and contours. Default is matplotlib's ytick color.
    new_fig : bool, optional
        If True, creates a new figure. Default is False.
    force_range : bool, optional
        If True, forces the use of the specified range even if data is outside. Default is False.
    """
    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            #logging.warning(
            #    "Deprecated keyword argument 'extent'. Use 'range' instead."
            #)
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = matplotlib.rcParams["ytick.color"]

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the base color of the axis (background color)
    base_color = ax.get_facecolor()

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, colorConverter.to_rgba(base_color, alpha=0.0)]
    )

    # This color map is used to hide the points at the high density areas.
    base_cmap = LinearSegmentedColormap.from_list(
        "base_cmap", [base_color, base_color], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")

    # We'll make the 2D histogram to directly estimate the density.
    bins_2d = []
    if axes_scale[0] == "linear":
        bins_2d.append(np.linspace(min(range[0]), max(range[0]), bins[0] + 1))
    elif axes_scale[0] == "log":
        bins_2d.append(
            np.logspace(
                np.log10(min(range[0])),
                np.log10(max(range[0])),
                bins[0] + 1,
            )
        )

    if axes_scale[1] == "linear":
        bins_2d.append(np.linspace(min(range[1]), max(range[1]), bins[1] + 1))
    elif axes_scale[1] == "log":
        bins_2d.append(
            np.logspace(
                np.log10(min(range[1])),
                np.log10(max(range[1])),
                bins[1] + 1,
            )
        )

    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins_2d,
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )
    if H.sum() == 0:
        raise ValueError(
            "It looks like the provided 'range' is not valid "
            "or the sample is empty."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    if plot_contours or plot_density:
        # Compute the density levels.
        Hflat = H.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]
        V = np.empty(len(levels))
        for i, v0 in enumerate(levels):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except IndexError:
                V[i] = Hflat[0]
        V.sort()
        m = np.diff(V) == 0
        while np.any(m):
            V[np.where(m)[0][0]] *= 1.0 - 1e-4
            m = np.diff(V) == 0
        V.sort()

        # Compute the bin centers.
        X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

        # Extend the array for the sake of the contours at the plot edges.
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate(
            [
                X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
                X1,
                X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
            ]
        )
        Y2 = np.concatenate(
            [
                Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
                Y1,
                Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
            ]
        )

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(
            X2,
            Y2,
            H2.T,
            [V.min(), H.max()],
            cmap=base_cmap,
            antialiased=False,
        )

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get(
            "antialiased", False
        )
        ax.contourf(
            X2,
            Y2,
            H2.T,
            np.concatenate([[0], V, [H.max() * (1 + 1e-4)]]),
            **contourf_kwargs,
        )

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        if pcolor_kwargs is None:
            pcolor_kwargs = dict()
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap, **pcolor_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    _set_xlim(force_range, new_fig, ax, range[0])
    _set_ylim(force_range, new_fig, ax, range[1])
    ax.set_xscale(axes_scale[0])
    ax.set_yscale(axes_scale[1])

