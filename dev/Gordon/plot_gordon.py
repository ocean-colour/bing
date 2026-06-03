"""
Plotting helpers for the variable-Gordon-coefficient assessment.

All figure-generation methods used by ``calc_gordon.py`` live here. They were
factored out of ``calc_gordon.py`` once that module grew too large to navigate
comfortably. The fitting / evaluation methods stay in ``calc_gordon.py``;
this module imports the model evaluators and constants it needs from there.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence

from calc_gordon import (
    rrs_model,
    rrs_model_const,
    rrs_model_bbp,
    calc_u,
    Rrs_to_rrs,
    G1_STANDARD,
    G2_STANDARD,
)


# =============================================================================
# Coefficient-vs-wavelength
# =============================================================================

def plot_g_coefficients(
    result: Dict,
    outfile: Optional[str] = None,
    compare: Optional[Dict] = None,
    compare_label: str = 'old',
):
    """
    Plot fitted G1(λ), G2(λ) with optional comparison curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, key, std in zip(axes, ('G1', 'G2'), (G1_STANDARD, G2_STANDARD)):
        ax.plot(result['wavelength'], result[key], 'C3-', lw=2, label='new fit')
        if compare is not None:
            ax.plot(compare['wavelength'], compare[key], 'C0--', lw=1.6, label=compare_label)
        ax.axhline(std, color='k', ls=':', lw=1, label=f'standard {key}={std}')
        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_title('G1(λ)')
    axes[1].set_title('G2(λ)')
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


# =============================================================================
# rRMS-vs-wavelength
# =============================================================================

def plot_rrms_vs_wavelength(eval_result: Dict, outfile: Optional[str] = None):
    """
    Plot per-wavelength relative-RMS error for variable vs standard Gordon.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(eval_result['wavelength'], eval_result['rrms_std_pct'],
            'C0-',  lw=2, label='standard Gordon')
    ax.plot(eval_result['wavelength'], eval_result['rrms_var_pct'],
            'C3-', lw=2, label='variable Gordon')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('rRMS  [%]')
    ax.set_title('Rrs reconstruction error vs Hydrolight')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_rrms_vs_wavelength_3case(
    eval_no_G0: Dict,
    eval_with_G0: Dict,
    outfile: Optional[str] = None,
):
    """
    Three-curve rRMS-vs-wavelength figure: standard, variable (no G0), variable (with G0).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_std_pct'],
            'C0-', lw=2, label='standard Gordon')
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_var_pct'],
            'C3-', lw=2, label='variable, no G0')
    ax.plot(eval_with_G0['wavelength'], eval_with_G0['rrms_var_pct'],
            'C2-', lw=2, label='variable, with G0')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('rRMS  [%]')
    ax.set_title('Rrs reconstruction error vs Hydrolight')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_rrms_vs_wavelength_4case(
    eval_no_G0: Dict,
    eval_with_G0: Dict,
    eval_with_Gb: Dict,
    outfile: Optional[str] = None,
):
    """
    Four-curve rRMS-vs-wavelength figure: standard, variable noG0, variable
    with G0, variable with Gb.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_std_pct'],
            'C0-', lw=2, label='standard Gordon')
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_var_pct'],
            'C3-', lw=2, label='variable, no G0')
    ax.plot(eval_with_G0['wavelength'], eval_with_G0['rrms_var_pct'],
            'C2-', lw=2, label='variable, with G0')
    ax.plot(eval_with_Gb['wavelength'], eval_with_Gb['rrms_var_pct'],
            'C4-', lw=2, label='variable, with Gb (bbp)')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('rRMS  [%]')
    ax.set_title('Rrs reconstruction error vs Hydrolight')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_rrms_vs_wavelength_5case(
    eval_no_G0: Dict,
    eval_with_G0: Dict,
    eval_with_Gb: Dict,
    eval_full: Dict,
    outfile: Optional[str] = None,
):
    """
    Five-curve rRMS-vs-wavelength figure: standard, no G0, with G0, with Gb,
    and the 4-parameter (G0+Gb) full fit.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_std_pct'],
            'C0-', lw=2, label='standard Gordon')
    ax.plot(eval_no_G0['wavelength'], eval_no_G0['rrms_var_pct'],
            'C3-', lw=2, label='variable, no G0')
    ax.plot(eval_with_G0['wavelength'], eval_with_G0['rrms_var_pct'],
            'C2-', lw=2, label='variable, with G0')
    ax.plot(eval_with_Gb['wavelength'], eval_with_Gb['rrms_var_pct'],
            'C4-', lw=2, label='variable, with Gb (bbp)')
    ax.plot(eval_full['wavelength'], eval_full['rrms_var_pct'],
            'C5-', lw=2.4, label='variable, G0 + Gb (4-param)')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('rRMS  [%]')
    ax.set_title('Rrs reconstruction error vs Hydrolight')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


# =============================================================================
# Residual-vs-bbp
# =============================================================================

def plot_residual_vs_bbp(
    wave: np.ndarray,
    Rrs_truth: np.ndarray,
    Rrs_var: np.ndarray,
    Rrs_std: np.ndarray,
    bbp: np.ndarray,
    plot_waves: Sequence[float] = (400., 500., 550., 600., 650., 700.),
    mask: Optional[np.ndarray] = None,
    outfile: Optional[str] = None,
    relative: bool = True,
):
    """
    Plot the Hydrolight - Gordon Rrs residual against bbp at selected wavelengths.
    """
    sel = np.ones(Rrs_truth.shape[0], dtype=bool) if mask is None else mask
    n = len(plot_waves)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.ravel()

    eps = 1e-12
    for ax, wv in zip(axes, plot_waves):
        j = int(np.argmin(np.abs(wave - wv)))
        x = bbp[sel, j]
        if relative:
            denom = np.maximum(np.abs(Rrs_truth[sel, j]), eps)
            y_std = 100.0 * (Rrs_truth[sel, j] - Rrs_std[sel, j]) / denom
            y_var = 100.0 * (Rrs_truth[sel, j] - Rrs_var[sel, j]) / denom
            ylabel = r'$(R_{rs}^{HL} - R_{rs}^{G})/R_{rs}^{HL}$  [%]'
        else:
            y_std = Rrs_truth[sel, j] - Rrs_std[sel, j]
            y_var = Rrs_truth[sel, j] - Rrs_var[sel, j]
            ylabel = r'$R_{rs}^{HL} - R_{rs}^{G}$  [sr$^{-1}$]'

        ax.scatter(x, y_std, s=10, alpha=0.5, color='C0', label='standard')
        ax.scatter(x, y_var, s=10, alpha=0.7, color='C3', label='variable')
        ax.axhline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel(r'$b_{bp}(\lambda)$  [m$^{-1}$]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{wv:.0f} nm')
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=9, loc='best')

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_residual_vs_bbp_3case(
    wave: np.ndarray,
    Rrs_truth: np.ndarray,
    Rrs_var_noG0: np.ndarray,
    Rrs_var_withG0: np.ndarray,
    Rrs_std: np.ndarray,
    bbp: np.ndarray,
    plot_waves: Sequence[float] = (400., 500., 550., 600., 650., 700.),
    mask: Optional[np.ndarray] = None,
    outfile: Optional[str] = None,
):
    """
    Three-case residual-vs-bbp panels (standard / variable noG0 / variable withG0).
    """
    sel = np.ones(Rrs_truth.shape[0], dtype=bool) if mask is None else mask
    n = len(plot_waves); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.ravel()

    eps = 1e-12
    for ax, wv in zip(axes, plot_waves):
        j = int(np.argmin(np.abs(wave - wv)))
        x = bbp[sel, j]
        denom = np.maximum(np.abs(Rrs_truth[sel, j]), eps)
        y_std    = 100.0 * (Rrs_truth[sel, j] - Rrs_std[sel, j]) / denom
        y_noG0   = 100.0 * (Rrs_truth[sel, j] - Rrs_var_noG0[sel, j]) / denom
        y_withG0 = 100.0 * (Rrs_truth[sel, j] - Rrs_var_withG0[sel, j]) / denom

        ax.scatter(x, y_std,    s=8, alpha=0.4, color='C0', label='standard')
        ax.scatter(x, y_noG0,   s=8, alpha=0.5, color='C3', label='variable, no G0')
        ax.scatter(x, y_withG0, s=8, alpha=0.7, color='C2', label='variable, with G0')
        ax.axhline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel(r'$b_{bp}(\lambda)$  [m$^{-1}$]')
        ax.set_ylabel(r'$(R_{rs}^{HL} - R_{rs}^{G})/R_{rs}^{HL}$  [%]')
        ax.set_title(f'{wv:.0f} nm')
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=9, loc='best')
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_residual_vs_bbp_4case(
    wave: np.ndarray,
    Rrs_truth: np.ndarray,
    Rrs_var_noG0: np.ndarray,
    Rrs_var_withG0: np.ndarray,
    Rrs_var_withGb: np.ndarray,
    Rrs_std: np.ndarray,
    bbp: np.ndarray,
    plot_waves: Sequence[float] = (400., 500., 550., 600., 650., 700.),
    mask: Optional[np.ndarray] = None,
    outfile: Optional[str] = None,
):
    """
    Four-case residual-vs-bbp panels: standard / variable noG0 / variable
    withG0 / variable withGb.
    """
    sel = np.ones(Rrs_truth.shape[0], dtype=bool) if mask is None else mask
    n = len(plot_waves); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.ravel()

    eps = 1e-12
    for ax, wv in zip(axes, plot_waves):
        j = int(np.argmin(np.abs(wave - wv)))
        x = bbp[sel, j]
        denom = np.maximum(np.abs(Rrs_truth[sel, j]), eps)
        y_std    = 100.0 * (Rrs_truth[sel, j] - Rrs_std[sel, j])    / denom
        y_noG0   = 100.0 * (Rrs_truth[sel, j] - Rrs_var_noG0[sel, j])   / denom
        y_withG0 = 100.0 * (Rrs_truth[sel, j] - Rrs_var_withG0[sel, j]) / denom
        y_withGb = 100.0 * (Rrs_truth[sel, j] - Rrs_var_withGb[sel, j]) / denom

        ax.scatter(x, y_std,    s=7, alpha=0.35, color='C0', label='standard')
        ax.scatter(x, y_noG0,   s=7, alpha=0.45, color='C3', label='variable, no G0')
        ax.scatter(x, y_withG0, s=7, alpha=0.55, color='C2', label='variable, with G0')
        ax.scatter(x, y_withGb, s=7, alpha=0.65, color='C4', label='variable, with Gb')
        ax.axhline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel(r'$b_{bp}(\lambda)$  [m$^{-1}$]')
        ax.set_ylabel(r'$(R_{rs}^{HL} - R_{rs}^{G})/R_{rs}^{HL}$  [%]')
        ax.set_title(f'{wv:.0f} nm')
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


def plot_residual_vs_bbp_5case(
    wave: np.ndarray,
    Rrs_truth: np.ndarray,
    Rrs_var_noG0: np.ndarray,
    Rrs_var_withG0: np.ndarray,
    Rrs_var_withGb: np.ndarray,
    Rrs_var_full: np.ndarray,
    Rrs_std: np.ndarray,
    bbp: np.ndarray,
    plot_waves: Sequence[float] = (400., 450., 500., 520., 550., 600., 650., 700.),
    bbp_wave: float = 700.,
    mask: Optional[np.ndarray] = None,
    outfile: Optional[str] = None,
):
    """
    Five-case residual-vs-bbp panels: adds the 4-parameter (G0 + Gb) fit on
    top of the existing 4-case plot.
    """
    sel = np.ones(Rrs_truth.shape[0], dtype=bool) if mask is None else mask
    n = len(plot_waves); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.ravel()
    jx = int(np.argmin(np.abs(wave - bbp_wave)))

    eps = 1e-12
    for ax, wv in zip(axes, plot_waves):
        j = int(np.argmin(np.abs(wave - wv)))
        x = bbp[sel, jx]
        denom = np.maximum(np.abs(Rrs_truth[sel, j]), eps)
        y_std    = 100.0 * (Rrs_truth[sel, j] - Rrs_std[sel, j])    / denom
        y_noG0   = 100.0 * (Rrs_truth[sel, j] - Rrs_var_noG0[sel, j])   / denom
        y_withG0 = 100.0 * (Rrs_truth[sel, j] - Rrs_var_withG0[sel, j]) / denom
        y_withGb = 100.0 * (Rrs_truth[sel, j] - Rrs_var_withGb[sel, j]) / denom
        y_full   = 100.0 * (Rrs_truth[sel, j] - Rrs_var_full[sel, j])   / denom

        ax.scatter(x, y_std,    s=6, alpha=0.30, color='C0', label='standard')
        ax.scatter(x, y_noG0,   s=6, alpha=0.40, color='C3', label='variable, no G0')
        ax.scatter(x, y_withG0, s=6, alpha=0.50, color='C2', label='variable, with G0')
        ax.scatter(x, y_withGb, s=6, alpha=0.55, color='C4', label='variable, with Gb')
        ax.scatter(x, y_full,   s=7, alpha=0.75, color='C5', label='variable, G0+Gb (4-p)')
        ax.axhline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_xlabel(rf'$b_{{bp}}({bbp_wave:.0f}\,\mathrm{{nm}})$  [m$^{{-1}}$]')
        ax.set_ylabel(r'$(R_{rs}^{HL} - R_{rs}^{G})/R_{rs}^{HL}$  [%]')
        ax.set_title(f'{wv:.0f} nm')
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=7, loc='best')
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig


# =============================================================================
# rrs-vs-u overlays
# =============================================================================

def plot_rrs_vs_u(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    result: Dict,
    plot_waves: Sequence[float] = (370., 440., 550., 670.),
    outfile: Optional[str] = None,
    result_with_G0: Optional[Dict] = None,
    result_with_Gb: Optional[Dict] = None,
    bbp: Optional[np.ndarray] = None,
):
    """
    Plot rrs vs u at select wavelengths with Gordon fits overlaid.

    Parameters
    ----------
    result : dict
        2-parameter fit result with 'wavelength', 'G1', 'G2'.
    result_with_G0 : dict, optional
        3-parameter (G0,G1,G2) fit result. Overlays the variable Gordon with G0 case.
    result_with_Gb : dict, optional
        3-parameter (G1,G2,Gb) fit result. Overlays the variable Gordon with Gb
        case, drawn at the median bbp at the panel's wavelength (representative
        line through the scatter).
    bbp : np.ndarray, optional
        Particulate backscatter (= bbnw), shape (n_samples, n_wave). Required
        when `result_with_Gb` is given.
    """
    n = len(plot_waves)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, wv in zip(axes, plot_waves):
        j = int(np.argmin(np.abs(wave - wv)))
        u = calc_u(a[:, j], bb[:, j])
        rrs = Rrs_to_rrs(Rrs[:, j])
        ax.scatter(u, rrs, s=6, alpha=0.3, color='gray', label='Loisel23')

        jf = int(np.argmin(np.abs(result['wavelength'] - wv)))
        G1f, G2f = result['G1'][jf], result['G2'][jf]
        u_grid = np.linspace(u.min(), u.max(), 200)

        ax.plot(u_grid, rrs_model(u_grid, G1_STANDARD, G2_STANDARD), 'C0--', lw=1.6,
                label=f'standard  G1={G1_STANDARD}, G2={G2_STANDARD}')
        ax.plot(u_grid, rrs_model(u_grid, G1f, G2f), 'C3-', lw=2,
                label=f'variable, no G0  G1={G1f:.3f}, G2={G2f:+.3f}')

        if result_with_G0 is not None:
            jf_c = int(np.argmin(np.abs(result_with_G0['wavelength'] - wv)))
            G0c = result_with_G0['G0'][jf_c]
            G1c = result_with_G0['G1'][jf_c]
            G2c = result_with_G0['G2'][jf_c]
            ax.plot(u_grid, rrs_model_const(u_grid, G0c, G1c, G2c), 'C2-', lw=2,
                    label=f'variable, with G0  G0={G0c:+.1e}, G1={G1c:.3f}, G2={G2c:+.3f}')

        if result_with_Gb is not None:
            if bbp is None:
                raise ValueError("`bbp` must be passed when result_with_Gb is given")
            jf_b = int(np.argmin(np.abs(result_with_Gb['wavelength'] - wv)))
            G1b = result_with_Gb['G1'][jf_b]
            G2b = result_with_Gb['G2'][jf_b]
            Gbb = result_with_Gb['Gb'][jf_b]
            bbp_med = float(np.median(bbp[:, j]))
            ax.plot(u_grid, rrs_model_bbp(u_grid, bbp_med, G1b, G2b, Gbb),
                    'C4-', lw=2,
                    label=(f'variable, with Gb  '
                           f'G1={G1b:.3f}, G2={G2b:+.3f}, Gb={Gbb:+.2e}  '
                           f'(@med bbp={bbp_med:.2e})'))

        ax.set_xlabel('u')
        ax.set_ylabel(r'$r_{rs}$')
        ax.set_title(f'{wv:.0f} nm')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150)
        print(f"Saved: {outfile}")
    return fig



# =============================================================================
# Notebook-friendly single-panel helpers
# (used by dev/Gordon/chk_gordon_tinker.ipynb to drive ad-hoc per-wavelength
#  investigations without going through the multi-panel batch helpers above)
# =============================================================================

def plot_rrs_vs_u_single(
    ax,
    u: np.ndarray,
    rrs: np.ndarray,
    fits: Optional[Sequence[Dict]] = None,
    log: bool = True,
    scatter_label: str = 'Loisel23',
):
    """
    Render a single rrs-vs-u panel on the given axis. Designed for notebook
    composition: the caller passes any number of fit overlays via ``fits``,
    each a dict with keys ``'u_grid'`` and ``'rrs_pred'``; optional keys
    ``'label'``, ``'color'``, ``'ls'``, ``'lw'``.
    """
    ax.scatter(u, rrs, s=6, alpha=0.3, color='gray', label=scatter_label)
    if fits:
        for f in fits:
            ax.plot(
                f['u_grid'], f['rrs_pred'],
                ls=f.get('ls', '-'),
                lw=f.get('lw', 2),
                color=f.get('color', 'C3'),
                label=f.get('label', ''),
            )
    ax.set_xlabel('u')
    ax.set_ylabel(r'$r_{rs}$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    return ax


def plot_residual_vs_bbp_single(
    ax,
    bbp_proxy: np.ndarray,
    residual_pct: np.ndarray,
    label: str = '',
    color: str = 'C3',
    bbp_wave: float = 700.,
    log_x: bool = True,
):
    """
    Render a single residual-vs-bbp panel on the given axis.
    ``residual_pct`` is the per-scene relative residual (Hydrolight - Gordon)/HL
    in percent. ``bbp_proxy`` is whatever bbp value the user wants on the
    x-axis (typically bbnw at 700 nm).
    """
    ax.scatter(bbp_proxy, residual_pct, s=10, alpha=0.5, color=color, label=label)
    ax.axhline(0, color='k', lw=0.8, alpha=0.6)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(rf'$b_{{bp}}({bbp_wave:.0f}\,\mathrm{{nm}})$  [m$^{{-1}}$]')
    ax.set_ylabel(r'$(R_{rs}^{HL} - R_{rs}^{G})/R_{rs}^{HL}$  [%]')
    ax.grid(alpha=0.3)
    if label:
        ax.legend(fontsize=8, loc='best')
    return ax
