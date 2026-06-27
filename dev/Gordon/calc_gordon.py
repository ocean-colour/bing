"""
Calculate wavelength-dependent Gordon coefficients from IOPs.

This module provides methods to derive G1 and G2 coefficients for the
Gordon semi-analytical bio-optical model:

    rrs(λ) = G1(λ) * u(λ) + G2(λ) * u(λ)²

where u = bb / (a + bb) is the backscattering ratio.

The standard Gordon (1988) values are G1=0.0949, G2=0.0794, but these
can vary with wavelength due to factors like scattering phase function
changes, viewing geometry, and sea surface roughness.

This module fits wavelength-dependent coefficients using Hydrolight
radiative transfer simulations from the Loisel et al. (2023) dataset.

References
----------
- Gordon, H.R. et al. (1988). "A semianalytic radiance model of ocean color,"
  J. Geophys. Res. 93, 10909-10924.
- Loisel, H. et al. (2023). Hydrolight synthetic ocean color dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Dict, Union, Sequence


# Conversion factors from rrs to Rrs (Lee et al. 2002).
# NOTE: must match bing.rt.rrs (A=0.52, B=1.7). The original fig_u() in
# papers/phytoplankton used B=1.17, which is a typo of 1.7; fitting in that
# convention and applying via bing.rt.rrs biases every reconstructed Rrs.
A_RRS = 0.52
B_RRS = 1.7

# Standard Gordon coefficients for reference
G1_STANDARD = 0.0949
G2_STANDARD = 0.0794


def rrs_model(u: np.ndarray, G1: float, G2: float) -> np.ndarray:
    """
    Gordon model for subsurface remote sensing reflectance.

    Parameters
    ----------
    u : np.ndarray
        Backscattering ratio: u = bb / (a + bb)
    G1 : float
        First-order Gordon coefficient.
    G2 : float
        Second-order Gordon coefficient.

    Returns
    -------
    np.ndarray
        Subsurface remote sensing reflectance rrs.
    """
    return G1 * u + G2 * u**2


def rrs_model_const(u: np.ndarray, G0: float, G1: float, G2: float) -> np.ndarray:
    """
    Gordon model with a constant offset: rrs = G0 + G1·u + G2·u².

    G0 captures structure the pure G1·u + G2·u² form cannot reach, e.g. the
    weak dependence of Hydrolight rrs on (a, bb) separately rather than only
    on u. Empirically, allowing G0 of order 10⁻⁴ at red wavelengths cuts the
    700 nm rRMS by an order of magnitude and removes the residual-vs-bbp tilt.
    """
    return G0 + G1 * u + G2 * u ** 2


def rrs_model_bbp(u: np.ndarray, bbp: np.ndarray,
                  G1: float, G2: float, Gb: float) -> np.ndarray:
    """
    Gordon model with an explicit bbp dependence:
        rrs = G1·u + G2·u² + Gb·bbp

    The third coefficient Gb is a linear slope in particulate backscatter (bbp,
    i.e. bbnw). Empirically this captures the residual bbp dependence that a
    pure function of u cannot, and is most useful around 500 nm where the
    constant-offset (G0) form fails (the trophic-state-driven residual there
    is roughly linear in bbp).
    """
    return G1 * u + G2 * u ** 2 + Gb * bbp


def rrs_model_full(u: np.ndarray, bbp: np.ndarray,
                   G0: float, G1: float, G2: float, Gb: float) -> np.ndarray:
    """
    Four-parameter Gordon model: rrs = G0 + G1·u + G2·u² + Gb·bbp.

    Combines the constant offset (G0) and the bbp slope (Gb) of the two
    competing 3-parameter recipes. Empirically the four-parameter form
    matches the G0-only fit at red wavelengths (550–700 nm) and improves on
    the Gb-only fit in the blue (400–500 nm).
    """
    return G0 + G1 * u + G2 * u ** 2 + Gb * bbp


def Rrs_to_rrs(Rrs: np.ndarray, A: float = A_RRS, B: float = B_RRS) -> np.ndarray:
    """
    Convert above-surface Rrs to subsurface rrs.

    Uses the relation: rrs = Rrs / (A + B * Rrs)

    Parameters
    ----------
    Rrs : np.ndarray
        Remote sensing reflectance (above surface).
    A : float
        Conversion coefficient (default 0.52).
    B : float
        Conversion coefficient (default 1.17).

    Returns
    -------
    np.ndarray
        Subsurface remote sensing reflectance.
    """
    return Rrs / (A + B * Rrs)


def rrs_to_Rrs(rrs: np.ndarray, A: float = A_RRS, B: float = B_RRS) -> np.ndarray:
    """
    Convert subsurface rrs to above-surface Rrs.

    Uses the relation: Rrs = A * rrs / (1 - B * rrs)

    Parameters
    ----------
    rrs : np.ndarray
        Subsurface remote sensing reflectance.
    A : float
        Conversion coefficient (default 0.52).
    B : float
        Conversion coefficient (default 1.17).

    Returns
    -------
    np.ndarray
        Remote sensing reflectance (above surface).
    """
    return A * rrs / (1 - B * rrs)


def calc_u(a: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """
    Calculate the backscattering ratio u.

    Parameters
    ----------
    a : np.ndarray
        Total absorption coefficient [m^-1].
    bb : np.ndarray
        Total backscattering coefficient [m^-1].

    Returns
    -------
    np.ndarray
        Backscattering ratio u = bb / (a + bb).
    """
    return bb / (a + bb)


def fit_gordon_at_wavelength(
    u: np.ndarray,
    rrs: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    p0: Tuple[float, float] = (0.1, 0.1),
    weight_mode: str = 'relative',
    rel_floor: float = 1e-5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Gordon coefficients G1 and G2 at a single wavelength.

    Uses scipy.optimize.curve_fit to fit rrs = G1*u + G2*u^2.

    Parameters
    ----------
    u : np.ndarray
        Backscattering ratio values (1D array, n_samples).
    rrs : np.ndarray
        Subsurface remote sensing reflectance (1D array, n_samples).
    sigma : np.ndarray, optional
        Per-sample uncertainties in rrs. If provided, overrides `weight_mode`.
    p0 : tuple
        Initial guess for (G1, G2).
    weight_mode : str
        One of:
        - 'absolute' : constant sigma = 3e-4 (legacy behavior). Minimizes
          absolute residuals in rrs. Strongly biased toward high-u (clear
          / blue) points; leaves G2 essentially unconstrained at red
          wavelengths where u is narrow and small.
        - 'relative' : sigma_i = max(|rrs_i|, rel_floor). Minimizes
          relative residuals; balances clear and turbid contributions at
          every wavelength.
    rel_floor : float
        Lower bound on rrs when computing relative sigma (avoids divide
        blow-ups for samples with rrs near zero or negative).
    bounds : (lower, upper), optional
        Box bounds on (G1, G2) passed to curve_fit. Use to keep coefficients
        physical when the quadratic term is poorly constrained.

    Returns
    -------
    params : np.ndarray
        Fitted parameters [G1, G2].
    cov : np.ndarray
        Covariance matrix from the fit.
    """
    if sigma is None:
        if weight_mode == 'absolute':
            sigma = np.full_like(u, 3e-4, dtype=float)
        elif weight_mode == 'relative':
            sigma = np.maximum(np.abs(rrs), rel_floor)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    kwargs = dict(p0=p0, sigma=sigma, absolute_sigma=False)
    if bounds is not None:
        kwargs['bounds'] = bounds

    params, cov = curve_fit(rrs_model, u, rrs, **kwargs)
    return params, cov


def fit_gordon_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    sigma_rrs: Optional[np.ndarray] = None,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Fit wavelength-dependent Gordon coefficients from IOPs and Rrs.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths [nm], shape (n_wave,).
    Rrs : np.ndarray
        Remote sensing reflectance, shape (n_samples, n_wave).
    a : np.ndarray
        Total absorption coefficients, shape (n_samples, n_wave).
    bb : np.ndarray
        Total backscattering coefficients, shape (n_samples, n_wave).
    wave_select : np.ndarray, optional
        Specific wavelengths to fit. If None, fits all wavelengths.
    sigma_rrs : np.ndarray, optional
        Uncertainties in rrs, shape (n_samples, n_wave). If provided, overrides
        `weight_mode`.
    return_stats : bool
        If True, return fitting statistics (RMS errors).
    weight_mode : str
        Per-wavelength weighting strategy when `sigma_rrs` is None. See
        `fit_gordon_at_wavelength`. Default 'relative'.
    bounds : (lower, upper), optional
        Box bounds on (G1, G2) for every wavelength, passed to curve_fit.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'wavelength': fitted wavelengths
        - 'G1': fitted G1 values at each wavelength
        - 'G2': fitted G2 values at each wavelength
        - 'G1_err': standard error in G1
        - 'G2_err': standard error in G2
    stats : dict (if return_stats=True)
        Dictionary with fitting statistics:
        - 'rRMS': relative RMS error at each wavelength
        - 'RMS': absolute RMS error at each wavelength
    """
    # Calculate u and rrs
    u = calc_u(a, bb)
    rrs = Rrs_to_rrs(Rrs)

    # Determine wavelengths to fit
    if wave_select is None:
        wave_fit = wave
        idx_fit = np.arange(len(wave))
    else:
        idx_fit = [np.argmin(np.abs(wave - w)) for w in wave_select]
        wave_fit = wave[idx_fit]

    # Arrays to store results
    n_wave = len(idx_fit)
    G1_arr = np.zeros(n_wave)
    G2_arr = np.zeros(n_wave)
    G1_err = np.zeros(n_wave)
    G2_err = np.zeros(n_wave)
    rRMS = np.zeros(n_wave)
    RMS = np.zeros(n_wave)

    # Fit at each wavelength
    for ii, idx in enumerate(idx_fit):
        u_wv = u[:, idx]
        rrs_wv = rrs[:, idx]

        if sigma_rrs is not None:
            sig_wv = sigma_rrs[:, idx]
        else:
            sig_wv = None

        # Fit
        params, cov = fit_gordon_at_wavelength(
            u_wv, rrs_wv, sigma=sig_wv,
            weight_mode=weight_mode, bounds=bounds,
        )
        G1_arr[ii] = params[0]
        G2_arr[ii] = params[1]
        G1_err[ii] = np.sqrt(cov[0, 0])
        G2_err[ii] = np.sqrt(cov[1, 1])

        # Calculate RMS
        rrs_fit = rrs_model(u_wv, params[0], params[1])
        RMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_fit)**2))
        rRMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_fit)**2 / rrs_fit**2))

    result = {
        'wavelength': wave_fit,
        'G1': G1_arr,
        'G2': G2_arr,
        'G1_err': G1_err,
        'G2_err': G2_err
    }

    if return_stats:
        stats = {
            'rRMS': rRMS,
            'RMS': RMS
        }
        return result, stats

    return result


def fit_gordon_from_loisel23(
    wave_select: Optional[np.ndarray] = None,
    wv_min: float = 350.,
    wv_max: float = 700.,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Fit wavelength-dependent Gordon coefficients using Loisel23 Hydrolight data.

    This function loads the Loisel et al. (2023) synthetic dataset containing
    Hydrolight radiative transfer simulations and derives G1/G2 coefficients
    at each wavelength.

    Parameters
    ----------
    wave_select : np.ndarray, optional
        Specific wavelengths [nm] to fit. If None, fits all wavelengths
        in the dataset within the wv_min/wv_max range.
    wv_min : float
        Minimum wavelength to include.
    wv_max : float
        Maximum wavelength to include.
    return_stats : bool
        If True, return fitting statistics.

    Returns
    -------
    result : dict
        Dictionary with fitted coefficients (see fit_gordon_coefficients).
    stats : dict (if return_stats=True)
        Dictionary with fitting statistics.

    Examples
    --------
    >>> # Fit at standard wavelengths
    >>> result = fit_gordon_from_loisel23(
    ...     wave_select=np.array([370., 440., 500., 600.]))
    >>> print(f"G1 at 440nm: {result['G1'][1]:.4f}")
    >>> print(f"G2 at 440nm: {result['G2'][1]:.4f}")

    >>> # Fit all wavelengths
    >>> result, stats = fit_gordon_from_loisel23(return_stats=True)
    >>> print(f"rRMS at 440nm: {stats['rRMS'][np.argmin(np.abs(result['wavelength']-440))]:.4f}")
    """
    try:
        from ocpy.hydrolight import loisel23
    except ImportError:
        raise ImportError(
            "ocpy package with loisel23 module required. "
            "Install with: pip install ocpy"
        )

    # Load Loisel23 dataset -- Elastic
    try: 
        ds = loisel23.load_ds(1, 0)
    except:
        raise IOError("Loisel23 files not available")

    # Extract data
    wave = ds.Lambda.data
    Rrs = ds.Rrs.data
    a = ds.a.data
    bb = ds.bb.data

    # Apply wavelength limits
    gd_wave = (wave >= wv_min) & (wave <= wv_max)
    wave = wave[gd_wave]
    Rrs = Rrs[:, gd_wave]
    a = a[:, gd_wave]
    bb = bb[:, gd_wave]

    return fit_gordon_coefficients(
        wave, Rrs, a, bb,
        wave_select=wave_select,
        return_stats=return_stats,
        weight_mode=weight_mode,
        bounds=bounds,
    )


def interpolate_gordon_coefficients(
    result: Dict,
    wave_target: np.ndarray,
    method: str = 'linear'
) -> Dict:
    """
    Interpolate fitted Gordon coefficients to target wavelengths.

    Parameters
    ----------
    result : dict
        Output from fit_gordon_coefficients or fit_gordon_from_loisel23.
    wave_target : np.ndarray
        Target wavelengths for interpolation.
    method : str
        Interpolation method ('linear', 'quadratic', 'cubic').

    Returns
    -------
    dict
        Dictionary with interpolated G1 and G2 at target wavelengths.
    """
    from scipy.interpolate import interp1d

    kind = method
    if method == 'quadratic':
        kind = 2
    elif method == 'cubic':
        kind = 3

    f_G1 = interp1d(result['wavelength'], result['G1'], kind=kind,
                    bounds_error=False, fill_value='extrapolate')
    f_G2 = interp1d(result['wavelength'], result['G2'], kind=kind,
                    bounds_error=False, fill_value='extrapolate')

    return {
        'wavelength': wave_target,
        'G1': f_G1(wave_target),
        'G2': f_G2(wave_target)
    }


def calc_Rrs_with_variable_gordon(
    a: np.ndarray,
    bb: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    G0: Optional[np.ndarray] = None,
    Gb: Optional[np.ndarray] = None,
    bbp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate Rrs using wavelength-dependent Gordon coefficients.

    Parameters
    ----------
    a, bb : np.ndarray
        IOPs, shape (..., n_wave).
    G1, G2 : np.ndarray
        Wavelength-dependent Gordon coefficients, shape (n_wave,).
    G0 : np.ndarray, optional
        Wavelength-dependent constant offset, shape (n_wave,). If provided,
        evaluates rrs = G0 + G1·u + G2·u² (instead of G1·u + G2·u²).
    Gb : np.ndarray, optional
        Wavelength-dependent slope on particulate backscatter. If provided
        with `bbp`, evaluates rrs = G1·u + G2·u² + Gb·bbp.
    bbp : np.ndarray, optional
        Particulate backscatter (= bbnw), shape matching `bb`. Required when
        `Gb` is provided.

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs.
    """
    u = calc_u(a, bb)
    rrs = G1 * u + G2 * u**2
    if G0 is not None:
        rrs = rrs + G0
    if Gb is not None:
        if bbp is None:
            raise ValueError("`bbp` must be supplied when `Gb` is given")
        rrs = rrs + Gb * bbp
    Rrs = rrs_to_Rrs(rrs)
    return Rrs


# ---------------------------------------------------------------------------
# 3-parameter (G0 + G1·u + G2·u²) per-wavelength fit
# ---------------------------------------------------------------------------

def fit_gordon_const_at_wavelength(
    u: np.ndarray,
    rrs: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    p0: Tuple[float, float, float] = (0.0, 0.1, 0.0),
    weight_mode: str = 'relative',
    rel_floor: float = 1e-5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-1e-3, 0.05, -5.0), (1e-3, 0.15, 0.5)
    ),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit (G0, G1, G2) at a single wavelength to rrs = G0 + G1·u + G2·u².

    Parameters match fit_gordon_at_wavelength; the only new argument is the
    expanded p0/bounds (3 entries). The default G0 bound of ±1e-3 is well
    outside the empirical magnitude (~10⁻⁴) but still anchors the optimizer.
    """
    if sigma is None:
        if weight_mode == 'absolute':
            sigma = np.full_like(u, 3e-4, dtype=float)
        elif weight_mode == 'relative':
            sigma = np.maximum(np.abs(rrs), rel_floor)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    kwargs = dict(p0=p0, sigma=sigma, absolute_sigma=False)
    if bounds is not None:
        kwargs['bounds'] = bounds

    params, cov = curve_fit(rrs_model_const, u, rrs, **kwargs)
    return params, cov


def fit_gordon_const_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-1e-3, 0.05, -5.0), (1e-3, 0.15, 0.5)
    ),
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Per-wavelength 3-parameter fit (G0, G1, G2) on a Hydrolight-like dataset.

    Returns
    -------
    result : dict with keys 'wavelength', 'G0', 'G1', 'G2', 'G0_err',
             'G1_err', 'G2_err'.
    stats : dict with 'rRMS', 'RMS' per wavelength (if return_stats=True).
    """
    u = calc_u(a, bb)
    rrs = Rrs_to_rrs(Rrs)

    if wave_select is None:
        wave_fit = wave
        idx_fit = np.arange(len(wave))
    else:
        idx_fit = [np.argmin(np.abs(wave - w)) for w in wave_select]
        wave_fit = wave[idx_fit]

    n = len(idx_fit)
    G0_arr = np.zeros(n); G1_arr = np.zeros(n); G2_arr = np.zeros(n)
    G0_err = np.zeros(n); G1_err = np.zeros(n); G2_err = np.zeros(n)
    rRMS = np.zeros(n);   RMS = np.zeros(n)

    for ii, idx in enumerate(idx_fit):
        u_wv = u[:, idx]; rrs_wv = rrs[:, idx]
        params, cov = fit_gordon_const_at_wavelength(
            u_wv, rrs_wv, weight_mode=weight_mode, bounds=bounds,
        )
        G0_arr[ii], G1_arr[ii], G2_arr[ii] = params
        G0_err[ii] = np.sqrt(cov[0, 0])
        G1_err[ii] = np.sqrt(cov[1, 1])
        G2_err[ii] = np.sqrt(cov[2, 2])

        rrs_pred = rrs_model_const(u_wv, *params)
        RMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_pred) ** 2))
        rRMS[ii] = np.sqrt(np.mean(
            (rrs_wv - rrs_pred) ** 2 / np.maximum(rrs_pred ** 2, 1e-20)
        ))

    result = {
        'wavelength': wave_fit,
        'G0': G0_arr, 'G1': G1_arr, 'G2': G2_arr,
        'G0_err': G0_err, 'G1_err': G1_err, 'G2_err': G2_err,
    }
    if return_stats:
        return result, {'rRMS': rRMS, 'RMS': RMS}
    return result


def save_gordon_const_to_csv(
    result: Dict,
    stats: Dict,
    filename: str,
    source: str = "Loisel23 elastic; 3-parameter fit (G0 + G1·u + G2·u²)",
    weight_mode: Optional[str] = None,
    A: float = A_RRS,
    B: float = B_RRS,
) -> None:
    """
    Save the 3-parameter (G0, G1, G2) fit to CSV. The file format mirrors
    save_gordon_to_csv but adds G0/G0_err columns. This CSV is NOT directly
    consumable by bing.rt.rrs.wave_dependent_gordon yet (it returns G1, G2
    only) -- it is an experimental output for evaluation.
    """
    import pandas as pd
    df = pd.DataFrame({
        'wavelength': result['wavelength'],
        'G0': result['G0'], 'G1': result['G1'], 'G2': result['G2'],
        'G0_err': result['G0_err'], 'G1_err': result['G1_err'], 'G2_err': result['G2_err'],
        'rRMS': stats['rRMS'], 'RMS': stats['RMS'],
    })
    with open(filename, 'w') as f:
        f.write("# Gordon coefficients (with constant) fitted from Loisel23\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Standard G1: {G1_STANDARD}\n")
        f.write(f"# Standard G2: {G2_STANDARD}\n")
        f.write(f"# Rrs<->rrs convention: A={A}, B={B}\n")
        if weight_mode is not None:
            f.write(f"# weight_mode: {weight_mode}\n")
        f.write("#\n")
        df.to_csv(f, index=False)
    print(f"Saved 3-parameter Gordon coefficients to: {filename}")


# ---------------------------------------------------------------------------
# 3-parameter (G1·u + G2·u² + Gb·bbp) per-wavelength fit
# ---------------------------------------------------------------------------

def fit_gordon_bbp_at_wavelength(
    u: np.ndarray,
    bbp: np.ndarray,
    rrs: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    p0: Tuple[float, float, float] = (0.1, 0.0, 0.0),
    weight_mode: str = 'relative',
    rel_floor: float = 1e-5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (0.05, -2.0, -1.0), (0.15, 0.5, 1.0)
    ),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit (G1, G2, Gb) at a single wavelength to rrs = G1·u + G2·u² + Gb·bbp.

    Note that the model takes two independent variables (u, bbp). They are
    bundled into a 2-row array and passed via curve_fit's multi-variable form.
    """
    if sigma is None:
        if weight_mode == 'absolute':
            sigma = np.full_like(u, 3e-4, dtype=float)
        elif weight_mode == 'relative':
            sigma = np.maximum(np.abs(rrs), rel_floor)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    def _f(X, G1, G2, Gb):
        u_, bbp_ = X
        return rrs_model_bbp(u_, bbp_, G1, G2, Gb)

    kwargs = dict(p0=p0, sigma=sigma, absolute_sigma=False)
    if bounds is not None:
        kwargs['bounds'] = bounds

    params, cov = curve_fit(_f, (u, bbp), rrs, **kwargs)
    return params, cov


def fit_gordon_bbp_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    bbp: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (0.05, -2.0, -1.0), (0.15, 0.5, 1.0)
    ),
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Per-wavelength 3-parameter fit (G1, G2, Gb) on a Hydrolight-like dataset.

    Parameters
    ----------
    wave, Rrs, a, bb : as in fit_gordon_coefficients.
    bbp : np.ndarray
        Particulate backscatter (= bbnw), shape (n_samples, n_wave).

    Returns
    -------
    result : dict with keys 'wavelength', 'G1', 'G2', 'Gb', 'G1_err', 'G2_err', 'Gb_err'.
    stats : dict with 'rRMS', 'RMS' per wavelength (if return_stats=True).
    """
    u = calc_u(a, bb)
    rrs = Rrs_to_rrs(Rrs)

    if wave_select is None:
        wave_fit = wave
        idx_fit = np.arange(len(wave))
    else:
        idx_fit = [np.argmin(np.abs(wave - w)) for w in wave_select]
        wave_fit = wave[idx_fit]

    n = len(idx_fit)
    G1_arr = np.zeros(n); G2_arr = np.zeros(n); Gb_arr = np.zeros(n)
    G1_err = np.zeros(n); G2_err = np.zeros(n); Gb_err = np.zeros(n)
    rRMS = np.zeros(n);   RMS = np.zeros(n)

    for ii, idx in enumerate(idx_fit):
        u_wv = u[:, idx]; rrs_wv = rrs[:, idx]; bbp_wv = bbp[:, idx]
        params, cov = fit_gordon_bbp_at_wavelength(
            u_wv, bbp_wv, rrs_wv,
            weight_mode=weight_mode, bounds=bounds,
        )
        G1_arr[ii], G2_arr[ii], Gb_arr[ii] = params
        G1_err[ii] = np.sqrt(cov[0, 0])
        G2_err[ii] = np.sqrt(cov[1, 1])
        Gb_err[ii] = np.sqrt(cov[2, 2])

        rrs_pred = rrs_model_bbp(u_wv, bbp_wv, *params)
        RMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_pred) ** 2))
        rRMS[ii] = np.sqrt(np.mean(
            (rrs_wv - rrs_pred) ** 2 / np.maximum(rrs_pred ** 2, 1e-20)
        ))

    result = {
        'wavelength': wave_fit,
        'G1': G1_arr, 'G2': G2_arr, 'Gb': Gb_arr,
        'G1_err': G1_err, 'G2_err': G2_err, 'Gb_err': Gb_err,
    }
    if return_stats:
        return result, {'rRMS': rRMS, 'RMS': RMS}
    return result


def save_gordon_bbp_to_csv(
    result: Dict,
    stats: Dict,
    filename: str,
    source: str = "Loisel23 elastic; 3-parameter fit (G1·u + G2·u² + Gb·bbp)",
    weight_mode: Optional[str] = None,
    A: float = A_RRS,
    B: float = B_RRS,
) -> None:
    """
    Save the (G1, G2, Gb) fit to CSV. Mirrors save_gordon_const_to_csv but with
    a Gb column instead of G0.
    """
    import pandas as pd
    df = pd.DataFrame({
        'wavelength': result['wavelength'],
        'G1': result['G1'], 'G2': result['G2'], 'Gb': result['Gb'],
        'G1_err': result['G1_err'], 'G2_err': result['G2_err'], 'Gb_err': result['Gb_err'],
        'rRMS': stats['rRMS'], 'RMS': stats['RMS'],
    })
    with open(filename, 'w') as f:
        f.write("# Gordon coefficients (with bbp slope) fitted from Loisel23\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Standard G1: {G1_STANDARD}\n")
        f.write(f"# Standard G2: {G2_STANDARD}\n")
        f.write(f"# Rrs<->rrs convention: A={A}, B={B}\n")
        if weight_mode is not None:
            f.write(f"# weight_mode: {weight_mode}\n")
        f.write("#\n")
        df.to_csv(f, index=False)
    print(f"Saved 3-parameter (Gb) Gordon coefficients to: {filename}")


# ---------------------------------------------------------------------------
# 4-parameter (G0 + G1·u + G2·u² + Gb·bbp) per-wavelength fit
# ---------------------------------------------------------------------------

def fit_gordon_full_at_wavelength(
    u: np.ndarray,
    bbp: np.ndarray,
    rrs: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    p0: Tuple[float, float, float, float] = (0.0, 0.1, 0.0, 0.0),
    weight_mode: str = 'relative',
    rel_floor: float = 1e-5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-1e-3, 0.05, -2.0, -1.0), (1e-3, 0.15, 0.5, 1.0)
    ),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit (G0, G1, G2, Gb) at a single wavelength to
        rrs = G0 + G1·u + G2·u² + Gb·bbp.
    """
    if sigma is None:
        if weight_mode == 'absolute':
            sigma = np.full_like(u, 3e-4, dtype=float)
        elif weight_mode == 'relative':
            sigma = np.maximum(np.abs(rrs), rel_floor)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    def _f(X, G0, G1, G2, Gb):
        u_, bbp_ = X
        return rrs_model_full(u_, bbp_, G0, G1, G2, Gb)

    kwargs = dict(p0=p0, sigma=sigma, absolute_sigma=False)
    if bounds is not None:
        kwargs['bounds'] = bounds

    params, cov = curve_fit(_f, (u, bbp), rrs, **kwargs)
    return params, cov


def fit_gordon_full_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    bbp: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-1e-3, 0.05, -2.0, -1.0), (1e-3, 0.15, 0.5, 1.0)
    ),
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Per-wavelength 4-parameter (G0, G1, G2, Gb) fit.

    Returns dict with 'wavelength', 'G0', 'G1', 'G2', 'Gb' and per-parameter
    standard errors; optionally per-wavelength stats {'rRMS', 'RMS'}.
    """
    u = calc_u(a, bb)
    rrs = Rrs_to_rrs(Rrs)

    if wave_select is None:
        wave_fit = wave
        idx_fit = np.arange(len(wave))
    else:
        idx_fit = [np.argmin(np.abs(wave - w)) for w in wave_select]
        wave_fit = wave[idx_fit]

    n = len(idx_fit)
    G0_arr = np.zeros(n); G1_arr = np.zeros(n)
    G2_arr = np.zeros(n); Gb_arr = np.zeros(n)
    G0_err = np.zeros(n); G1_err = np.zeros(n)
    G2_err = np.zeros(n); Gb_err = np.zeros(n)
    rRMS = np.zeros(n);   RMS = np.zeros(n)

    for ii, idx in enumerate(idx_fit):
        u_wv = u[:, idx]; rrs_wv = rrs[:, idx]; bbp_wv = bbp[:, idx]
        params, cov = fit_gordon_full_at_wavelength(
            u_wv, bbp_wv, rrs_wv,
            weight_mode=weight_mode, bounds=bounds,
        )
        G0_arr[ii], G1_arr[ii], G2_arr[ii], Gb_arr[ii] = params
        G0_err[ii] = np.sqrt(cov[0, 0])
        G1_err[ii] = np.sqrt(cov[1, 1])
        G2_err[ii] = np.sqrt(cov[2, 2])
        Gb_err[ii] = np.sqrt(cov[3, 3])

        rrs_pred = rrs_model_full(u_wv, bbp_wv, *params)
        RMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_pred) ** 2))
        rRMS[ii] = np.sqrt(np.mean(
            (rrs_wv - rrs_pred) ** 2 / np.maximum(rrs_pred ** 2, 1e-20)
        ))

    result = {
        'wavelength': wave_fit,
        'G0': G0_arr, 'G1': G1_arr, 'G2': G2_arr, 'Gb': Gb_arr,
        'G0_err': G0_err, 'G1_err': G1_err, 'G2_err': G2_err, 'Gb_err': Gb_err,
    }
    if return_stats:
        return result, {'rRMS': rRMS, 'RMS': RMS}
    return result


def save_gordon_full_to_csv(
    result: Dict,
    stats: Dict,
    filename: str,
    source: str = "Loisel23 elastic; 4-parameter fit (G0 + G1·u + G2·u² + Gb·bbp)",
    weight_mode: Optional[str] = None,
    A: float = A_RRS,
    B: float = B_RRS,
) -> None:
    """
    Save the (G0, G1, G2, Gb) fit to CSV.
    """
    import pandas as pd
    df = pd.DataFrame({
        'wavelength': result['wavelength'],
        'G0': result['G0'], 'G1': result['G1'],
        'G2': result['G2'], 'Gb': result['Gb'],
        'G0_err': result['G0_err'], 'G1_err': result['G1_err'],
        'G2_err': result['G2_err'], 'Gb_err': result['Gb_err'],
        'rRMS': stats['rRMS'], 'RMS': stats['RMS'],
    })
    with open(filename, 'w') as f:
        f.write("# Gordon coefficients (with G0 + Gb) fitted from Loisel23\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Standard G1: {G1_STANDARD}\n")
        f.write(f"# Standard G2: {G2_STANDARD}\n")
        f.write(f"# Rrs<->rrs convention: A={A}, B={B}\n")
        if weight_mode is not None:
            f.write(f"# weight_mode: {weight_mode}\n")
        f.write("#\n")
        df.to_csv(f, index=False)
    print(f"Saved 4-parameter (G0,Gb) Gordon coefficients to: {filename}")


def fit_gordon_2stage_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    bbp_proxy: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    return_stats: bool = False,
    weight_mode: str = 'relative',
    bounds_u: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (0.05, -2.0), (0.15, 0.5)
    ),
    bounds_corr: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-1e-3, -1.0), (1e-3, 1.0)
    ),
) -> Union[Dict, Tuple[Dict, Dict]]:
    """
    Two-stage 4-parameter fit:

        Stage 1: fit (G1, G2) to rrs ≈ G1·u + G2·u² with the usual recipe.
        Stage 2: fit (G0, Gb) to the Stage-1 residuals as a function of
                 ``bbp_proxy`` (a single scalar per sample, e.g. bbp at 700
                 nm, so it acts as a trophic-state proxy independent of λ).

    The two-stage approach decouples the u-dependent part of the fit (which
    is what the u-only Gordon family was designed to capture) from the
    (a, bb) decoupling that the constant offset and bbp slope correct.
    Motivation: a joint 4-param fit lets G1/G2 absorb part of the bbp-driven
    residual that should be attributed to G0/Gb -- leaving residual bbp
    dependence after a joint fit. Holding G1/G2 fixed after Stage 1 prevents
    that leakage.

    Parameters
    ----------
    wave, Rrs, a, bb : as in ``fit_gordon_full_coefficients``.
    bbp_proxy : np.ndarray
        Per-sample bbp value used at every wavelength, shape (n_samples,).
        Typically the L23 ``bbnw`` evaluated at 700 nm.

    Returns
    -------
    result : dict with 'wavelength', 'G0', 'G1', 'G2', 'Gb' (+ _err columns).
    stats : dict with 'rRMS', 'RMS' per wavelength (if return_stats).
    """
    u = calc_u(a, bb)
    rrs = Rrs_to_rrs(Rrs)

    if wave_select is None:
        wave_fit = wave
        idx_fit = np.arange(len(wave))
    else:
        idx_fit = [np.argmin(np.abs(wave - w)) for w in wave_select]
        wave_fit = wave[idx_fit]

    n = len(idx_fit)
    G0_arr = np.zeros(n); G1_arr = np.zeros(n)
    G2_arr = np.zeros(n); Gb_arr = np.zeros(n)
    G0_err = np.zeros(n); G1_err = np.zeros(n)
    G2_err = np.zeros(n); Gb_err = np.zeros(n)
    rRMS = np.zeros(n);   RMS = np.zeros(n)

    def _quad(u_, G1, G2):
        return G1 * u_ + G2 * u_ ** 2

    def _corr(bbp_, G0, Gb):
        return G0 + Gb * bbp_

    for ii, idx in enumerate(idx_fit):
        u_wv = u[:, idx]
        rrs_wv = rrs[:, idx]

        if weight_mode == 'absolute':
            sigma = np.full_like(u_wv, 3e-4, dtype=float)
        elif weight_mode == 'relative':
            sigma = np.maximum(np.abs(rrs_wv), 1e-5)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

        # Stage 1: fit G1, G2 (no offset, no bbp)
        p_u, cov_u = curve_fit(
            _quad, u_wv, rrs_wv,
            p0=(0.1, 0.0), sigma=sigma, absolute_sigma=False,
            bounds=bounds_u,
        )
        G1_arr[ii], G2_arr[ii] = p_u
        G1_err[ii] = np.sqrt(cov_u[0, 0])
        G2_err[ii] = np.sqrt(cov_u[1, 1])

        # Stage 2: fit G0, Gb to the residuals against bbp_proxy
        res = rrs_wv - _quad(u_wv, *p_u)
        p_c, cov_c = curve_fit(
            _corr, bbp_proxy, res,
            p0=(0.0, 0.0), sigma=sigma, absolute_sigma=False,
            bounds=bounds_corr,
        )
        G0_arr[ii], Gb_arr[ii] = p_c
        G0_err[ii] = np.sqrt(cov_c[0, 0])
        Gb_err[ii] = np.sqrt(cov_c[1, 1])

        # Stats on the full (4-param) reconstruction
        rrs_pred = _quad(u_wv, *p_u) + _corr(bbp_proxy, *p_c)
        RMS[ii] = np.sqrt(np.mean((rrs_wv - rrs_pred) ** 2))
        rRMS[ii] = np.sqrt(np.mean(
            (rrs_wv - rrs_pred) ** 2 / np.maximum(rrs_pred ** 2, 1e-20)
        ))

    result = {
        'wavelength': wave_fit,
        'G0': G0_arr, 'G1': G1_arr, 'G2': G2_arr, 'Gb': Gb_arr,
        'G0_err': G0_err, 'G1_err': G1_err, 'G2_err': G2_err, 'Gb_err': Gb_err,
    }
    if return_stats:
        return result, {'rRMS': rRMS, 'RMS': RMS}
    return result


def load_gordon_csv(filename: str) -> Tuple[Dict, Dict]:
    """
    Load a Gordon-coefficient CSV (any of the 2/3/4-parameter recipes) into
    the (result, stats) dict pair that the fit functions return.

    Used by ``run_full_assessment(..., clobber=False)`` to avoid refitting
    when a saved CSV is already present.

    Returns
    -------
    result : dict
        {'wavelength', 'G1', 'G2'} always; 'G0' and/or 'Gb' if present in the
        CSV; corresponding '_err' columns when present (or zeros otherwise).
    stats : dict
        {'rRMS', 'RMS'} (from the CSV; if absent, both are NaN arrays).
    """
    import pandas as pd
    df = pd.read_csv(filename, comment='#')

    wave = df['wavelength'].values
    result = {'wavelength': wave}
    for col in ('G0', 'G1', 'G2', 'Gb'):
        if col in df.columns:
            result[col] = df[col].values
        # Per-parameter standard errors
        ecol = f'{col}_err'
        if ecol in df.columns:
            result[ecol] = df[ecol].values
        elif col in result:
            result[ecol] = np.zeros_like(result[col])

    stats = {}
    stats['rRMS'] = df['rRMS'].values if 'rRMS' in df.columns else np.full_like(wave, np.nan, dtype=float)
    stats['RMS']  = df['RMS'].values  if 'RMS' in df.columns  else np.full_like(wave, np.nan, dtype=float)
    print(f"Loaded Gordon coefficients from: {filename}")
    return result, stats


# Convenience functions for common use cases

def get_standard_gordon() -> Tuple[float, float]:
    """Return standard Gordon coefficients (G1, G2)."""
    return G1_STANDARD, G2_STANDARD


def print_gordon_comparison(result: Dict, wavelengths: np.ndarray = None) -> None:
    """
    Print a comparison of fitted vs standard Gordon coefficients.

    Parameters
    ----------
    result : dict
        Output from fit_gordon_coefficients.
    wavelengths : np.ndarray, optional
        Wavelengths to display. If None, shows all.
    """
    if wavelengths is None:
        idx = np.arange(len(result['wavelength']))
    else:
        idx = [np.argmin(np.abs(result['wavelength'] - w)) for w in wavelengths]

    print("Wavelength   G1 (fit)   G2 (fit)   G1 (std)   G2 (std)")
    print("-" * 58)
    for ii in idx:
        wv = result['wavelength'][ii]
        G1 = result['G1'][ii]
        G2 = result['G2'][ii]
        print(f"  {wv:6.1f}     {G1:.4f}     {G2:.4f}     {G1_STANDARD:.4f}     {G2_STANDARD:.4f}")


def save_gordon_to_csv(
    result: Dict,
    stats: Dict,
    filename: str,
    source: str = "Loisel et al. (2023) synthetic dataset",
    weight_mode: Optional[str] = None,
    A: float = A_RRS,
    B: float = B_RRS,
) -> None:
    """
    Save fitted Gordon coefficients to CSV file.

    Parameters
    ----------
    result : dict
        Output from fit_gordon_coefficients containing G1, G2, etc.
    stats : dict
        Dictionary with fitting statistics (rRMS, RMS).
    filename : str
        Output CSV filename.
    source : str
        Data source description for metadata header.
    weight_mode : str, optional
        Recorded in the header so consumers know which fit produced the file.
    A, B : float
        Rrs<->rrs conversion constants used during the fit. Recorded so
        downstream code can verify they match its own convention.
    """
    import pandas as pd

    df = pd.DataFrame({
        'wavelength': result['wavelength'],
        'G1': result['G1'],
        'G2': result['G2'],
        'G1_err': result['G1_err'],
        'G2_err': result['G2_err'],
        'rRMS': stats['rRMS'],
        'RMS': stats['RMS']
    })

    with open(filename, 'w') as f:
        f.write("# Gordon coefficients fitted from Loisel23 Hydrolight simulations\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Standard G1: {G1_STANDARD}\n")
        f.write(f"# Standard G2: {G2_STANDARD}\n")
        f.write(f"# Rrs<->rrs convention: A={A}, B={B}\n")
        if weight_mode is not None:
            f.write(f"# weight_mode: {weight_mode}\n")
        f.write("#\n")
        df.to_csv(f, index=False)

    print(f"Saved Gordon coefficients to: {filename}")


# =============================================================================
# Smoothness-regularized joint fit
# =============================================================================

def fit_gordon_smooth(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    alpha_G1: float = 1e6,
    alpha_G2: float = 1e4,
    weight_mode: str = 'relative',
    rel_floor: float = 1e-5,
    bounds_G1: Tuple[float, float] = (0.05, 0.15),
    bounds_G2: Tuple[float, float] = (-2.0, 0.2),
    init_result: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Joint fit of G1(λ), G2(λ) across all wavelengths with Tikhonov smoothness
    regularization on the second derivative w.r.t. wavelength.

    Minimizes
        Σ_{i,s} w_{is} (rrs_{is} - G1[i]·u_{is} - G2[i]·u_{is}²)²
        + α₁ Σ_i (G1[i+1] - 2 G1[i] + G1[i-1])²
        + α₂ Σ_i (G2[i+1] - 2 G2[i] + G2[i-1])²

    subject to box bounds on G1[i], G2[i].

    The smoothness penalty couples adjacent wavelengths, so wavelengths where
    the per-wavelength fit is weak (small/narrow u, e.g. red) borrow strength
    from neighbors that are better-constrained, suppressing the runaway G₂
    seen at long wavelengths and reducing high/low-bbp tail bias.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths in nm, shape (n_wave,). Must be roughly evenly spaced
        (the second-difference operator assumes uniform spacing).
    Rrs : np.ndarray
        Above-surface Rrs, shape (n_samples, n_wave).
    a, bb : np.ndarray
        Total absorption / backscattering, shape (n_samples, n_wave).
    alpha_G1, alpha_G2 : float
        Smoothness penalties. Larger -> smoother. G2 typically has more
        wavelength structure than G1, so its default penalty is smaller.
    weight_mode, rel_floor :
        Per-sample weighting; see fit_gordon_at_wavelength.
    bounds_G1, bounds_G2 : (min, max)
        Per-wavelength box bounds.
    init_result : dict, optional
        Initial guess from a per-wavelength fit. If None, computed internally.
    verbose : bool
        If True, print scipy.optimize.least_squares progress.

    Returns
    -------
    result : dict
        {'wavelength', 'G1', 'G2', 'G1_err', 'G2_err'}.
        Errors are returned as zeros (no covariance is computed for the
        smoothness-regularized solution).
    stats : dict
        {'rRMS', 'RMS'} per wavelength.
    """
    from scipy.optimize import least_squares

    N = len(wave)
    u = bb / (a + bb)
    rrs = Rrs / (A_RRS + B_RRS * Rrs)

    if weight_mode == 'relative':
        sigma = np.maximum(np.abs(rrs), rel_floor)
    elif weight_mode == 'absolute':
        sigma = np.full_like(rrs, 3e-4)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

    # Per-wavelength bounded fit as a warm start
    if init_result is None:
        init_result = fit_gordon_coefficients(
            wave, Rrs, a, bb,
            return_stats=False,
            weight_mode=weight_mode,
            bounds=([bounds_G1[0], bounds_G2[0]], [bounds_G1[1], bounds_G2[1]]),
        )
    p0 = np.concatenate([init_result['G1'], init_result['G2']])

    sqrt_a1 = np.sqrt(alpha_G1)
    sqrt_a2 = np.sqrt(alpha_G2)

    def residuals(params):
        G1 = params[:N]
        G2 = params[N:]
        # Data residuals, normalized by sigma (shape: (n_samples, n_wave))
        pred = G1[None, :] * u + G2[None, :] * u ** 2
        r_data = ((rrs - pred) / sigma).ravel()
        # 2nd-derivative smoothness on G1(λ), G2(λ)
        d2_G1 = G1[2:] - 2.0 * G1[1:-1] + G1[:-2]
        d2_G2 = G2[2:] - 2.0 * G2[1:-1] + G2[:-2]
        r_smooth = np.concatenate([sqrt_a1 * d2_G1, sqrt_a2 * d2_G2])
        return np.concatenate([r_data, r_smooth])

    lower = np.concatenate([np.full(N, bounds_G1[0]), np.full(N, bounds_G2[0])])
    upper = np.concatenate([np.full(N, bounds_G1[1]), np.full(N, bounds_G2[1])])

    sol = least_squares(
        residuals, p0,
        bounds=(lower, upper),
        method='trf',
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=200,
        verbose=2 if verbose else 0,
    )

    G1 = sol.x[:N]
    G2 = sol.x[N:]

    # Per-wavelength rRMS for compatibility with the stats dict shape
    pred = G1[None, :] * u + G2[None, :] * u ** 2
    rRMS = np.sqrt(np.mean((rrs - pred) ** 2 / np.maximum(pred ** 2, 1e-20), axis=0))
    RMS = np.sqrt(np.mean((rrs - pred) ** 2, axis=0))

    result = {
        'wavelength': wave,
        'G1': G1,
        'G2': G2,
        'G1_err': np.zeros(N),
        'G2_err': np.zeros(N),
    }
    stats = {'rRMS': rRMS, 'RMS': RMS}
    return result, stats


# =============================================================================
# Performance assessment
# =============================================================================

def evaluate_gordon_on_dataset(
    wave: np.ndarray,
    Rrs_truth: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    G1_var: np.ndarray,
    G2_var: np.ndarray,
    G1_std: float = G1_STANDARD,
    G2_std: float = G2_STANDARD,
    G0_var: Optional[np.ndarray] = None,
    Gb_var: Optional[np.ndarray] = None,
    bbp: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compare variable and standard Gordon Rrs against a reference Rrs dataset.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths, shape (n_wave,).
    Rrs_truth : np.ndarray
        Reference Rrs (e.g. Hydrolight), shape (n_samples, n_wave).
    a, bb : np.ndarray
        Total absorption / backscattering, shape (n_samples, n_wave).
    G1_var, G2_var : np.ndarray
        Wavelength-dependent Gordon coefficients, shape (n_wave,).
    G1_std, G2_std : float
        Standard scalar coefficients.

    Returns
    -------
    dict
        Per-wavelength statistics:
        - wavelength
        - Rrs_var, Rrs_std : reconstructed Rrs arrays, shape (n_samples, n_wave)
        - bias_var, bias_std : mean signed residual (truth - pred)
        - rrms_var_pct, rrms_std_pct : relative RMS in percent
    """
    Rrs_var = calc_Rrs_with_variable_gordon(
        a, bb, G1_var, G2_var, G0=G0_var, Gb=Gb_var, bbp=bbp,
    )
    Rrs_std = calc_Rrs_with_variable_gordon(
        a, bb,
        np.full(len(wave), G1_std),
        np.full(len(wave), G2_std),
    )

    eps = 1e-12
    rel_var = (Rrs_truth - Rrs_var) / np.maximum(np.abs(Rrs_truth), eps)
    rel_std = (Rrs_truth - Rrs_std) / np.maximum(np.abs(Rrs_truth), eps)

    return {
        'wavelength': wave,
        'Rrs_var': Rrs_var,
        'Rrs_std': Rrs_std,
        'bias_var': np.mean(Rrs_truth - Rrs_var, axis=0),
        'bias_std': np.mean(Rrs_truth - Rrs_std, axis=0),
        'rrms_var_pct': 100.0 * np.sqrt(np.mean(rel_var ** 2, axis=0)),
        'rrms_std_pct': 100.0 * np.sqrt(np.mean(rel_std ** 2, axis=0)),
    }


# =============================================================================
# Figures for performance assessment
# =============================================================================

# All plotting helpers live in plot_gordon.py. Re-export them so the existing
# run_full_assessment driver below (and any external caller) continues to find
# them under their original names.
from plot_gordon import (  # noqa: E402
    plot_g_coefficients,
    plot_rrms_vs_wavelength,
    plot_rrms_vs_wavelength_3case,
    plot_rrms_vs_wavelength_4case,
    plot_rrms_vs_wavelength_5case,
    plot_residual_vs_bbp,
    plot_residual_vs_bbp_3case,
    plot_residual_vs_bbp_4case,
    plot_residual_vs_bbp_5case,
    plot_rrs_vs_u,
)


def run_full_assessment(
    weight_mode: str = 'relative',
    wv_min: float = 350.,
    wv_max: float = 750.,
    csv_out: str = 'gordon_coefficients.csv',
    figs_dir: str = 'figs',
    # Per-wavelength bounds. G2 lower bound extended to -2.0 (was -0.5) so the
    # quadratic term can take the strongly-negative values the data want at
    # red wavelengths; smoothness regularization (below) keeps it well-behaved.
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (0.05, -2.0), (0.15, 0.2)
    ),
    # Smoothness penalties for the joint Tikhonov fit on G1(λ), G2(λ).
    alpha_G1: float = 1e6,
    alpha_G2: float = 1e4,
    canonical: str = 'smooth',     # 'smooth' or 'perwave' -- which goes into the CSV
    clobber: bool = False,
):
    """
    End-to-end: fit Gordon coefficients on Loisel23, save CSV, save assessment figures.

    Runs all recipes side-by-side for diagnostics:
      - 'old'     : legacy per-wavelength fit, constant-sigma  (absolute σ).
      - 'perwave' : per-wavelength fit, relative σ, bounded.
      - 'smooth'  : joint G1(λ), G2(λ) fit with Tikhonov 2nd-derivative penalty.
      - 'const'   : 3-parameter (G0, G1, G2).
      - 'bbp'     : 3-parameter (G1, G2, Gb).
      - 'full'    : 4-parameter (G0, G1, G2, Gb).

    The fit named by ``canonical`` is written to CSV.

    Parameters
    ----------
    clobber : bool, default False
        If False (default), reuse coefficients from any already-existing CSV
        instead of refitting. The legacy and per-wavelength baselines are
        always recomputed (they're cheap and the per-wavelength fit is the
        warm-start for `smooth`); the smooth, const, bbp, and full fits load
        from their CSVs when present.
        If True, always refit and overwrite all CSVs.
    """
    from ocpy.hydrolight import loisel23

    os.makedirs(figs_dir, exist_ok=True)

    # Load Loisel23 elastic dataset
    ds = loisel23.load_ds(1, 0)
    wave_all = ds.Lambda.data
    Rrs_all  = ds.Rrs.data
    a_all    = ds.a.data
    bb_all   = ds.bb.data
    bbnw_all = ds.bbnw.data
    aph_all  = ds.aph.data

    gd = (wave_all >= wv_min) & (wave_all <= wv_max)
    wave = wave_all[gd]
    Rrs  = Rrs_all[:, gd]
    a    = a_all[:, gd]
    bb   = bb_all[:, gd]
    bbnw = bbnw_all[:, gd]
    aph  = aph_all[:, gd]

    # ------ (1) Per-wavelength fit, relative weighting, extended bounds ------
    result_pw, stats_pw = fit_gordon_coefficients(
        wave, Rrs, a, bb,
        return_stats=True,
        weight_mode=weight_mode,
        bounds=bounds,
    )

    # Paths for the cached CSVs. The canonical (smooth or perwave) lives at
    # csv_out; the other three are alongside it.
    csv_dir = os.path.dirname(csv_out) or '.'
    const_path = os.path.join(csv_dir, 'gordon_coefficients_with_const.csv')
    bbp_path   = os.path.join(csv_dir, 'gordon_coefficients_with_Gb.csv')
    full_path  = os.path.join(csv_dir, 'gordon_coefficients_with_G0_Gb.csv')

    # ------ (2) Smoothness-regularized joint fit (uses per-wave as warm start) ------
    bounds_G1 = (bounds[0][0], bounds[1][0])
    bounds_G2 = (bounds[0][1], bounds[1][1])
    if (not clobber) and canonical == 'smooth' and os.path.exists(csv_out):
        result_sm, stats_sm = load_gordon_csv(csv_out)
    else:
        result_sm, stats_sm = fit_gordon_smooth(
            wave, Rrs, a, bb,
            alpha_G1=alpha_G1, alpha_G2=alpha_G2,
            weight_mode=weight_mode,
            bounds_G1=bounds_G1, bounds_G2=bounds_G2,
            init_result=result_pw,
        )

    # ------ (3) Legacy (absolute-sigma) per-wavelength fit for diagnostics ------
    # No CSV for this -- always recompute (cheap; it's the legacy baseline).
    result_old, stats_old = fit_gordon_coefficients(
        wave, Rrs, a, bb,
        return_stats=True,
        weight_mode='absolute',
    )

    # ------ (4) 3-parameter (G0 + G1·u + G2·u²) per-wavelength fit ------
    # Allowing a small constant offset absorbs the non-pure-u dependence in
    # Hydrolight rrs(a, bb). At red wavelengths this drops rRMS by ~10x and
    # eliminates the residual-vs-bbp tilt.
    if (not clobber) and os.path.exists(const_path):
        result_c, stats_c = load_gordon_csv(const_path)
    else:
        result_c, stats_c = fit_gordon_const_coefficients(
            wave, Rrs, a, bb,
            return_stats=True,
            weight_mode=weight_mode,
        )
        save_gordon_const_to_csv(
            result_c, stats_c, const_path,
            weight_mode=weight_mode, A=A_RRS, B=B_RRS,
        )

    # ------ (5) 3-parameter (G1·u + G2·u² + Gb·bbp) per-wavelength fit ------
    # The constant offset (G0) form fails near 500 nm where the residual is
    # linear in bbp rather than constant. A Gb·bbp term captures this directly.
    if (not clobber) and os.path.exists(bbp_path):
        result_b, stats_b = load_gordon_csv(bbp_path)
    else:
        result_b, stats_b = fit_gordon_bbp_coefficients(
            wave, Rrs, a, bb, bbnw,
            return_stats=True,
            weight_mode=weight_mode,
        )
        save_gordon_bbp_to_csv(
            result_b, stats_b, bbp_path,
            weight_mode=weight_mode, A=A_RRS, B=B_RRS,
        )

    # ------ (6) 4-parameter (G0 + G1·u + G2·u² + Gb·bbp(700)) two-stage fit ------
    # Recipe (per user request, prior turn): Stage 1 fits G1, G2 from a
    # standard 2-parameter Gordon. Stage 2 fits G0, Gb to the Stage-1
    # residuals, using bbp(700) as a single trophic-state proxy independent
    # of wavelength. Decouples u-shape from (a, bb) offsets so G1/G2 don't
    # absorb bbp-driven residual structure.
    j_700 = int(np.argmin(np.abs(wave - 700.)))
    bbp700 = bbnw[:, j_700]
    if (not clobber) and os.path.exists(full_path):
        result_f, stats_f = load_gordon_csv(full_path)
    else:
        result_f, stats_f = fit_gordon_2stage_coefficients(
            wave, Rrs, a, bb, bbp700,
            return_stats=True,
            weight_mode=weight_mode,
        )
        save_gordon_full_to_csv(
            result_f, stats_f, full_path,
            source=("Loisel23 elastic; 4-parameter two-stage fit: "
                    "Stage1 (G1,G2); Stage2 (G0,Gb) on residuals vs bbp(700nm)"),
            weight_mode=weight_mode, A=A_RRS, B=B_RRS,
        )

    # Choose which 2-parameter result is canonical (written to CSV).
    canonical_result, canonical_stats, canonical_label = {
        'smooth':  (result_sm, stats_sm,
                    f'smooth (α_G1={alpha_G1:g}, α_G2={alpha_G2:g})'),
        'perwave': (result_pw, stats_pw, 'per-wavelength bounded'),
    }[canonical]
    # Only write the canonical CSV when we actually recomputed it. The smooth
    # branch above writes the smooth fit to csv_out; for canonical='perwave'
    # we write here unconditionally (its CSV path equals csv_out).
    if canonical == 'perwave':
        save_gordon_to_csv(canonical_result, canonical_stats, csv_out,
                       weight_mode=weight_mode, A=A_RRS, B=B_RRS,
                       source=f"Loisel23 elastic; fit recipe: {canonical_label}")
    elif clobber or not os.path.exists(csv_out):
        save_gordon_to_csv(canonical_result, canonical_stats, csv_out,
                       weight_mode=weight_mode, A=A_RRS, B=B_RRS,
                       source=f"Loisel23 elastic; fit recipe: {canonical_label}")

    # ------ Performance assessment against Hydrolight Rrs ------
    eval_pw  = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_pw['G1'], result_pw['G2'])
    eval_sm  = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_sm['G1'], result_sm['G2'])
    eval_old = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_old['G1'], result_old['G2'])
    eval_c   = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_c['G1'], result_c['G2'],
                                           G0_var=result_c['G0'])
    eval_b   = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_b['G1'], result_b['G2'],
                                           Gb_var=result_b['Gb'], bbp=bbnw)
    # 4-param fit uses bbp(700) as the trophic-state proxy at every wavelength.
    # Reshape to (n_samples, 1) so Gb·bbp broadcasts to (n_samples, n_wave).
    bbp700_col = bbnw[:, j_700:j_700 + 1]
    eval_f   = evaluate_gordon_on_dataset(wave, Rrs, a, bb, result_f['G1'], result_f['G2'],
                                           G0_var=result_f['G0'],
                                           Gb_var=result_f['Gb'], bbp=bbp700_col)

    # ------ Figures ------
    plot_g_coefficients(
        result_sm, outfile=os.path.join(figs_dir, 'g_coefficients.png'),
        compare=result_pw, compare_label='per-wave bounded',
    )
    plot_rrms_vs_wavelength(eval_pw,  outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_perwave.png'))
    plot_rrms_vs_wavelength(eval_sm,  outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_smooth.png'))
    plot_rrms_vs_wavelength(eval_old, outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_old.png'))
    plot_rrs_vs_u(
        wave, Rrs, a, bb, result_sm,
        outfile=os.path.join(figs_dir, 'rrs_vs_u_smooth.png'),
    )
    # 3-case rrs-vs-u, with 500 nm explicitly included to expose the G0
    # sign-change wavelength and the residual bbp dependence there.
    plot_rrs_vs_u(
        wave, Rrs, a, bb, result_sm,
        plot_waves=(370., 440., 500., 550., 600., 670.),
        result_with_G0=result_c,
        outfile=os.path.join(figs_dir, 'rrs_vs_u_3case.png'),
    )

    # Oligotrophic residual: aph(440) <= 0.015 (~ Chl <= 0.1 mg/m^3)
    j440 = int(np.argmin(np.abs(wave - 440.)))
    oligo = aph[:, j440] <= 0.015
    print(f"Oligotrophic scenes: {oligo.sum()} / {len(oligo)}")

    plot_residual_vs_bbp(
        wave, Rrs, eval_sm['Rrs_var'], eval_sm['Rrs_std'], bbnw,
        mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_smooth.png'),
    )
    plot_residual_vs_bbp(
        wave, Rrs, eval_pw['Rrs_var'], eval_pw['Rrs_std'], bbnw,
        mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_perwave.png'),
    )
    plot_residual_vs_bbp(
        wave, Rrs, eval_c['Rrs_var'], eval_c['Rrs_std'], bbnw,
        mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_const.png'),
    )
    plot_rrms_vs_wavelength(eval_c, outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_const.png'))

    # ------ Three-case comparison figures (requested explicitly) ------
    # Use the smooth 2-param fit as the "variable, no G0" reference and the
    # 3-parameter fit as "variable, with G0".
    plot_rrms_vs_wavelength_3case(
        eval_sm, eval_c,
        outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_3case.png'),
    )
    plot_residual_vs_bbp_3case(
        wave, Rrs,
        eval_sm['Rrs_var'], eval_c['Rrs_var'], eval_sm['Rrs_std'],
        bbnw, mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_3case.png'),
    )

    # ------ Four-case comparison figures (standard / noG0 / withG0 / withGb) ------
    plot_rrms_vs_wavelength_4case(
        eval_sm, eval_c, eval_b,
        outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_4case.png'),
    )
    plot_residual_vs_bbp_4case(
        wave, Rrs,
        eval_sm['Rrs_var'], eval_c['Rrs_var'], eval_b['Rrs_var'],
        eval_sm['Rrs_std'], bbnw, mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_4case.png'),
    )
    plot_rrs_vs_u(
        wave, Rrs, a, bb, result_sm,
        plot_waves=(370., 440., 500., 550., 600., 670.),
        result_with_G0=result_c,
        result_with_Gb=result_b, bbp=bbnw,
        outfile=os.path.join(figs_dir, 'rrs_vs_u_4case.png'),
    )

    # ------ Five-case rRMS + residual figures (adds the 4-parameter G0+Gb fit) ------
    plot_rrms_vs_wavelength_5case(
        eval_sm, eval_c, eval_b, eval_f,
        outfile=os.path.join(figs_dir, 'rrms_vs_wavelength_5case.png'),
    )
    plot_residual_vs_bbp_5case(
        wave, Rrs,
        eval_sm['Rrs_var'], eval_c['Rrs_var'], eval_b['Rrs_var'], eval_f['Rrs_var'],
        eval_sm['Rrs_std'], bbnw, mask=oligo,
        outfile=os.path.join(figs_dir, 'residual_vs_bbp_5case.png'),
    )

    # G0(λ), Gb(λ), and (G0,Gb)-from-4-param panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(result_c['wavelength'], result_c['G0'], 'C2-',  lw=2, label='G0 (3-param)')
    axes[0].plot(result_f['wavelength'], result_f['G0'], 'C5--', lw=2, label='G0 (4-param)')
    axes[0].axhline(0, color='k', lw=0.8, alpha=0.6)
    axes[0].set_xlabel('wavelength (nm)'); axes[0].set_ylabel(r'$G_0$')
    axes[0].set_title('Constant offset')
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(result_b['wavelength'], result_b['Gb'], 'C4-',  lw=2, label='Gb (3-param)')
    axes[1].plot(result_f['wavelength'], result_f['Gb'], 'C5--', lw=2, label='Gb (4-param)')
    axes[1].axhline(0, color='k', lw=0.8, alpha=0.6)
    axes[1].set_xlabel('wavelength (nm)'); axes[1].set_ylabel(r'$G_b$')
    axes[1].set_title('bbp slope')
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'G0_Gb_vs_wavelength.png'), dpi=150)
    print(f"Saved: {os.path.join(figs_dir, 'G0_Gb_vs_wavelength.png')}")

    # ------ Summary table at the canonical check wavelengths ------
    check_waves = [400., 500., 550., 600., 650., 700.]
    print("\nWavelength   rRMS_std%   rRMS_smooth%   rRMS_G0%   rRMS_Gb%   rRMS_full%")
    print("-" * 80)
    for wv in check_waves:
        j = int(np.argmin(np.abs(wave - wv)))
        print(f"  {wv:6.1f}     {eval_pw['rrms_std_pct'][j]:8.3f}    "
              f"{eval_sm['rrms_var_pct'][j]:11.3f}    "
              f"{eval_c['rrms_var_pct'][j]:7.3f}    "
              f"{eval_b['rrms_var_pct'][j]:7.3f}    "
              f"{eval_f['rrms_var_pct'][j]:9.3f}")

    print(f"\nG2 ranges:")
    print(f"  per-wave: [{result_pw['G2'].min():.4f}, {result_pw['G2'].max():.4f}]")
    print(f"  smooth:   [{result_sm['G2'].min():.4f}, {result_sm['G2'].max():.4f}]")
    print(f"  const:    [{result_c['G2'].min():.4f}, {result_c['G2'].max():.4f}]")
    print(f"  bbp:      [{result_b['G2'].min():.4f}, {result_b['G2'].max():.4f}]")
    print(f"  full:     [{result_f['G2'].min():.4f}, {result_f['G2'].max():.4f}]")
    print(f"G0 range (const fit): [{result_c['G0'].min():+.4e}, {result_c['G0'].max():+.4e}]")
    print(f"G0 range (full fit):  [{result_f['G0'].min():+.4e}, {result_f['G0'].max():+.4e}]")
    print(f"Gb range (bbp fit):   [{result_b['Gb'].min():+.4e}, {result_b['Gb'].max():+.4e}]")
    print(f"Gb range (full fit):  [{result_f['Gb'].min():+.4e}, {result_f['Gb'].max():+.4e}]")

    return {
        'wave': wave,
        'result_pw':  result_pw,  'stats_pw':  stats_pw,  'eval_pw':  eval_pw,
        'result_sm':  result_sm,  'stats_sm':  stats_sm,  'eval_sm':  eval_sm,
        'result_old': result_old, 'stats_old': stats_old, 'eval_old': eval_old,
        'result_c':   result_c,   'stats_c':   stats_c,   'eval_c':   eval_c,
        'result_b':   result_b,   'stats_b':   stats_b,   'eval_b':   eval_b,
        'result_f':   result_f,   'stats_f':   stats_f,   'eval_f':   eval_f,
    }


if __name__ == '__main__':
    print("Fitting wavelength-dependent Gordon coefficients from Loisel23 data...")
    run_full_assessment(clobber=False)

