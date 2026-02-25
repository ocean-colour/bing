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

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, Tuple, Dict, Union


# Conversion factors from rrs to Rrs (Lee et al. 2002)
A_RRS = 0.52
B_RRS = 1.17  # Used in fig_u analysis (differs from the 1.7 in some literature)

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
    p0: Tuple[float, float] = (0.1, 0.1)
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
        Uncertainties in rrs for weighted fitting.
    p0 : tuple
        Initial guess for (G1, G2).

    Returns
    -------
    params : np.ndarray
        Fitted parameters [G1, G2].
    cov : np.ndarray
        Covariance matrix from the fit.
    """
    if sigma is None:
        sigma = np.ones_like(u) * 0.0003  # Default small uncertainty

    params, cov = curve_fit(rrs_model, u, rrs, p0=p0, sigma=sigma)
    return params, cov


def fit_gordon_coefficients(
    wave: np.ndarray,
    Rrs: np.ndarray,
    a: np.ndarray,
    bb: np.ndarray,
    wave_select: Optional[np.ndarray] = None,
    sigma_rrs: Optional[np.ndarray] = None,
    return_stats: bool = False
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
        Uncertainties in rrs, shape (n_samples, n_wave).
    return_stats : bool
        If True, return fitting statistics (RMS errors).

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
        params, cov = fit_gordon_at_wavelength(u_wv, rrs_wv, sigma=sig_wv)
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
    return_stats: bool = False
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
        return_stats=return_stats
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
    G2: np.ndarray
) -> np.ndarray:
    """
    Calculate Rrs using wavelength-dependent Gordon coefficients.

    Parameters
    ----------
    a : np.ndarray
        Total absorption coefficient, shape (..., n_wave).
    bb : np.ndarray
        Total backscattering coefficient, shape (..., n_wave).
    G1 : np.ndarray
        First-order Gordon coefficient, shape (n_wave,).
    G2 : np.ndarray
        Second-order Gordon coefficient, shape (n_wave,).

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs.
    """
    u = calc_u(a, bb)
    rrs = G1 * u + G2 * u**2
    Rrs = rrs_to_Rrs(rrs)
    return Rrs


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
    source: str = "Loisel et al. (2023) synthetic dataset"
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
    """
    import pandas as pd

    # Create DataFrame with results
    df = pd.DataFrame({
        'wavelength': result['wavelength'],
        'G1': result['G1'],
        'G2': result['G2'],
        'G1_err': result['G1_err'],
        'G2_err': result['G2_err'],
        'rRMS': stats['rRMS'],
        'RMS': stats['RMS']
    })

    # Write to CSV with metadata as header comments
    with open(filename, 'w') as f:
        f.write(f"# Gordon coefficients fitted from Loisel23 Hydrolight simulations\n")
        f.write(f"# Source: {source}\n")
        f.write(f"# Standard G1: {G1_STANDARD}\n")
        f.write(f"# Standard G2: {G2_STANDARD}\n")
        f.write("#\n")
        # Write the DataFrame to CSV
        df.to_csv(f, index=False)

    print(f"Saved Gordon coefficients to: {filename}")


if __name__ == '__main__':
    # Example usage
    print("Fitting wavelength-dependent Gordon coefficients from Loisel23 data...")

    # Fit at selected wavelengths
    wave_select = np.array([370., 440., 500., 600.])
    result, stats = fit_gordon_from_loisel23(
        wave_select=wave_select,
        return_stats=True
    )

    print("\nResults:")
    print_gordon_comparison(result)

    print("\nFitting statistics (rRMS * 10):")
    for ii, wv in enumerate(result['wavelength']):
        print(f"  {wv:.0f} nm: {10*stats['rRMS'][ii]:.4f}")

    # Fit all wavelengths and save to CSV
    print("\n\nFitting all wavelengths (350-750 nm)...")
    result_full, stats_full = fit_gordon_from_loisel23(
        wv_min=350.,
        wv_max=750.,
        return_stats=True
    )

    # Save to CSV
    save_gordon_to_csv(result_full, stats_full, 'gordon_coefficients.csv')

    print(f"\nFitted {len(result_full['wavelength'])} wavelengths")
    print(f"Wavelength range: {result_full['wavelength'].min():.0f} - {result_full['wavelength'].max():.0f} nm")
