"""
Radiation Transfer methods for BING
====================================

This module implements radiative transfer models for computing remote sensing
reflectance (Rrs) from inherent optical properties, including:

1. Standard Gordon (1988) elastic scattering model
2. Raman scattering corrections based on Sathyendranath & Platt (1998)
3. Chlorophyll fluorescence contribution based on Gordon (1979)

References
----------
- Gordon, H.R. et al. (1988). "A semianalytic radiance model of ocean color,"
  J. Geophys. Res. 93, 10909-10924.
- Sathyendranath, S. and Platt, T. (1998). "Ocean-color model incorporating
  transspectral processes," Appl. Opt. 37, 2216-2227.
- Gordon, H.R. (1979). "Diffuse reflectance of the ocean: the theory of its
  augmentation by chlorophyll a fluorescence at 685 nm," Appl. Opt. 18, 1161-1166.
- Bricaud, A. et al. (1995). "Variability in the chlorophyll-specific absorption
  coefficients of natural phytoplankton," J. Geophys. Res. 100, 13321-13332.
"""

import os
from importlib import resources
import numpy as np
import pandas
from typing import Union, Optional, Tuple
from scipy import interpolate 

from bing.rt import raman

# Conversion from rrs to Rrs
A_Rrs, B_Rrs = 0.52, 1.7

# Gordon factors
G1_STANDARD, G2_STANDARD = 0.0949, 0.0794  # Standard Gordon factors



def wave_dependent_gordon(wave:np.ndarray, bounds_error:bool=True):
    """
    Load and interpolate wavelength-dependent Gordon coefficients G1 and G2.

    The Gordon coefficients parameterize the relationship between inherent
    optical properties (IOPs) and remote sensing reflectance. Wavelength-dependent
    coefficients provide improved accuracy over constant values, especially in
    the UV and red wavelength ranges.

    Parameters
    ----------
    wave : np.ndarray
        Wavelengths in nanometers at which to interpolate the Gordon coefficients.
    bounds_error : bool, optional
        If True (default), raises an error if wavelengths are outside the
        tabulated range. If False, extrapolates using cubic spline.

    Returns
    -------
    G1 : np.ndarray
        First-order Gordon coefficient G₀ at each wavelength.
    G2 : np.ndarray
        Second-order Gordon coefficient G₁ at each wavelength.

    Notes
    -----
    The coefficients are loaded from 'bing/data/RT/gordon_coefficients.csv'
    and interpolated using cubic splines.

    See Also
    --------
    calc_elastic_Rrs : Uses these coefficients to compute Rrs from IOPs.
    """
    # Load
    gordon_file = os.path.join(
            resources.files('bing'), 
            'data', 'RT', 'gordon_coefficients.csv')
    result = pandas.read_csv(gordon_file, comment='#')

    # Interpolate
    f_G1 = interpolate.interp1d(result['wavelength'], result['G1'], kind=3,
                        bounds_error=bounds_error)#, fill_value='extrapolate')
    f_G2 = interpolate.interp1d(result['wavelength'], result['G2'], kind=3,
                        bounds_error=bounds_error)#, fill_value='extrapolate')

    # Apply                    
    return f_G1(wave), f_G2(wave)


def Rrs_to_rrs(Rrs: np.ndarray, A: float = A_Rrs, B: float = B_Rrs) -> np.ndarray:
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


def rrs_to_Rrs(rrs: np.ndarray, A: float = A_Rrs, B: float = B_Rrs) -> np.ndarray:
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

def calc_Rrs(a, bb, in_G1:float|np.ndarray=None, in_G2:float|np.ndarray=None,
    a_ex: Union[float, np.ndarray]=None,
    bb_ex: Union[float, np.ndarray]=None,
    bb_R: Union[float, np.ndarray]=None,
    ):
    """
    Calculate remote sensing reflectance (Rrs) including optional Raman correction.

    This is the main Rrs calculation function that combines elastic scattering
    (Gordon model) with an optional Raman scattering correction based on
    Sathyendranath & Platt (1998).

    Parameters
    ----------
    a : float or np.ndarray
        Total absorption coefficient at emission wavelength(s) [m^-1].
    bb : float or np.ndarray
        Total backscattering coefficient at emission wavelength(s) [m^-1].
    in_G1 : float or np.ndarray, optional
        First-order Gordon coefficient. If None, uses default (0.0949).
    in_G2 : float or np.ndarray, optional
        Second-order Gordon coefficient. If None, uses default (0.0794).
    a_ex : float or np.ndarray, optional
        Absorption coefficient at Raman excitation wavelength(s) [m^-1].
        Required for Raman correction.
    bb_ex : float or np.ndarray, optional
        Backscattering coefficient at Raman excitation wavelength(s) [m^-1].
        Required for Raman correction.
    bb_R : float or np.ndarray, optional
        Raman backscattering coefficient [m^-1].
        Required for Raman correction. Can be computed using
        bing.rt.raman.raman_backscattering_coeff().

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs [sr^-1].

    Raises
    ------
    IOError
        If a_ex is provided but bb_ex or bb_R are not.

    Notes
    -----
    When Raman parameters (a_ex, bb_ex, bb_R) are provided, the elastic Rrs
    is multiplied by a correction factor that accounts for the Raman scattering
    contribution. This correction is typically 1.0-1.25, with largest values
    in clear oligotrophic waters at longer wavelengths.

    Examples
    --------
    >>> # Elastic-only calculation
    >>> Rrs = calc_Rrs(a=0.05, bb=0.002)
    >>>
    >>> # With Raman correction
    >>> from bing.rt import raman
    >>> bb_R = raman.raman_backscattering_coeff(wave_ex)
    >>> Rrs = calc_Rrs(a, bb, a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

    See Also
    --------
    calc_elastic_Rrs : Calculate elastic-only Rrs.
    calc_raman_correction_factor : Compute the Raman correction factor.
    """
    # Elastic
    Rrs = calc_elastic_Rrs(a, bb, in_G1=in_G1, in_G2=in_G2)

    # Raman?
    if a_ex is not None:
        if bb_ex is None or bb_R is None:
            raise IOError("bb_ex/bb_R must be set if a_ex is provided")
        corr = calc_raman_correction_factor(a, bb, a_ex, bb_ex, bb_R)
        # Apply
        Rrs *= corr

    # Return
    return Rrs



def calc_elastic_Rrs(a, bb, in_G1:float|np.ndarray=None, in_G2:float|np.ndarray=None):
    """
    Calculates the remote sensing reflectance (Rrs) using 
    the given absorption (a) and backscattering (bb) coefficients.

    Parameters:
        a (float or array-like): Absorption coefficient.
        bb (float or array-like): Backscattering coefficient.
        in_G1 (float or array-like, optional): G1 value. Default is None.
        in_G2 (float or array-like, optional): G2 value. Default is None.

    Returns:
        float or array-like: Remote Sensing Reflectance (Rrs) value.
    """
    # u
    u = bb / (a+bb)
    # rrs
    if in_G1 is not None:
        t1 = in_G1 * u
    else: 
        t1 = G1_STANDARD * u
    if in_G2 is not None:
        t2 = in_G2 * u*u
    else:
        t2 = G2_STANDARD * u*u
    rrs = t1 + t2
    
    # Return Rrs
    return rrs_to_Rrs(rrs)


# =============================================================================
# Raman Scattering Correction Functions
# Based on Sathyendranath & Platt (1998), Applied Optics 37, 2216-2227
# =============================================================================
# NOTE: Most Raman scattering functions have been moved to bing.rt.raman module.
# This section now contains only the correction factor function that bridges
# between the Gordon model (in rrs.py) and Raman scattering (in raman.py).
# For detailed Raman scattering calculations, see bing.rt.raman.


def calc_raman_correction_factor(
    a_em: Union[float, np.ndarray],
    bb_em: Union[float, np.ndarray],
    a_ex: Union[float, np.ndarray],
    bb_ex: Union[float, np.ndarray],
    bb_R: Union[float, np.ndarray],
    Ed_ratio: Union[float, np.ndarray] = 1.0,
    s_E: float = 1.0,
    mu_d: Optional[float] = None,
    mu_u: Optional[float] = None,
    mu_R: Optional[float] = None,
    include_second_order: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate the multiplicative correction factor for Raman scattering.

    This returns the ratio of total reflectance (with Raman) to elastic
    reflectance, useful for correcting Rrs retrievals.

    Parameters
    ----------
    a_em : float or ndarray
        Absorption coefficient at emission wavelength λ [m^-1].
    bb_em : float or ndarray
        Elastic backscattering coefficient at emission wavelength λ [m^-1].
    a_ex : float or ndarray
        Absorption coefficient at excitation wavelength λ' [m^-1].
    bb_ex : float or ndarray
        Elastic backscattering coefficient at excitation wavelength λ' [m^-1].
    bb_R : float or ndarray
        Raman backscattering coefficient at λ' [m^-1].
        Can be computed using bing.rt.raman.raman_backscattering_coeff().
    Ed_ratio : float or ndarray
        Ratio Ed(λ')/Ed(λ). Default is 1.0.
    s_E : float
        Shape factor for elastic scattering.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. If None, uses default from raman module.
    mu_u : float, optional
        Mean cosine for upwelling irradiance. If None, uses default from raman module.
    mu_R : float, optional
        Mean cosine for Raman-scattered light. If None, uses default from raman module.
    include_second_order : bool
        If True, include second-order Raman terms. Default is True.

    Returns
    -------
    float or ndarray
        Correction factor: (R^E + R_raman) / R^E

    Notes
    -----
    To remove Raman contribution from measured reflectance:
        R_elastic = R_measured / correction_factor

    Typical correction factors range from 1.0 to ~1.25, with largest
    corrections in clear (oligotrophic) waters at longer wavelengths.

    This function now calls bing.rt.raman for the detailed Raman calculations.

    Examples
    --------
    >>> from bing.rt import raman
    >>> a_520, bb_520 = 0.05, 0.002
    >>> a_443, bb_443 = 0.03, 0.003
    >>> bb_R = raman.raman_backscattering_coeff(443)
    >>> correction = calc_raman_correction_factor(a_520, bb_520, a_443, bb_443, bb_R)
    """

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_u is None:
        mu_u = raman.MU_U_DEFAULT
    if mu_R is None:
        mu_R = raman.MU_R_DEFAULT

    # Calculate elastic reflectance
    R_E = raman.calc_R_elastic(a_em, bb_em, s_E, mu_d, mu_u)

    # Calculate total Raman reflectance
    R_raman = raman.calc_R_raman_total(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio,
        s_E, mu_d, mu_u, mu_R, include_second_order
    )

    return (R_E + R_raman) / R_E

# =============================================================================
# Chl Fluorescence Functions
# =============================================================================

# Default mean cosine for fluorescence (isotropic emission)
MU_F_DEFAULT = 0.5

def calc_Rrs_fluorescence(
    wavelength: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    wavelength_ex: Optional[Union[float, np.ndarray]] = None,
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate Rrs contribution from chlorophyll fluorescence as a function of wavelength.

    This function computes the fluorescence contribution to remote sensing reflectance
    integrating over excitation wavelengths and using the Bricaud parameterization
    for phytoplankton absorption.

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) λ in nanometers. Typically in range 650-750 nm.
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield (0-1). Default is 0.02.
    wavelength_ex : float or ndarray, optional
        Excitation wavelength(s) for integration. If None, uses 400-680 nm range.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission (isotropic). Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission (685 + 730 nm peaks). Default is False.

    Returns
    -------
    float or ndarray
        Fluorescence contribution to Rrs in sr^-1.

    Notes
    -----
    The calculation integrates over excitation wavelengths:

    Rrs_fl(λ) = ∫ h_C(λ) × Ed(λ') × (λ'/λ) × [b_bF(λ')/μ_d] / [K(λ') + κ_F(λ)] dλ'

    where:
    - h_C(λ) is the fluorescence emission line shape (Gaussian at 685 nm)
    - Ed(λ') is the downwelling irradiance spectrum (assumed flat here)
    - b_bF = 0.5 × Φ_C × a_ph is the fluorescence backscattering coefficient
    - K and κ_F are attenuation coefficients

    Examples
    --------
    >>> wavelengths = np.linspace(650, 750, 100)
    >>> Rrs_fl = calc_Rrs_fluorescence(wavelengths, Chl=1.0)
    >>> print(f"Peak fluorescence Rrs: {Rrs_fl.max():.2e} sr^-1")
    """
    from . import chl_fl, raman

    wavelength = np.atleast_1d(wavelength)

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_f is None:
        mu_f = 0.5  # Default for fluorescence (isotropic)

    # Set up excitation wavelength grid if not provided
    if wavelength_ex is None:
        wavelength_ex = np.arange(400, 681, 5)  # 5 nm resolution
    else:
        wavelength_ex = np.atleast_1d(wavelength_ex)

    # Calculate IOPs at excitation wavelengths
    a_ph_ex = calc_a_ph_bricaud(wavelength_ex, Chl)
    a_w_ex = calc_a_water(wavelength_ex)
    bb_w_ex = calc_bb_water(wavelength_ex)

    # Total absorption and backscattering at excitation
    a_ex = a_w_ex + a_ph_ex
    bb_ex = bb_w_ex  # Assume particle backscatter is small for open ocean

    # Calculate IOPs at emission wavelengths
    a_ph_em = calc_a_ph_bricaud(wavelength, Chl)
    a_w_em = calc_a_water(wavelength)
    bb_w_em = calc_bb_water(wavelength)

    a_em = a_w_em + a_ph_em
    bb_em = bb_w_em

    # Initialize output
    Rrs_fl = np.zeros_like(wavelength, dtype=float)

    # Calculate fluorescence emission line shape at each emission wavelength
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    # Upwelling attenuation at emission wavelengths
    kappa_F_em = (a_em + bb_em) / mu_f

    # Integrate over excitation wavelengths
    for i, lambda_em in enumerate(wavelength):
        if h_C[i] < 1e-12:
            continue

        # For each excitation wavelength, calculate contribution
        integrand = np.zeros(len(wavelength_ex))

        for j, lambda_ex in enumerate(wavelength_ex):
            # Skip wavelengths outside valid excitation range
            if lambda_ex < chl_fl.LAMBDA_EX_MIN or lambda_ex > chl_fl.LAMBDA_EX_MAX:
                continue

            # Fluorescence backscattering coefficient at excitation wavelength
            bb_F = chl_fl.fluorescence_backscattering_coeff(a_ph_ex[j], phi_C)

            # Downwelling attenuation at excitation wavelength
            K_ex = (a_ex[j] + bb_ex[j]) / mu_d

            # Wavelength ratio (energy conversion)
            lambda_ratio = lambda_ex / lambda_em

            # Contribution (assume flat Ed spectrum, Ed_ratio = 1)
            integrand[j] = h_C[i] * lambda_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em[i])

        # Integrate using trapezoidal rule
        if len(wavelength_ex) > 1:
            R_F = np.trapz(integrand, wavelength_ex)
        else:
            R_F = integrand[0]

        # Convert subsurface reflectance to Rrs
        Rrs_fl[i] = A_Rrs * R_F / (1 - B_Rrs * R_F)

    return np.squeeze(Rrs_fl)


def calc_Rrs_fluorescence_simple(
    wavelength: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    lambda_ex: float = 440.0,
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate Rrs fluorescence contribution using simplified single-excitation model.

    This is a faster version that assumes all excitation occurs at a single
    wavelength (typically 440 nm, the blue absorption peak).

    Parameters
    ----------
    wavelength : float or ndarray
        Emission wavelength(s) λ in nanometers.
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    lambda_ex : float, optional
        Excitation wavelength in nm. Default is 440 nm.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission. Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.

    Returns
    -------
    float or ndarray
        Fluorescence contribution to Rrs in sr^-1.

    Examples
    --------
    >>> wavelengths = np.linspace(650, 750, 100)
    >>> Rrs_fl = calc_Rrs_fluorescence_simple(wavelengths, Chl=1.0)
    >>> peak_idx = np.argmax(Rrs_fl)
    >>> print(f"Peak at {wavelengths[peak_idx]:.0f} nm: {Rrs_fl[peak_idx]:.2e} sr^-1")
    """
    from . import chl_fl, raman

    wavelength = np.atleast_1d(wavelength)

    # Use default mean cosines if not provided
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT
    if mu_f is None:
        mu_f = 0.5  # Default for fluorescence (isotropic)

    # IOPs at excitation wavelength
    a_ph_ex = calc_a_ph_bricaud(lambda_ex, Chl)
    a_w_ex = calc_a_water(lambda_ex)
    bb_w_ex = calc_bb_water(lambda_ex)

    a_ex = a_w_ex + a_ph_ex
    bb_ex = bb_w_ex

    # IOPs at emission wavelengths
    a_ph_em = calc_a_ph_bricaud(wavelength, Chl)
    a_w_em = calc_a_water(wavelength)
    bb_w_em = calc_bb_water(wavelength)

    a_em = a_w_em + a_ph_em
    bb_em = bb_w_em

    # Fluorescence backscattering coefficient
    bb_F = chl_fl.fluorescence_backscattering_coeff(a_ph_ex, phi_C)

    # Attenuation coefficients
    K_ex = (a_ex + bb_ex) / mu_d
    kappa_F_em = (a_em + bb_em) / mu_f

    # Emission line shape
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    # Wavelength ratio
    lambda_ratio = lambda_ex / wavelength

    # Subsurface fluorescence reflectance
    R_F = h_C * lambda_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em)

    # Convert to Rrs
    Rrs_fl = A_Rrs * R_F / (1 - B_Rrs * R_F)

    return np.squeeze(Rrs_fl)