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


def calc_Rrs(a, bb, in_G1:float|np.ndarray=None, in_G2:float|np.ndarray=None):
    """
    Calculates the remote sensing reflectance (Rrs) using the given absorption (a) and backscattering (bb) coefficients.

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
# Chlorophyll Fluorescence Functions
# Based on Gordon (1979) and Bricaud et al. (1995)
# =============================================================================

# Default mean cosine for fluorescence (isotropic emission)
MU_F_DEFAULT = 0.5

# Bricaud et al. (1995) coefficients for phytoplankton absorption
# a_ph(λ) = A(λ) * Chl^E(λ)
# These are approximate values at key wavelengths; full spectrum uses ocpy data
BRICAUD_COEFFS = {
    # wavelength: (A, E)
    400: (0.0654, 0.668),
    410: (0.0714, 0.668),
    420: (0.0763, 0.668),
    430: (0.0800, 0.667),
    440: (0.0654, 0.668),  # Blue peak
    450: (0.0590, 0.670),
    460: (0.0510, 0.673),
    470: (0.0430, 0.676),
    480: (0.0355, 0.680),
    490: (0.0290, 0.685),
    500: (0.0240, 0.690),
    510: (0.0200, 0.695),
    520: (0.0170, 0.700),
    530: (0.0145, 0.705),
    540: (0.0125, 0.710),
    550: (0.0110, 0.715),
    560: (0.0100, 0.720),
    570: (0.0092, 0.725),
    580: (0.0085, 0.730),
    590: (0.0080, 0.735),
    600: (0.0078, 0.740),
    620: (0.0085, 0.750),
    640: (0.0105, 0.760),
    660: (0.0200, 0.770),  # Red absorption
    675: (0.0260, 0.775),  # Red peak
    680: (0.0240, 0.776),
    690: (0.0180, 0.778),
    700: (0.0100, 0.780),
}

'''
def calc_a_ph_bricaud(
    wavelength: Union[float, np.ndarray],
    Chl: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate phytoplankton absorption using Bricaud et al. (1995) parameterization.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.
    Chl : float or ndarray
        Chlorophyll-a concentration in mg m^-3.

    Returns
    -------
    float or ndarray
        Phytoplankton absorption coefficient a_ph in m^-1.
        If both wavelength and Chl are arrays, returns shape (len(Chl), len(wavelength)).

    Notes
    -----
    The Bricaud model parameterizes phytoplankton absorption as:
        a_ph(λ, Chl) = A(λ) * Chl^E(λ)

    where A(λ) and E(λ) are wavelength-dependent coefficients derived from
    measurements of diverse phytoplankton populations.
    """
    wavelength = np.atleast_1d(wavelength)
    Chl = np.atleast_1d(Chl)

    # Get reference wavelengths and coefficients
    ref_waves = np.array(sorted(BRICAUD_COEFFS.keys()))
    A_vals = np.array([BRICAUD_COEFFS[w][0] for w in ref_waves])
    E_vals = np.array([BRICAUD_COEFFS[w][1] for w in ref_waves])

    # Interpolate to requested wavelengths
    # Use boundary values for extrapolation (more physically realistic than linear extrapolation)
    f_A = interpolate.interp1d(ref_waves, A_vals, kind='linear',
                               bounds_error=False, fill_value=(A_vals[0], A_vals[-1]))
    f_E = interpolate.interp1d(ref_waves, E_vals, kind='linear',
                               bounds_error=False, fill_value=(E_vals[0], E_vals[-1]))

    A = f_A(wavelength)
    E = f_E(wavelength)

    # Calculate a_ph for each Chl value
    if Chl.size == 1:
        a_ph = A * Chl[0]**E
    else:
        # Output shape: (len(Chl), len(wavelength))
        a_ph = np.zeros((len(Chl), len(wavelength)))
        for i, chl in enumerate(Chl):
            a_ph[i, :] = A * chl**E

    # Ensure non-negative values (absorption cannot be negative)
    a_ph = np.maximum(a_ph, 0.0)

    return np.squeeze(a_ph)


def calc_a_water(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate pure water absorption coefficient.

    Uses Pope & Fry (1997) and Smith & Baker (1981) data.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.

    Returns
    -------
    float or ndarray
        Pure water absorption coefficient a_w in m^-1.
    """
    wavelength = np.atleast_1d(wavelength)

    # Reference data: Pope & Fry (1997) and Smith & Baker (1981)
    # Approximate values at key wavelengths
    ref_waves = np.array([
        350, 375, 400, 425, 450, 475, 500, 525, 550, 575,
        600, 625, 650, 675, 700, 725, 750
    ])
    ref_a_w = np.array([
        0.0204, 0.0156, 0.0106, 0.0093, 0.0094, 0.0150, 0.0267, 0.0430, 0.0593, 0.0960,
        0.2440, 0.3140, 0.3490, 0.4290, 0.6240, 1.2700, 2.4700
    ])

    f_aw = interpolate.interp1d(ref_waves, ref_a_w, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

    return np.squeeze(f_aw(wavelength))


def calc_bb_water(wavelength: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate pure water backscattering coefficient.

    Uses Morel (1974) wavelength dependence: bb_w(λ) = bb_w(500) * (500/λ)^4.32

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.

    Returns
    -------
    float or ndarray
        Pure water backscattering coefficient bb_w in m^-1.
    """
    wavelength = np.atleast_1d(wavelength)

    # Reference value at 500 nm
    bb_w_500 = 0.00144  # m^-1

    bb_w = bb_w_500 * (500.0 / wavelength)**4.32

    return np.squeeze(bb_w)
'''

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


def calc_Rrs_with_fluorescence(
    wavelength: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    in_G1: Optional[float] = None,
    in_G2: Optional[float] = None,
    mu_d: Optional[float] = None,
    mu_f: Optional[float] = None,
    double_gaussian: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate total Rrs including elastic scattering and fluorescence.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.
    a : float or ndarray
        Total absorption coefficient [m^-1].
    bb : float or ndarray
        Total backscattering coefficient [m^-1].
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    in_G1, in_G2 : float, optional
        Gordon coefficients. If None, use defaults.
    mu_d : float, optional
        Mean cosine for downwelling irradiance. Default is 0.9.
    mu_f : float, optional
        Mean cosine for fluorescence emission. Default is 0.5.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.

    Returns
    -------
    float or ndarray
        Total Rrs (elastic + fluorescence) in sr^-1.

    Examples
    --------
    >>> wavelength = np.linspace(400, 750, 100)
    >>> a = 0.1 * np.ones_like(wavelength)  # Simplified
    >>> bb = 0.002 * np.ones_like(wavelength)
    >>> Rrs_total = calc_Rrs_with_fluorescence(wavelength, a, bb, Chl=1.0)
    """
    # Elastic Rrs (Gordon model)
    Rrs_elastic = calc_Rrs(a, bb, in_G1, in_G2)

    # Fluorescence Rrs
    Rrs_fl = calc_Rrs_fluorescence_simple(
        wavelength, Chl, phi_C,
        mu_d=mu_d, mu_f=mu_f,
        double_gaussian=double_gaussian
    )

    return Rrs_elastic + Rrs_fl


def calc_fluorescence_spectrum(
    Chl: Union[float, np.ndarray],
    wavelength: Optional[np.ndarray] = None,
    phi_C: float = 0.02,
    double_gaussian: bool = False,
    return_components: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Calculate the full fluorescence Rrs spectrum for given Chl concentrations.

    This is a convenience function that returns the fluorescence spectrum
    across the emission range (typically 650-750 nm).

    Parameters
    ----------
    Chl : float or ndarray
        Chlorophyll-a concentration(s) in mg m^-3.
    wavelength : ndarray, optional
        Emission wavelengths in nm. If None, uses 650-750 nm at 1 nm resolution.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    double_gaussian : bool, optional
        If True, use double Gaussian emission. Default is False.
    return_components : bool, optional
        If True, also return component spectra. Default is False.

    Returns
    -------
    Rrs_fl : ndarray
        Fluorescence Rrs spectrum. Shape is (len(Chl), len(wavelength)) if
        Chl is an array, otherwise (len(wavelength),).
    components : dict, optional
        If return_components=True, dictionary containing:
        - 'wavelength': wavelength array
        - 'emission_shape': normalized emission line shape
        - 'a_water': water absorption at emission wavelengths
        - 'a_ph': phytoplankton absorption at emission wavelengths

    Examples
    --------
    >>> Chl_values = [0.1, 1.0, 10.0]  # mg m^-3
    >>> Rrs_fl = calc_fluorescence_spectrum(Chl_values)
    >>> print(f"Shape: {Rrs_fl.shape}")  # (3, 101)

    >>> wavelength, Rrs_fl, components = calc_fluorescence_spectrum(
    ...     1.0, return_components=True)
    """
    from . import chl_fl

    if wavelength is None:
        wavelength = np.arange(650, 751, 1.0)

    Chl = np.atleast_1d(Chl)

    # Output array
    Rrs_fl = np.zeros((len(Chl), len(wavelength)))

    for i, chl in enumerate(Chl):
        Rrs_fl[i, :] = calc_Rrs_fluorescence_simple(
            wavelength, chl, phi_C, double_gaussian=double_gaussian
        )

    Rrs_fl = np.squeeze(Rrs_fl)

    if return_components:
        # Get emission line shape
        if double_gaussian:
            emission_shape = chl_fl.emission_line_double_gaussian(wavelength)
        else:
            emission_shape = chl_fl.emission_line_single_gaussian(wavelength)

        components = {
            'wavelength': wavelength,
            'emission_shape': emission_shape,
            'a_water': calc_a_water(wavelength),
            'a_ph': calc_a_ph_bricaud(wavelength, Chl[0] if len(Chl) == 1 else 1.0),
        }
        return wavelength, Rrs_fl, components

    return Rrs_fl


def calc_fluorescence_correction_factor(
    wavelength: Union[float, np.ndarray],
    a: Union[float, np.ndarray],
    bb: Union[float, np.ndarray],
    Chl: float,
    phi_C: float = 0.02,
    in_G1: Optional[float] = None,
    in_G2: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Calculate the multiplicative correction factor for fluorescence.

    This returns the ratio of total Rrs (elastic + fluorescence) to elastic Rrs,
    useful for understanding the fluorescence contribution.

    Parameters
    ----------
    wavelength : float or ndarray
        Wavelength(s) in nanometers.
    a : float or ndarray
        Total absorption coefficient [m^-1].
    bb : float or ndarray
        Total backscattering coefficient [m^-1].
    Chl : float
        Chlorophyll-a concentration in mg m^-3.
    phi_C : float, optional
        Fluorescence quantum yield. Default is 0.02.
    in_G1, in_G2 : float, optional
        Gordon coefficients. If None, use defaults.

    Returns
    -------
    float or ndarray
        Correction factor: (Rrs_elastic + Rrs_fluorescence) / Rrs_elastic

    Notes
    -----
    Values > 1 indicate wavelengths where fluorescence adds signal.
    The correction is typically largest near 685 nm (the fluorescence peak)
    and approaches 1 at wavelengths away from the emission band.
    """
    Rrs_elastic = calc_Rrs(a, bb, in_G1, in_G2)

    Rrs_fl = calc_Rrs_fluorescence_simple(wavelength, Chl, phi_C)

    return (Rrs_elastic + Rrs_fl) / Rrs_elastic